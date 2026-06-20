# metarcwa/solver/tvf/tvf_utils.py
# Utility functions for TVF (Tangent Vector Field) computation.
#
# All functions accept plain torch.Tensor / Python types. 
# Device and dtype are inherited from
# input tensors; no explicit .to() / .cuda() calls appear here.
#
# Grid convention (axis ↔ lattice vector)
# ----------------------------------------
# The permittivity field ``s`` is assumed to be sampled on the fractional
# unit-cell parallelogram grid: sample ``(i/D0, j/D1)`` is mapped to
# Cartesian position ``f2*a2 + f1*a1`` (where ``f2 = i/D0``, ``f1 = j/D1``).
#
#   axis -2  (size D0)  ↔  a2 direction  (fractional coord f2)
#   axis -1  (size D1)  ↔  a1 direction  (fractional coord f1)
#
# In the centred Fourier domain, axis -2 indexes the a2-direction harmonic
# order n and axis -1 indexes the a1-direction harmonic order m.

import math
from typing import Tuple
import torch

from metarcwa.solver.harmonics import reciprocal_lattice_vectors


# ---------------------------------------------------------------------------
# Gradient computation for 2-D scalar fields
# ---------------------------------------------------------------------------

def _grad_periodic(s: torch.Tensor,
                   a1: torch.Tensor,
                   a2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Cartesian gradient of a 2-D scalar field with periodic boundary conditions.

    The field is sampled on the fractional parallelogram grid spanned by
    ``a1`` (axis -1) and ``a2`` (axis -2).  Central differences give the
    fractional-coordinate derivatives ``∂s/∂f1``, ``∂s/∂f2``; the chain
    rule via reciprocal vectors converts these to Cartesian components::

        ∇s = (∂s/∂f1)·b1/(2π) + (∂s/∂f2)·b2/(2π)

    where ``b1, b2`` satisfy ``bi·aj = 2π δij``.  For a rectangular lattice
    (``a1 = [Lx, 0]``, ``a2 = [0, Ly]``) this reduces to the familiar
    central-difference formulas divided by the physical grid spacing.

    Works for any 2-D lattice including oblique (hexagonal, skewed, …).

    Parameters
    ----------
    s : torch.Tensor
        Input scalar field. Shape ``[B, D0, D1]``.
        Axis -2 (size D0) runs along **a2**; axis -1 (size D1) along **a1**.
    a1 : torch.Tensor
        First lattice vector, shape ``[2]``.
    a2 : torch.Tensor
        Second lattice vector, shape ``[2]``.

    Returns
    -------
    gradx : torch.Tensor
        Cartesian x-component of the gradient. Shape ``[B, D0, D1]``.
    grady : torch.Tensor
        Cartesian y-component of the gradient. Shape ``[B, D0, D1]``.
    """
    D0 = s.shape[-2]   # a2 direction
    D1 = s.shape[-1]   # a1 direction

    # Reciprocal vectors (device / dtype match a1, a2)
    b1, b2 = reciprocal_lattice_vectors(a1, a2)   # each shape [2]

    # Fractional-coordinate central differences, scaled to ∂s/∂fi
    # Factor Di converts the unit-cell-normalised step to fractional units
    ds_df2 = 0.5 * D0 * (torch.roll(s, shifts=-1, dims=-2) - torch.roll(s, shifts=1, dims=-2))
    ds_df1 = 0.5 * D1 * (torch.roll(s, shifts=-1, dims=-1) - torch.roll(s, shifts=1, dims=-1))

    # Chain rule: ∇s = (∂s/∂f1)·b1/(2π) + (∂s/∂f2)·b2/(2π)
    two_pi = 2.0 * torch.pi
    gradx = (ds_df1 * b1[0] + ds_df2 * b2[0]) / two_pi
    grady = (ds_df1 * b1[1] + ds_df2 * b2[1]) / two_pi

    return gradx, grady


# ---------------------------------------------------------------------------
# Low-pass filtering in Fourier domain
# ---------------------------------------------------------------------------

def low_pass_mask(D0: int, D1: int, M: int, N: int,
                  device=None) -> torch.Tensor:
    """
    Boolean low-pass mask in the centred Fourier domain.

    Keeps the following harmonic orders (integer indices after FFT-shift):

    * axis -2 (size D0, a2 direction): orders n with ``|n| ≤ N``
    * axis -1 (size D1, a1 direction): orders m with ``|m| ≤ M``

    ``M`` is the a1-direction half-bandwidth; ``N`` is the a2-direction
    half-bandwidth.

    Parameters
    ----------
    D0 : int
        Size of spatial axis -2 (a2 direction).
    D1 : int
        Size of spatial axis -1 (a1 direction).
    M : int
        Half-bandwidth in the a1 direction (Fourier orders kept: ``-M … +M``).
    N : int
        Half-bandwidth in the a2 direction (Fourier orders kept: ``-N … +N``).
    device : torch.device or str, optional
        Device for the output. Defaults to CPU.

    Returns
    -------
    mask : torch.Tensor
        Bool mask, shape ``[D0, D1]``.
    """
    # Centred harmonic indices for each axis
    n = torch.arange(D0, device=device) - D0 // 2   # a2 orders, along axis -2
    m = torch.arange(D1, device=device) - D1 // 2   # a1 orders, along axis -1

    N_grid, M_grid = torch.meshgrid(n, m, indexing="ij")   # [D0, D1]
    return (N_grid.abs() <= N) & (M_grid.abs() <= M)


def low_pass_filter(grad: Tuple[torch.Tensor, torch.Tensor],
                    M: int, N: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Low-pass filter a pair of 2-D real gradient fields in Fourier space.

    Steps: FFT → centre-shift → apply mask → unshift → IFFT → real part.
    The real part is taken on output because gradients of a real field are
    real; discarding the small imaginary round-off keeps dtype clean.

    ``M`` truncates in the a1 direction (axis -1); ``N`` in a2 (axis -2).

    Parameters
    ----------
    grad : tuple[torch.Tensor, torch.Tensor]
        ``(gradx, grady)``, each of shape ``[B, D0, D1]``.
    M : int
        a1-direction half-bandwidth.
    N : int
        a2-direction half-bandwidth.

    Returns
    -------
    filteredx : torch.Tensor
        Filtered x-gradient, shape ``[B, D0, D1]``, real.
    filteredy : torch.Tensor
        Filtered y-gradient, shape ``[B, D0, D1]``, real.
    """
    if grad is None or len(grad) != 2:
        raise ValueError("grad must be a tuple of two tensors.")
    if grad[0].shape != grad[1].shape:
        raise ValueError("Gradient components must have the same shape.")

    gradx, grady = grad
    _, D0, D1 = gradx.shape

    # Forward FFT → centre
    gradx_fft = torch.fft.fftshift(torch.fft.fft2(gradx, dim=(-2, -1)), dim=(-2, -1))
    grady_fft = torch.fft.fftshift(torch.fft.fft2(grady, dim=(-2, -1)), dim=(-2, -1))

    # Mask: M for a1 (axis -1), N for a2 (axis -2); cast to complex dtype
    mask = low_pass_mask(D0, D1, M, N, device=gradx.device).to(gradx_fft.dtype)

    filteredx_fft = gradx_fft * mask
    filteredy_fft = grady_fft * mask

    # Unshift → inverse FFT → real part
    filteredx = torch.fft.ifft2(torch.fft.ifftshift(filteredx_fft, dim=(-2, -1)), dim=(-2, -1)).real
    filteredy = torch.fft.ifft2(torch.fft.ifftshift(filteredy_fft, dim=(-2, -1)), dim=(-2, -1)).real

    return filteredx, filteredy


# ---------------------------------------------------------------------------
# Normalization functions for 2-D vector fields
# ---------------------------------------------------------------------------

def _field_magnitude(field: torch.Tensor) -> torch.Tensor:
    """
    Per-pixel magnitude of a 2-D vector field, safe for autograd at zero.

    Parameters
    ----------
    field : torch.Tensor
        Vector field, shape ``[..., 2]``.

    Returns
    -------
    magnitude : torch.Tensor
        Shape ``[..., 1]``.
    """
    mag_sq = torch.sum(torch.abs(field) ** 2, dim=-1, keepdim=True)
    is_zero = mag_sq == 0
    mag_sq_safe = torch.where(is_zero, torch.ones_like(mag_sq), mag_sq)
    mag = torch.where(is_zero, torch.zeros_like(mag_sq), torch.sqrt(mag_sq_safe))
    return mag


def normalize_max_global(field: torch.Tensor) -> torch.Tensor:
    """
    Normalise a vector field by its global maximum per-pixel magnitude.

    Parameters
    ----------
    field : torch.Tensor
        Shape ``[B, D0, D1, 2]``.

    Returns
    -------
    normalized : torch.Tensor
        Shape ``[B, D0, D1, 2]``.
    """
    mag = _field_magnitude(field)                                       # [B, D0, D1, 1]
    max_mag = torch.amax(mag, dim=(-3, -2), keepdim=True)              # [B, 1, 1, 1]
    max_mag_safe = torch.where(max_mag == 0, torch.ones_like(max_mag), max_mag)
    return field / max_mag_safe


def normalize_elementwise(field: torch.Tensor) -> torch.Tensor:
    """
    Normalise each pixel of a vector field to unit magnitude.

    Parameters
    ----------
    field : torch.Tensor
        Shape ``[B, D0, D1, 2]``.

    Returns
    -------
    normalized : torch.Tensor
        Shape ``[B, D0, D1, 2]``.
    """
    mag_sq = torch.sum(torch.abs(field) ** 2, dim=-1, keepdim=True)
    is_zero = mag_sq == 0
    mag_sq_safe = torch.where(is_zero, torch.ones_like(mag_sq), mag_sq)
    return field / torch.sqrt(mag_sq_safe)


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------

def alignment_loss(field: torch.Tensor,
                   target_field: torch.Tensor,
                   weights: torch.Tensor) -> torch.Tensor:
    """
    Weighted MSE between a spatial vector field and a target.

    ``loss = mean(weights * ||field - target||²)``

    Parameters
    ----------
    field : torch.Tensor
        Shape ``[B, D0, D1, 2]``.
    target_field : torch.Tensor
        Shape ``[B, D0, D1, 2]``.
    weights : torch.Tensor
        Shape ``[B, D0, D1, 1]`` or ``[B, D0, D1]``.

    Returns
    -------
    loss : torch.Tensor
        Shape ``[B]``.
    """
    if field.shape != target_field.shape:
        raise ValueError("field and target_field must have the same shape.")
    expected_w = field.shape[:-1]
    if weights.shape != expected_w and weights.shape != expected_w + (1,):
        raise ValueError("weights must have shape [B, D0, D1] or [B, D0, D1, 1].")

    diff = field - target_field
    diff_sq = torch.sum(torch.abs(diff) ** 2, dim=-1, keepdim=True)    # [B, D0, D1, 1]
    weighted = (weights * diff_sq).squeeze(-1)                          # [B, D0, D1]

    return torch.mean(weighted, dim=(-1, -2))                           # [B]


def fourier_regularization_loss(field_fft: torch.Tensor,
                                a1: torch.Tensor,
                                a2: torch.Tensor) -> torch.Tensor:
    """
    Fourier-domain smoothness penalty: penalises high-frequency content.

    ``loss = mean(|G|² * ||F̂(field)||²)``

    where ``G = m·b1 + n·b2`` is the Cartesian reciprocal-lattice vector
    for harmonic order ``(m, n)`` (m along a1, n along a2), and the cell
    area ``|det[a1|a2]|`` provides the correct dimensional weight.

    The input must be the **centred** FFT of the 2-component vector field
    with spatial dims at axes -3, -2 (a2 and a1 directions respectively).

    Parameters
    ----------
    field_fft : torch.Tensor
        Centred Fourier coefficients of the 2-component field.
        Shape ``[B, D0, D1, 2]``, complex.
        Axis -3 (size D0): a2 direction, harmonic order n.
        Axis -2 (size D1): a1 direction, harmonic order m.
        Axis -1 (size 2): Cartesian vector components (x, y).
    a1 : torch.Tensor
        First lattice vector, shape ``[2]``.
    a2 : torch.Tensor
        Second lattice vector, shape ``[2]``.

    Returns
    -------
    loss : torch.Tensor
        Shape ``[B]``.
    """
    b1, b2 = reciprocal_lattice_vectors(a1, a2)                        # each [2]
    area = (a1[0] * a2[1] - a1[1] * a2[0]).abs()                      # scalar

    _, D0, D1, _ = field_fft.shape

    # Centred harmonic indices on the field's device, with the same real dtype as a1
    n = (torch.arange(D0, device=field_fft.device, dtype=a1.dtype) - D0 // 2)  # a2, [D0]
    m = (torch.arange(D1, device=field_fft.device, dtype=a1.dtype) - D1 // 2)  # a1, [D1]

    N_grid, M_grid = torch.meshgrid(n, m, indexing="ij")               # [D0, D1]

    # Cartesian reciprocal-lattice vector G = m*b1 + n*b2
    Gx = M_grid * b1[0] + N_grid * b2[0]                               # [D0, D1]
    Gy = M_grid * b1[1] + N_grid * b2[1]

    K_norm2 = area * (Gx ** 2 + Gy ** 2)                               # [D0, D1]

    # Sum over Cartesian vector components; average over spatial harmonics
    power = torch.sum(torch.abs(field_fft) ** 2, dim=-1)               # [B, D0, D1]
    return torch.mean(K_norm2 * power, dim=(-2, -1))                    # [B]


def smoothness_loss(field: torch.Tensor,
                    a1: torch.Tensor,
                    a2: torch.Tensor) -> torch.Tensor:
    """
    Spatial smoothness penalty: mean squared gradient of a vector field.

    ``loss = mean(||∇field||²)``

    Parameters
    ----------
    field : torch.Tensor
        Spatial vector field, shape ``[B, D0, D1, 2]``.
    a1 : torch.Tensor
        First lattice vector, shape ``[2]``.
    a2 : torch.Tensor
        Second lattice vector, shape ``[2]``.

    Returns
    -------
    loss : torch.Tensor
        Shape ``[B]``.
    """
    fx = field[..., 0]    # [B, D0, D1]
    fy = field[..., 1]

    gradx_fx, grady_fx = _grad_periodic(fx, a1, a2)
    gradx_fy, grady_fy = _grad_periodic(fy, a1, a2)

    # Use |grad|² (not grad²) so the loss is real for complex fields too.
    grad_sq = (torch.abs(gradx_fx) ** 2 + torch.abs(grady_fx) ** 2
               + torch.abs(gradx_fy) ** 2 + torch.abs(grady_fy) ** 2)  # [B, D0, D1] real

    return torch.mean(grad_sq, dim=(-1, -2))                            # [B]


def total_loss(params: torch.Tensor,
               target_field: torch.Tensor,
               weights: torch.Tensor,
               a1: torch.Tensor,
               a2: torch.Tensor,
               alpha: float = 1.0,
               beta: float = 1e-3,
               gamma: float = 1.0) -> torch.Tensor:
    """
    Weighted sum of alignment, Fourier-regularization and smoothness losses.

    The optimization variable ``params`` stores the real/imag parts of the
    centred Fourier representation of the 2-component vector field::

        params shape [B, D0, D1, 2, 2]
            axis -5  B:  batch
            axis -4  D0: a2 direction (harmonic order n after shift)
            axis -3  D1: a1 direction (harmonic order m after shift)
            axis -2  2:  Cartesian vector component (x, y)
            axis -1  2:  (real, imaginary) of each Fourier coefficient

    Parameters
    ----------
    params : torch.Tensor
        Fourier-domain optimization variable, shape ``[B, D0, D1, 2, 2]``.
    target_field : torch.Tensor
        Spatial target vector field, shape ``[B, D0, D1, 2]``.
    weights : torch.Tensor
        Alignment weights, shape ``[B, D0, D1, 1]`` or ``[B, D0, D1]``.
    a1 : torch.Tensor
        First lattice vector, shape ``[2]``.
    a2 : torch.Tensor
        Second lattice vector, shape ``[2]``.
    alpha, beta, gamma : float
        Weights for alignment, Fourier regularization, and smoothness losses.

    Returns
    -------
    loss : torch.Tensor
        Shape ``[B]``.
    """
    # Centred complex Fourier field: [B, D0, D1, 2] complex
    field_fft = params[..., 0] + 1j * params[..., 1]

    # Inverse FFT back to spatial domain; keep complex to support complex Jones targets.
    # alignment_loss uses |diff|^2 which handles complex correctly.
    # For real-target methods the imaginary part is numerical noise (≈1e-15).
    field_spatial = torch.fft.ifft2(
        torch.fft.ifftshift(field_fft, dim=(-3, -2)), dim=(-3, -2)
    )                                                                    # [B, D0, D1, 2] complex

    loss_al = alignment_loss(field_spatial, target_field, weights)
    loss_f  = fourier_regularization_loss(field_fft, a1, a2)
    loss_s  = smoothness_loss(field_spatial, a1, a2)

    return alpha * loss_al + beta * loss_f + gamma * loss_s


# ---------------------------------------------------------------------------
# Jones normalization
# ---------------------------------------------------------------------------

def normalize_jones(field: torch.Tensor) -> torch.Tensor:
    """
    Generate a Jones vector field (Antos 2009, DOI 10.1364/OE.17.007269).

    Projects a real orientation field into the complex Jones representation:
    each pixel becomes a unit-norm complex 2-vector whose phase encodes the
    local orientation in a way that is smooth even across sign flips.

    Parameters
    ----------
    field : torch.Tensor
        Vector field, shape ``[B, D0, D1, 2]``. Real or complex.

    Returns
    -------
    jones_field : torch.Tensor
        Jones-normalized field, shape ``[B, D0, D1, 2]``, complex.
    """
    if field.shape[-1] != 2:
        raise ValueError("Last dimension of field must be 2.")
    if field.ndim != 4:
        raise ValueError("field must be 4-D with shape [B, D0, D1, 2].")

    field = normalize_max_global(field)
    magnitude = _field_magnitude(field)                                  # [B, D0, D1, 1]

    near_zero = torch.isclose(magnitude, torch.zeros_like(magnitude))
    magnitude_safe = torch.where(near_zero, torch.ones_like(magnitude), magnitude)

    inv_sqrt2 = 1.0 / math.sqrt(2.0)

    tx_norm = torch.where(near_zero,
                          torch.full_like(field[..., 0:1], inv_sqrt2),
                          field[..., 0:1] / magnitude_safe)
    ty_norm = torch.where(near_zero,
                          torch.full_like(field[..., 1:2], inv_sqrt2),
                          field[..., 1:2] / magnitude_safe)

    # Phase parameter: smooth function of per-pixel magnitude
    phi = torch.pi / 8.0 * (1.0 + torch.cos(torch.pi * magnitude))

    # Complex angle of the normalised 2-D orientation
    theta = torch.angle(tx_norm + 1j * ty_norm)
    exp_i_theta = torch.exp(1j * theta)

    jx = exp_i_theta * (tx_norm * torch.cos(phi) - ty_norm * 1j * torch.sin(phi))
    jy = exp_i_theta * (ty_norm * torch.cos(phi) + tx_norm * 1j * torch.sin(phi))

    return torch.cat([jx, jy], dim=-1)
