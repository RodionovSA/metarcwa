# metarcwa/solver/homogeneous.py
# Solver utility functions for a homogeneous medium

import torch
from typing import Tuple
import warnings


def homogeneous_kz(epsilon: torch.Tensor,
                   kx: torch.Tensor, ky: torch.Tensor,
                   forward: str = "positive", tol: float = 1e-12) -> torch.Tensor:
    """
    Compute normalized kz for every mode of a homogeneous isotropic layer.

    For each Fourier harmonic (m, n) the squared z-wavenumber is:

        kz^2 = epsilon - kx^2 - ky^2

    Taking the square root yields two branches (forward and backward).
    The sign of each mode is chosen so that:
      - propagating modes (|Im(kz)| <= tol): Re(kz) > 0
      - evanescent modes  (|Im(kz)| >  tol): Im(kz) > 0

    This module uses the exp(-j*omega*t) time convention.  The modal exponent is
    lam = 1j * kz, so forward-propagating fields vary as exp(+lam * z_tilde)
    where z_tilde = k0 * z.  Pass ``forward="negative"`` to flip all signs
    (i.e. select the backward-propagating branch).

    Parameters:
    -----------
    epsilon : torch.Tensor
        Relative permittivity (isotropic scalar) of the layer.
        Shape ``[N_wl, ...]`` and must broadcast against kx / ky.
    kx : torch.Tensor
        x-component of the in-plane wavevector, normalised by k0.
        Shape ``[..., Nh]``.
    ky : torch.Tensor
        y-component of the in-plane wavevector, normalised by k0.
        Shape ``[..., Nh]``.
    forward : str, optional
        Branch selector. ``"positive"`` (default) returns the forward-propagating
        branch; ``"negative"`` returns the backward-propagating branch.
    tol : float, optional
        Threshold below which a mode is treated as propagating (not evanescent).
        Default ``1e-12``.

    Returns:
    --------
    kz : torch.Tensor
        Complex z-wavenumber for each mode and harmonic pair.
        Shape ``(..., 2N)`` where ``N = Nh`` (each harmonic contributes two modes).
    """
    # Broadcast epsilon (shape [...]) against kx/ky (shape [..., Nh]) by
    # inserting trailing singleton dims so the harmonic axis doesn't alias.
    ndim_extra = kx.ndim - epsilon.ndim
    eps = epsilon.reshape(*epsilon.shape, *([1] * ndim_extra))

    lam2_block = kx**2 + ky**2 - eps                         # (..., Nh)
    lam2 = torch.cat([lam2_block, lam2_block], dim=-1)        # (..., 2Nh)

    lam = torch.sqrt(lam2)
    kz = -1j * lam

    is_evan = kz.imag.abs() > tol
    sign = torch.where(is_evan, torch.sign(kz.imag), torch.sign(kz.real))
    if forward == "negative":
        sign = -sign
    elif forward != "positive":
        raise ValueError("forward must be 'positive' or 'negative'")
    sign = torch.where(sign == 0, torch.ones_like(sign), sign)
    return kz * sign


def homogeneous_Q(epsilon: torch.Tensor,
                  kx: torch.Tensor, ky: torch.Tensor) -> torch.Tensor:
    """
    Assemble the Q matrix for a homogeneous isotropic non-magnetic layer.

    For a non-magnetic medium (mu = 1) the Q operator has the block form:

        Q = [[ -Kx*Ky ,     Kx^2 - eps*I ],
             [ eps*I - Ky^2 ,   Ky*Kx    ]]

    where Kx, Ky are the diagonal matrices of in-plane wavevector components.
    All four blocks are diagonal, so they are stored as ``torch.diag_embed``
    of the corresponding harmonic-vector products.

    Sign convention: this module uses exp(-j*omega*t); Q is defined so that
    the modal relation V = Q * diag(1/lam) is consistent with forward fields
    varying as exp(+lam * z_tilde), where z_tilde = k0 * z.

    Parameters:
    -----------
    epsilon : torch.Tensor
        Relative permittivity of the layer, normalised to k0.
        Shape ``[N_wl, ...]``, must broadcast against kx / ky.
    kx : torch.Tensor
        x-component of the in-plane wavevector, normalised by k0.
        Shape ``[..., N]``.
    ky : torch.Tensor
        y-component of the in-plane wavevector, normalised by k0.
        Shape ``[..., N]``.

    Returns:
    --------
    Q0 : torch.Tensor
        Q matrix for the homogeneous layer.  Shape ``(..., 2N, 2N)``.
    """
    ndim_extra = kx.ndim - epsilon.ndim
    eps = epsilon.reshape(*epsilon.shape, *([1] * ndim_extra))
    
    Q11 = torch.diag_embed(-kx * ky)
    Q12 = torch.diag_embed(kx**2 - eps)
    Q21 = torch.diag_embed(eps - ky**2)
    Q22 = torch.diag_embed(ky * kx)
    top = torch.cat([Q11, Q12], dim=-1)
    bot = torch.cat([Q21, Q22], dim=-1)
    return torch.cat([top, bot], dim=-2)


def homogeneous_modes(epsilon: torch.Tensor,
                      kx: torch.Tensor, ky: torch.Tensor,
                      forward: str = "positive",
                      tol: float = 1e-6) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Closed-form modal decomposition for a homogeneous isotropic layer.

    For a homogeneous medium the Fourier harmonics are already eigenvectors of
    the PQ operator, so no eigendecomposition is required.  The mode matrix is
    the identity (``W = I``) and the H-mode matrix is:

        V = Q0 @ diag(1 / lam)

    which is evaluated as column-wise scaling: ``Q0 * (1/lam)[..., None, :]``.

    The boundary-condition matrix (gap matrix) is then:

        Phi0 = [[W,  W],
                [V, -V]]

    Assembling Phi0 is left to the caller.

    .. note::
        Division by ``lam`` is undefined for grazing modes (kx^2 + ky^2 == eps).
        This edge case is not guarded here — avoid exact grazing incidence or
        handle it upstream.

    Parameters:
    -----------
    epsilon : torch.Tensor
        Relative permittivity of the layer, normalised to k0.
        Shape ``[N_wl, ...]``, must broadcast against kx / ky.
    kx : torch.Tensor
        x-component of the in-plane wavevector, normalised by k0.
        Shape ``[..., N]``.
    ky : torch.Tensor
        y-component of the in-plane wavevector, normalised by k0.
        Shape ``[..., N]``.
    forward : str, optional
        Branch selector passed to :func:`homogeneous_kz`.
        ``"positive"`` (default) for forward-propagating modes.

    Returns:
    --------
    lam : torch.Tensor
        Modal exponents, ``lam = 1j * kz``.  Shape ``(..., 2N)``.
    W : torch.Tensor
        E-mode matrix (identity).  Shape ``(..., 2N, 2N)``.
    V : torch.Tensor
        H-mode matrix, ``Q0 @ diag(1/lam)``.  Shape ``(..., 2N, 2N)``.
    """
    kz = homogeneous_kz(epsilon=epsilon, kx=kx, ky=ky, forward=forward)
    lam = 1j * kz                                           # (..., 2N)
    
    zero_mask = lam.abs() < tol
    if zero_mask.any():
        warnings.warn(
            f"homogeneous_modes: {zero_mask.sum().item()} mode(s) have |lam| < {tol} "
            "(grazing incidence). Corresponding columns of V will be nan.",
            RuntimeWarning,
            stacklevel=2,
        )

    n2 = lam.shape[-1]
    W = torch.eye(n2, dtype=lam.dtype, device=lam.device)
    W = W.expand(*lam.shape[:-1], n2, n2)

    Q0 = homogeneous_Q(epsilon=epsilon, kx=kx, ky=ky)      # (..., 2N, 2N)
    V = Q0 * (1.0 / lam).unsqueeze(-2)                     # scale column j by 1/lam_j
    return lam, W, V
