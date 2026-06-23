# metarcwa/solver/homogeneous.py
"""
homogeneous — closed-form modal solver for homogeneous layers
=============================================================

**Scope:** isotropic, non-magnetic (μ = 1) homogeneous layers only.
For anisotropic or magnetic media an eigendecomposition is required; these
functions will give wrong results if called with such materials.

Three public functions, in dependency order:

  homogeneous_kz(epsilon, kx, ky)    → kz  [..., 2Nh]
      Branch-corrected z-wavenumber for every Fourier harmonic.

  homogeneous_Q(epsilon, kx, ky)     → Block2x2 of DIAG blocks
      Q operator [[−Kx Ky, Kx²−εI], [εI−Ky², Ky Kx]].
      All four blocks are diagonal → stored cheaply as Block(DIAG, …).

  homogeneous_modes(epsilon, kx, ky) → (lam, V)
      lam : modal exponents 1j·kz            [..., 2Nh]
      V   : H-mode matrix Q · diag(1/lam)    Block2x2 of DIAG blocks

      For a homogeneous layer W = I (Fourier harmonics are already eigenvectors),
      so W is not returned — the caller can use Block2x2.identity() if needed.

All functions use the exp(−j ω t) time convention.
"""

import torch
from typing import Tuple
import warnings

from .blockmatrix import Block, Block2x2


def homogeneous_kz(epsilon: torch.Tensor,
                   kx: torch.Tensor, ky: torch.Tensor,
                   forward: str = "positive", tol: float = 1e-12) -> torch.Tensor:
    """
    Compute normalized kz for every Fourier harmonic of a homogeneous layer.

    Valid only for isotropic scalar permittivity (μ = 1 assumed).

    For each harmonic the squared z-wavenumber is:

        kz² = ε − kx² − ky²

    The sign of each mode is chosen so that:
      - propagating modes (|Im(kz)| ≤ tol): Re(kz) > 0
      - evanescent  modes (|Im(kz)| >  tol): Im(kz) > 0

    Time convention: exp(−j ω t). The modal exponent is lam = 1j·kz, so
    forward-propagating fields vary as exp(+lam · z̃) where z̃ = k0·z.
    Pass ``forward="negative"`` to select the backward-propagating branch.

    Parameters
    ----------
    epsilon : torch.Tensor
        Isotropic relative permittivity of the layer. Shape ``[N_wl, ...]``;
        must broadcast against kx / ky.
    kx : torch.Tensor
        x-component of the in-plane wavevector, normalised by k0.
        Shape ``[..., Nh]``.
    ky : torch.Tensor
        y-component of the in-plane wavevector, normalised by k0.
        Shape ``[..., Nh]``.
    forward : str, optional
        ``"positive"`` (default) — forward branch; ``"negative"`` — backward branch.
    tol : float, optional
        Threshold below which a mode is treated as propagating. Default ``1e-12``.

    Returns
    -------
    kz : torch.Tensor
        Complex z-wavenumber for each harmonic, duplicated for the two
        polarisation blocks. Shape ``[..., 2Nh]``.
    """
    ndim_extra = kx.ndim - epsilon.ndim
    eps = epsilon.reshape(*epsilon.shape, *([1] * ndim_extra))

    lam2_block = kx**2 + ky**2 - eps                          # [..., Nh]
    lam2 = torch.cat([lam2_block, lam2_block], dim=-1)         # [..., 2Nh]

    lam = torch.sqrt(lam2)
    kz  = -1j * lam

    is_evan = kz.imag.abs() > tol
    sign    = torch.where(is_evan, torch.sign(kz.imag), torch.sign(kz.real))
    if forward == "negative":
        sign = -sign
    elif forward != "positive":
        raise ValueError("forward must be 'positive' or 'negative'")
    sign = torch.where(sign == 0, torch.ones_like(sign), sign)
    return kz * sign


def homogeneous_Q(epsilon: torch.Tensor,
                  kx: torch.Tensor, ky: torch.Tensor) -> Block2x2:
    """
    Assemble the Q matrix for a homogeneous isotropic non-magnetic layer.

    Valid only for isotropic ε and μ = 1. For magnetic or anisotropic media
    the off-diagonal coupling terms differ and this function is not applicable.

    For μ = 1 the Q operator has the 2×2 block form:

        Q = [[ −Kx·Ky,     Kx²−ε·I ],
             [ ε·I−Ky²,    Ky·Kx   ]]

    where Kx, Ky are the diagonal matrices of in-plane wavevector components.
    Because each block is a product of two diagonal operators (or a diagonal ±
    a scalar multiple of I), all four blocks are diagonal and are stored
    efficiently as ``Block(DIAG, …)`` — no dense matrices are allocated.

    Parameters
    ----------
    epsilon : torch.Tensor
        Isotropic relative permittivity. Shape ``[N_wl, ...]``;
        must broadcast against kx / ky.
    kx : torch.Tensor
        x-component of the in-plane wavevector, normalised by k0.
        Shape ``[..., Nh]``.
    ky : torch.Tensor
        y-component of the in-plane wavevector, normalised by k0.
        Shape ``[..., Nh]``.

    Returns
    -------
    Q : Block2x2
        Q matrix; each of the four entries is a ``Block(DIAG, …)`` of
        shape ``[..., Nh]``.
    """
    ndim_extra = kx.ndim - epsilon.ndim
    eps = epsilon.reshape(*epsilon.shape, *([1] * ndim_extra))

    Q11 = Block(Block.DIAG, -kx * ky)
    Q12 = Block(Block.DIAG,  kx**2 - eps)
    Q21 = Block(Block.DIAG,  eps - ky**2)
    Q22 = Block(Block.DIAG,  ky * kx)
    return Block2x2(Q11, Q12, Q21, Q22)


def homogeneous_modes(epsilon: torch.Tensor,
                      kx: torch.Tensor, ky: torch.Tensor,
                      forward: str = "positive",
                      tol: float = 1e-6) -> Tuple[torch.Tensor, Block2x2]:
    """
    Closed-form modal decomposition for a homogeneous isotropic layer.

    Valid only for isotropic ε and μ = 1. For anisotropic or magnetic media
    the Fourier harmonics are not eigenvectors of PQ and an eigendecomposition
    is required instead.

    For a homogeneous medium the Fourier harmonics are already eigenvectors of
    the PQ operator, so no eigendecomposition is needed. The E-mode matrix is
    the identity (W = I, not returned) and the H-mode matrix is:

        V = Q · diag(1/lam)

    computed by splitting lam into its two Nh-sized blocks and treating
    diag(1/lam) as a Block2x2 diagonal, so V remains all-DIAG (no dense
    matrices are allocated).

    .. note::
        Division by lam is undefined for grazing modes (kx² + ky² = ε).
        This edge case is not guarded here — avoid exact grazing incidence or
        handle it upstream.

    Parameters
    ----------
    epsilon : torch.Tensor
        Isotropic relative permittivity. Shape ``[N_wl, ...]``;
        must broadcast against kx / ky.
    kx : torch.Tensor
        x-component of the in-plane wavevector, normalised by k0.
        Shape ``[..., Nh]``.
    ky : torch.Tensor
        y-component of the in-plane wavevector, normalised by k0.
        Shape ``[..., Nh]``.
    forward : str, optional
        Branch selector passed to :func:`homogeneous_kz`.
        ``"positive"`` (default) for the forward-propagating branch.
    tol : float, optional
        Modes with |lam| < tol trigger a RuntimeWarning (grazing incidence).
        Default ``1e-6``.

    Returns
    -------
    lam : torch.Tensor
        Modal exponents lam = 1j·kz. Shape ``[..., 2Nh]``.
    V : Block2x2
        H-mode matrix Q · diag(1/lam). Each entry is a ``Block(DIAG, …)``
        of shape ``[..., Nh]``. The E-mode matrix W = I is implicit.
    """
    kz  = homogeneous_kz(epsilon=epsilon, kx=kx, ky=ky, forward=forward)
    lam = 1j * kz                                              # [..., 2Nh]

    zero_mask = lam.abs() < tol
    if zero_mask.any():
        warnings.warn(
            f"homogeneous_modes: {zero_mask.sum().item()} mode(s) have |lam| < {tol} "
            "(grazing incidence). Corresponding columns of V will be nan.",
            RuntimeWarning,
            stacklevel=2,
        )

    Q0  = homogeneous_Q(epsilon=epsilon, kx=kx, ky=ky)        # Block2x2, all DIAG
    Nh  = kx.shape[-1]
    kw  = dict(device=lam.device, dtype=lam.dtype)
    lam_inv = Block2x2(
        Block(Block.DIAG, 1.0 / lam[..., :Nh]),               # left  column block
        Block.zeros(**kw),
        Block.zeros(**kw),
        Block(Block.DIAG, 1.0 / lam[..., Nh:]),               # right column block
    )
    V = Q0 @ lam_inv                                           # Block2x2 @ Block2x2
    return lam, V
