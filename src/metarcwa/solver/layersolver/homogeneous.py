# metarcwa/solver/layersolver/homogeneous.py
"""
homogeneous — closed-form modal solver for homogeneous layers
=============================================================

**Scope:** isotropic, non-magnetic (μ = 1) homogeneous layers only.

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

from metarcwa.solver.blockmatrix import Block, Block2x2
from metarcwa._dtypes import _REAL_TO_COMPLEX
from metarcwa.solver.layersolver._modes import _branch_select, _lam_inv_block, _warn_grazing


def homogeneous_kz(epsilon: torch.Tensor,
                   kx: torch.Tensor, ky: torch.Tensor,
                   forward: str = "positive", tol: float = 1e-4, delta=1e-30) -> torch.Tensor:
    """
    Compute normalized kz for every Fourier harmonic of a homogeneous layer.

    Valid only for isotropic scalar permittivity (μ = 1 assumed).

    For each harmonic the squared z-wavenumber is:

        kz² = ε − kx² − ky²

    The sign of each mode is chosen by :func:`_branch_select` so that:
      - propagating modes (|Re(lam)| ≤ tol·|lam|): Re(kz) > 0
      - evanescent  modes (|Re(lam)| >  tol·|lam|): Im(kz) > 0
    (``lam = 1j·kz``; see :func:`_branch_select` for why classification keys
    on ``Re(lam)`` rather than ``Im(kz)``.)

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
        Relative threshold (as a fraction of ``|lam|``) passed to
        :func:`_branch_select`. Default ``1e-4``.
    delta : float, optional
        Lorentzian regularisation for the square-root gradient.  kz is computed
        as ``lam2 / sqrt(lam2 + delta)`` instead of ``sqrt(lam2)``, which keeps
        the gradient ``d(kz)/d(lam2) = delta / (lam2 + delta)^(3/2)`` finite at
        the grazing singularity (lam2 = 0) where the plain sqrt gradient blows
        up.  The O(delta) error in kz is negligible for default ``delta = 1e-30``.
        Set to ``0`` to recover the unregularised sqrt (may give NaN gradients
        near grazing incidence).

    Returns
    -------
    kz : torch.Tensor
        Complex z-wavenumber for each harmonic, duplicated for the two
        polarisation blocks. Shape ``[..., 2Nh]``.
    """
    ndim_extra = kx.ndim - epsilon.ndim
    eps = epsilon.reshape(*epsilon.shape, *([1] * ndim_extra)).to(_REAL_TO_COMPLEX[kx.real.dtype])

    lam2_block = kx**2 + ky**2 - eps                          # [..., Nh]
    lam2 = torch.cat([lam2_block, lam2_block], dim=-1)         # [..., 2Nh]

    lam = _branch_select(lam2 / torch.sqrt(lam2 + delta), tol)

    if forward == "negative":
        lam = -lam
    elif forward != "positive":
        raise ValueError("forward must be 'positive' or 'negative'")

    return -1j * lam


def homogeneous_Q(epsilon: torch.Tensor,
                  kx: torch.Tensor, ky: torch.Tensor) -> Block2x2:
    """
    Assemble the Q matrix for a homogeneous isotropic non-magnetic layer.

    Valid only for isotropic scalar permittivity (μ = 1 assumed).

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
    eps = epsilon.reshape(*epsilon.shape, *([1] * ndim_extra)).to(_REAL_TO_COMPLEX[kx.real.dtype])

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

    Valid only for isotropic scalar permittivity (μ = 1 assumed).

    For a homogeneous medium the Fourier harmonics are already eigenvectors of
    the PQ operator, so no eigendecomposition is needed. The E-mode matrix is
    the identity (W = I, not returned) and the H-mode matrix is:

        V = Q · diag(1/lam)

    computed by splitting lam into its two Nh-sized blocks and treating
    diag(1/lam) as a Block2x2 diagonal, so V remains all-DIAG (no dense
    matrices are allocated).

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

    _warn_grazing(lam, tol, "homogeneous_modes")

    Q0  = homogeneous_Q(epsilon=epsilon, kx=kx, ky=ky)        # Block2x2, all DIAG
    Nh  = kx.shape[-1]
    lam_inv = _lam_inv_block(lam, Nh)                          # Lorentzian-regularized
    V = Q0 @ lam_inv                                           # Block2x2 @ Block2x2
    return lam, V
