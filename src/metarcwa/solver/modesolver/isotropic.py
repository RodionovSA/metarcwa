# metarcwa/solver/modesolver/isotropic.py
"""
isotropic — P and Q operators for patterned isotropic layers
=============================================================

**Scope:** non-magnetic (μ = 1) patterned layers with isotropic permittivity.
All operators are assembled from the Laurent convolution matrix of ε(r) and,
optionally, the TVF anisotropy correction blocks (Li factorization rules).

Four internal builders, one combiner, and one top-level entry point:

  compute_Q0(Kx, Ky, epsilon_conv)            → Block2x2
      Base Q matrix without TVF correction.

  compute_A(epsilon_grid, m_flat, n_flat, tvf) → (Axx, Axy, Ayx, Ayy)
      TVF anisotropy blocks from the tangent vector field.

  compute_Qfact(epsilon_conv, Axx, Axy, Ayx, Ayy) → Block2x2
      Factorization correction to Q0 derived from the A blocks.

  compute_Q(Kx, Ky, epsilon_conv, [A blocks])  → Block2x2
      Full Q = Q0 (+ Qfact if TVF blocks are provided).

  compute_P(Kx, Ky, epsilon_conv)              → Block2x2
      P matrix; ε⁻¹ factored into every term via Block.solve().

  compute_isotropic(epsilon_grid, m_flat, n_flat, kx, ky, tvf=None) → (P, Q)
      Top-level entry: builds ε_conv once and returns both operators.

All functions use the exp(−j ω t) time convention.
"""

import torch
from typing import Tuple

from metarcwa.solver.blockmatrix import Block, Block2x2
from metarcwa.solver.tvf import TVF
from metarcwa.solver.convolution import convolution_matrix


def compute_Q0(Kx: Block, Ky: Block, epsilon_conv: Block) -> Block2x2:
    """
    Assemble the base Q matrix for an isotropic patterned layer.

    For a non-magnetic (μ = 1) isotropic medium the Q operator has the
    2×2 block form:

        Q₀ = [[ −Kx·Ky,       Kx²−ε ],
               [  ε−Ky²,       Ky·Kx ]]

    where Kx, Ky are diagonal matrices of in-plane wavevector components
    and ε is the convolution (Toeplitz) matrix of the permittivity.

    Parameters
    ----------
    Kx : Block
        Diagonal Block of x-components of the in-plane wavevector, shape
        ``[..., Nh]``.
    Ky : Block
        Diagonal Block of y-components of the in-plane wavevector, shape
        ``[..., Nh]``.
    epsilon_conv : Block
        Dense Block holding the Toeplitz convolution matrix of ε(r), shape
        ``[..., Nh, Nh]``.

    Returns
    -------
    Q0 : Block2x2
        Q matrix without TVF correction. Entry kinds depend on the
        inputs: DIAG inputs promote to DENSE wherever epsilon_conv is DENSE.
    """
    a = -Kx @ Ky
    b = Kx @ Kx - epsilon_conv
    c = epsilon_conv - Ky @ Ky
    d = -a
    return Block2x2(a, b, c, d)


def compute_A(epsilon_grid: torch.Tensor, m_flat: torch.Tensor,
              n_flat: torch.Tensor, tvf: TVF) -> Tuple[Block, Block, Block, Block]:
    """
    Compute the TVF anisotropy correction blocks for the Li factorization.

    The tangent vector field (Tx, Ty) encodes the local polarization
    direction at every grid point.  The anisotropy blocks are the
    convolution matrices of the outer-product components of the TVF:

        Axx  ←  Conv(|Ty|²)          Ayy  ←  Conv(|Tx|²)
        Axy  ←  Conv(Tx* · Ty)       Ayx  ←  Conv(Tx · Ty*)

    Parameters
    ----------
    epsilon_grid : torch.Tensor
        Permittivity sampled on the real-space grid, shape ``[..., Ny, Nx]``.
        Passed to ``tvf.compute()`` to derive the tangent vector field.
    m_flat : torch.Tensor
        Integer harmonic indices along x, shape ``[Nh]``.
    n_flat : torch.Tensor
        Integer harmonic indices along y, shape ``[Nh]``.
    tvf : TVF
        Configured TVF instance used to compute the tangent vector field.

    Returns
    -------
    Axx : Block
        DENSE Block, shape ``[..., Nh, Nh]``. Convolution of |Ty|².
    Axy : Block
        DENSE Block, shape ``[..., Nh, Nh]``. Convolution of Tx*·Ty.
    Ayx : Block
        DENSE Block, shape ``[..., Nh, Nh]``. Convolution of Tx·Ty*.
    Ayy : Block
        DENSE Block, shape ``[..., Nh, Nh]``. Convolution of |Tx|².
    """
    Tx, Ty = tvf.compute(epsilon_grid)
    Tx_fft = torch.fft.fft2(Tx, dim=(-2, -1))
    Ty_fft = torch.fft.fft2(Ty, dim=(-2, -1))

    axx = Ty_fft.abs() ** 2
    axy = Tx_fft.conj() * Ty_fft
    ayx = Tx_fft * Ty_fft.conj()
    ayy = Tx_fft.abs() ** 2

    Axx = Block(Block.DENSE, convolution_matrix(axx, m_flat, n_flat))
    Axy = Block(Block.DENSE, convolution_matrix(axy, m_flat, n_flat))
    Ayx = Block(Block.DENSE, convolution_matrix(ayx, m_flat, n_flat))
    Ayy = Block(Block.DENSE, convolution_matrix(ayy, m_flat, n_flat))

    return Axx, Axy, Ayx, Ayy


def compute_Qfact(epsilon_conv: Block,
                  Axx: Block, Axy: Block, Ayx: Block, Ayy: Block) -> Block2x2:
    """
    Assemble the TVF factorization correction to the Q matrix.

    The correction encodes the difference between the inverse-rule and
    direct-rule Fourier factorizations, weighted by the TVF anisotropy
    blocks.  The four entries are:

        Qfact = [[ -(ε − ε⁻¹)·Ayx,   (ε − ε⁻¹)·Ayy ],
                 [ -(ε − ε⁻¹)·Axx,   (ε − ε⁻¹)·Axy ]]

    Parameters
    ----------
    epsilon_conv : Block
        Dense convolution matrix of ε(r), shape ``[..., Nh, Nh]``.
    Axx : Block
        Anisotropy block from :func:`compute_A`, shape ``[..., Nh, Nh]``.
    Axy : Block
        Anisotropy block from :func:`compute_A`, shape ``[..., Nh, Nh]``.
    Ayx : Block
        Anisotropy block from :func:`compute_A`, shape ``[..., Nh, Nh]``.
    Ayy : Block
        Anisotropy block from :func:`compute_A`, shape ``[..., Nh, Nh]``.

    Returns
    -------
    Qfact : Block2x2
        Factorization correction; add to Q0 to get the full TVF-corrected Q.
    """
    a_fact = -epsilon_conv @ Ayx + epsilon_conv.solve(Ayx)
    b_fact =  epsilon_conv @ Ayy - epsilon_conv.solve(Ayy)
    c_fact = -epsilon_conv @ Axx + epsilon_conv.solve(Axx)
    d_fact =  epsilon_conv @ Axy - epsilon_conv.solve(Axy)
    return Block2x2(a_fact, b_fact, c_fact, d_fact)


def compute_Q(Kx: Block, Ky: Block, epsilon_conv: Block,
              Axx: Block | None = None, Axy: Block | None = None,
              Ayx: Block | None = None, Ayy: Block | None = None) -> Block2x2:
    """
    Assemble the full Q matrix, optionally with TVF factorization correction.

    Returns ``Q0`` when all A blocks are ``None`` (homogeneous or no TVF).
    Returns ``Q0 + Qfact`` when all four A blocks are provided.

    Parameters
    ----------
    Kx : Block
        Diagonal Block of x-wavevector components, shape ``[..., Nh]``.
    Ky : Block
        Diagonal Block of y-wavevector components, shape ``[..., Nh]``.
    epsilon_conv : Block
        Dense convolution matrix of ε(r), shape ``[..., Nh, Nh]``.
    Axx, Axy, Ayx, Ayy : Block or None
        TVF anisotropy blocks from :func:`compute_A`. Must all be provided
        or all be ``None``; mixing raises no error but returns Q0.

    Returns
    -------
    Q : Block2x2
        Full Q matrix (= Q0 when no A blocks are given).
    """
    Q0 = compute_Q0(Kx, Ky, epsilon_conv)
    if Axx is None or Axy is None or Ayx is None or Ayy is None:
        return Q0
    Qfact = compute_Qfact(epsilon_conv, Axx, Axy, Ayx, Ayy)
    return Q0 + Qfact


def compute_P(Kx: Block, Ky: Block, epsilon_conv: Block) -> Block2x2:
    """
    Assemble the P matrix for an isotropic patterned layer.

    For a non-magnetic (μ = 1) isotropic medium the P operator is:

        P = [[ −Kx·ε⁻¹·Ky,    −I + Kx·ε⁻¹·Kx ],
              [  I − Ky·ε⁻¹·Ky,   Ky·ε⁻¹·Kx   ]]

    The ε⁻¹ action is applied via ``Block.solve()`` to avoid materialising
    the explicit inverse when ε is DENSE.

    Parameters
    ----------
    Kx : Block
        Diagonal Block of x-components of the in-plane wavevector, shape
        ``[..., Nh]``.
    Ky : Block
        Diagonal Block of y-components of the in-plane wavevector, shape
        ``[..., Nh]``.
    epsilon_conv : Block
        Dense Block of the Toeplitz convolution matrix of ε(r), shape
        ``[..., Nh, Nh]``.

    Returns
    -------
    P : Block2x2
        P matrix. Entry kinds promote to DENSE when epsilon_conv is DENSE.
    """
    a = -Kx @ epsilon_conv.solve(Ky)
    b = -Kx.eye_like() + Kx @ epsilon_conv.solve(Kx)
    c =  Ky.eye_like() - Ky @ epsilon_conv.solve(Ky)
    d =  Ky @ epsilon_conv.solve(Kx)
    return Block2x2(a, b, c, d)


def compute_isotropic(epsilon_grid: torch.Tensor,
                      m_flat: torch.Tensor, n_flat: torch.Tensor,
                      kx: torch.Tensor, ky: torch.Tensor,
                      tvf: TVF | None = None) -> Tuple[Block2x2, Block2x2]:
    """
    Build the P and Q operators for an isotropic patterned layer.

    Constructs the Toeplitz convolution matrix of ε(r) once and dispatches
    to :func:`compute_P` and :func:`compute_Q`.  When ``tvf`` is provided
    the TVF anisotropy blocks are computed and folded into Q via
    :func:`compute_A` and :func:`compute_Qfact`.

    Parameters
    ----------
    epsilon_grid : torch.Tensor
        Permittivity sampled on the real-space unit-cell grid, shape
        ``[..., Ny, Nx]``.
    m_flat : torch.Tensor
        Integer harmonic indices along x (b1 direction), shape ``[Nh]``.
    n_flat : torch.Tensor
        Integer harmonic indices along y (b2 direction), shape ``[Nh]``.
    kx : torch.Tensor
        x-components of the in-plane wavevectors, normalised by k0,
        shape ``[..., Nh]``.
    ky : torch.Tensor
        y-components of the in-plane wavevectors, normalised by k0,
        shape ``[..., Nh]``.
    tvf : TVF or None, optional
        Configured TVF instance for Li factorization. ``None`` (default)
        uses the plain Laurent rule (no correction).

    Returns
    -------
    P : Block2x2
        P operator; each entry is a DENSE Block of shape ``[..., Nh, Nh]``.
    Q : Block2x2
        Q operator; each entry is a DENSE Block of shape ``[..., Nh, Nh]``.
    """
    epsilon_conv = Block(Block.DENSE, convolution_matrix(epsilon_grid, m_flat, n_flat))
    Kx = Block(Block.DIAG, kx)
    Ky = Block(Block.DIAG, ky)

    P = compute_P(Kx, Ky, epsilon_conv)
    if tvf is None:
        Q = compute_Q(Kx, Ky, epsilon_conv)
    else:
        Axx, Axy, Ayx, Ayy = compute_A(epsilon_grid, m_flat, n_flat, tvf)
        Q = compute_Q(Kx, Ky, epsilon_conv, Axx, Axy, Ayx, Ayy)
    return P, Q
