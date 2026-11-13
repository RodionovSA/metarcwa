# metarcwa/solver/layersolver/isotropic.py
"""
isotropic — P and Q operators for patterned isotropic layers
=============================================================

**Scope:** non-magnetic (μ = 1) patterned layers with isotropic permittivity.
All operators are assembled from the Laurent convolution matrix of ε(r) and,
optionally, the TVF anisotropy correction blocks (Li factorization rules).

Four internal builders, one combiner, and one top-level entry point:

  compute_Q0(Kx, Ky, epsilon_conv)            → Block2x2
      Base Q matrix without TVF correction.

  compute_A(Tx, Ty, m_flat, n_flat) → (Axx, Axy, Ayx, Ayy)
      TVF anisotropy blocks from precomputed tangent vector field components.

  compute_Qfact(epsilon_conv, Axx, Axy, Ayx, Ayy) → Block2x2
      Factorization correction to Q0 derived from the A blocks.

  compute_Q(Kx, Ky, epsilon_conv, [A blocks])  → Block2x2
      Full Q = Q0 (+ Qfact if TVF blocks are provided).

  compute_P(Kx, Ky, epsilon_conv)              → Block2x2
      P matrix; ε⁻¹ factored into every term via Block.solve().

  compute_isotropic(epsilon_grid, m_flat, n_flat, kx, ky, tvf_fields=None) → (P, Q)
      Top-level entry: builds ε_conv once and returns both operators.
      ``kx``/``ky`` are ``[..., Nh]`` tensors (wrapped as DIAG) or ``Block``s.
      ``tvf_fields`` is a precomputed ``(Tx, Ty)`` pair, not a ``TVF``
      instance — callers compute the field once (see ``LayerSolver._patterned``)
      and may pass a batch-1 field that broadcasts against a batched ε_conv.

All functions use the exp(−j ω t) time convention.
"""

import torch
from typing import Tuple

from metarcwa.solver.blockmatrix import Block, Block2x2
from metarcwa.solver.convolution import convolution_matrix
from metarcwa._dtypes import _REAL_TO_COMPLEX


def compute_Q0(Kx: Block, Ky: Block, epsilon_conv: Block) -> Block2x2:
    """
    Assemble the base Q matrix for an isotropic patterned layer.

    For a non-magnetic (μ = 1) isotropic medium the Q operator has the
    2×2 block form:

        Q₀ = [[ −Kx·Ky,       Kx²−ε ],
               [  ε−Ky²,       Ky·Kx ]]

    where Kx, Ky are the in-plane wavevector operators and ε is the
    convolution (Toeplitz) matrix of the permittivity. DENSE Kx, Ky need
    not commute, so ``Ky·Kx`` is computed explicitly.

    Parameters
    ----------
    Kx : Block
        x-wavevector operator: DIAG ``[..., Nh]`` or DENSE ``[..., Nh, Nh]``.
    Ky : Block
        y-wavevector operator: DIAG ``[..., Nh]`` or DENSE ``[..., Nh, Nh]``.
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
    d = Ky @ Kx        # equals -a only when Kx, Ky commute
    return Block2x2(a, b, c, d)


def compute_A(Tx: torch.Tensor, Ty: torch.Tensor, 
              m_flat: torch.Tensor, n_flat: torch.Tensor) -> Tuple[Block, Block, Block, Block]:
    """
    Compute the TVF anisotropy correction blocks for the Li factorization.

    The tangent vector field (Tx, Ty) encodes the local polarization
    direction at every grid point.  The anisotropy blocks are the
    convolution matrices of the outer-product components of the TVF:

        Axx  ←  Conv(|Ty|²)          Ayy  ←  Conv(|Tx|²)
        Axy  ←  Conv(Tx* · Ty)       Ayx  ←  Conv(Tx · Ty*)

    Parameters
    ----------
    Tx, Ty : tangent vector field components, shape [B, Ny, Nx].
             B may be 1 (wavelength-independent field, broadcast downstream)
             or match the batch of epsilon_conv.
    m_flat : torch.Tensor
        Integer harmonic indices along x, shape ``[Nh]``.
    n_flat : torch.Tensor
        Integer harmonic indices along y, shape ``[Nh]``.

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
    axx = Ty.abs() ** 2
    axy = Tx.conj() * Ty
    ayx = Tx * Ty.conj()
    ayy = Tx.abs() ** 2

    Axx = Block(Block.DENSE, convolution_matrix(axx, m_flat, n_flat))
    Axy = Block(Block.DENSE, convolution_matrix(axy, m_flat, n_flat))
    Ayx = Block(Block.DENSE, convolution_matrix(ayx, m_flat, n_flat))
    Ayy = Block(Block.DENSE, convolution_matrix(ayy, m_flat, n_flat))

    return Axx, Axy, Ayx, Ayy


def compute_Qfact(epsilon_conv: Block, epsilon_inv_conv: Block,
                  Axx: Block, Axy: Block, Ayx: Block, Ayy: Block) -> Block2x2:
    """
    Assemble the TVF factorization correction to the Q matrix.

    The correction encodes the difference between the inverse-rule and
    direct-rule Fourier factorizations, weighted by the TVF anisotropy
    blocks.  The four entries are:

        Qfact = [[ -(ε − (1/ε)⁻¹)·Ayx,   (ε − (1/ε)⁻¹)·Ayy ],
                 [ -(ε − (1/ε)⁻¹)·Axx,   (ε − (1/ε)⁻¹)·Axy ]]

    Parameters
    ----------
    epsilon_conv : Block
        Dense convolution matrix of ε(r), shape ``[..., Nh, Nh]``.
    epsilon_inv_conv : Block
        Dense convolution matrix of 1/ε(r), shape ``[..., Nh, Nh]``.
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
    inv_Ayx, inv_Ayy, inv_Axx, inv_Axy = epsilon_inv_conv.solve_many(Ayx, Ayy, Axx, Axy)
    a_fact = -epsilon_conv @ Ayx + inv_Ayx
    b_fact =  epsilon_conv @ Ayy - inv_Ayy
    c_fact = -epsilon_conv @ Axx + inv_Axx
    d_fact =  epsilon_conv @ Axy - inv_Axy
    return Block2x2(a_fact, b_fact, c_fact, d_fact)


def compute_Q(Kx: Block, Ky: Block, epsilon_conv: Block, epsilon_inv_conv: Block,
              Axx: Block, Axy: Block, Ayx: Block, Ayy: Block) -> Block2x2:
    """
    Assemble the full TVF-corrected Q matrix: ``Q0 + Qfact``.

    Callers that want the plain Laurent rule (no TVF correction) should call
    :func:`compute_Q0` directly instead — see :func:`compute_isotropic`,
    which dispatches between the two based on whether a TVF field is given.

    Parameters
    ----------
    Kx : Block
        Diagonal Block of x-wavevector components, shape ``[..., Nh]``.
    Ky : Block
        Diagonal Block of y-wavevector components, shape ``[..., Nh]``.
    epsilon_conv : Block
        Dense convolution matrix of ε(r), shape ``[..., Nh, Nh]``.
    epsilon_inv_conv : Block
        Dense convolution matrix of 1/ε(r), shape ``[..., Nh, Nh]``.
    Axx, Axy, Ayx, Ayy : Block
        TVF anisotropy blocks from :func:`compute_A`.

    Returns
    -------
    Q : Block2x2
        Full Q matrix.
    """
    Q0 = compute_Q0(Kx, Ky, epsilon_conv)
    Qfact = compute_Qfact(epsilon_conv, epsilon_inv_conv, Axx, Axy, Ayx, Ayy)
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
        x-wavevector operator: DIAG ``[..., Nh]`` or DENSE ``[..., Nh, Nh]``.
    Ky : Block
        y-wavevector operator: DIAG ``[..., Nh]`` or DENSE ``[..., Nh, Nh]``.
    epsilon_conv : Block
        Dense Block of the Toeplitz convolution matrix of ε(r), shape
        ``[..., Nh, Nh]``.

    Returns
    -------
    P : Block2x2
        P matrix. Entry kinds promote to DENSE when epsilon_conv is DENSE.
    """
    eps_inv_Ky, eps_inv_Kx = epsilon_conv.solve_many(Ky, Kx)
    a = -Kx @ eps_inv_Ky
    b = -Kx.eye_like() + Kx @ eps_inv_Kx
    c =  Ky.eye_like() - Ky @ eps_inv_Ky
    d =  Ky @ eps_inv_Kx
    return Block2x2(a, b, c, d)


def compute_isotropic(epsilon_grid: torch.Tensor,
                      m_flat: torch.Tensor, n_flat: torch.Tensor,
                      kx: torch.Tensor | Block, ky: torch.Tensor | Block,
                      tvf_fields: tuple[torch.Tensor, torch.Tensor] | None = None,
                      return_parts: bool = False,
                      ) -> Tuple[Block2x2, Block2x2] | Tuple[Block2x2, Block2x2, Block, Block2x2 | None]:
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
    kx : torch.Tensor or Block
        x-components of the in-plane wavevectors, normalised by k0: either
        a tensor of shape ``[..., Nh]`` (wrapped as a DIAG Block), or a
        Block — DIAG ``[..., Nh]`` or DENSE ``[..., Nh, Nh]``.
    ky : torch.Tensor or Block
        y-components, same conventions as ``kx``.
    tvf_fields: Tx, Ty tangent vector field components, shape [B, Ny, Nx].
             B may be 1 (wavelength-independent field, broadcast downstream)
             or match the batch of epsilon_conv. ``None`` (default) uses the plain 
             Laurent rule (no correction).
    return_parts : bool, optional
        Also return the convolution matrix ``[[eps]]`` and the TVF
        correction ``Qfact = Q - Q0`` (``None`` without TVF), i.e.
        ``(P, Q, epsilon_conv, Qfact)``. The PML scattered-field source is
        built from them (``metarcwa.solver.scattered``).

    Returns
    -------
    P : Block2x2
        P operator; each entry is a DENSE Block of shape ``[..., Nh, Nh]``.
    Q : Block2x2
        Q operator; each entry is a DENSE Block of shape ``[..., Nh, Nh]``.
    """
    Kx = kx if isinstance(kx, Block) else Block(Block.DIAG, kx)
    Ky = ky if isinstance(ky, Block) else Block(Block.DIAG, ky)
    epsilon_grid = epsilon_grid.to(dtype=_REAL_TO_COMPLEX[epsilon_grid.real.dtype])
    k_batch_ndim = Kx.data.ndim - (2 if Kx.kind == Block.DENSE else 1)
    n_extra = k_batch_ndim - (epsilon_grid.ndim - 2)
    if n_extra > 0:
        epsilon_grid = epsilon_grid.reshape(
            *epsilon_grid.shape[:-2], *([1] * n_extra), *epsilon_grid.shape[-2:]
        )
    epsilon_conv = Block(Block.DENSE, convolution_matrix(epsilon_grid, m_flat, n_flat))

    P = compute_P(Kx, Ky, epsilon_conv)
    Qfact = None
    if tvf_fields is None:
        Q = compute_Q0(Kx, Ky, epsilon_conv)
    else:
        Tx, Ty = tvf_fields
        epsilon_inv_conv = Block(Block.DENSE,
                                 convolution_matrix(1.0 / epsilon_grid, m_flat, n_flat))
        Axx, Axy, Ayx, Ayy = compute_A(Tx, Ty, m_flat, n_flat)
        if return_parts:
            Qfact = compute_Qfact(epsilon_conv, epsilon_inv_conv, Axx, Axy, Ayx, Ayy)
            Q = compute_Q0(Kx, Ky, epsilon_conv) + Qfact
        else:
            Q = compute_Q(Kx, Ky, epsilon_conv, epsilon_inv_conv, Axx, Axy, Ayx, Ayy)
    if return_parts:
        return P, Q, epsilon_conv, Qfact
    return P, Q
