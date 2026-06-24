# metarcwa/solver/smatrix.py
"""
smatrix — S-matrix building blocks for RCWA layer stacks
=========================================================

Three public functions, in dependency order:

  S_boundary(WL, VL, WR, VR)             → Block2x2
      Interface S-matrix from E/H-mode boundary continuity.

  S_prop(lam, wvl, d)                     → Block2x2
      Propagation S-matrix for a single homogeneous layer.

  S_layer(W0, V0, W, V, lam, d, wvl)         → Block2x2
      Full layer S-matrix: boundary ⋆ propagation ⋆ boundary.

All S-matrices follow the convention:

  [[S11, S12],   =   [[reflection from left,  transmission from right],
   [S21, S22]]        [transmission from left, reflection from right ]]

and compose via the Redheffer star product (⋆), which is associative but
not commutative — left-to-right physical order must be preserved.

All functions use the exp(−j ω t) time convention.
"""

import torch

from metarcwa.solver.blockmatrix import Block, Block2x2


def S_boundary(WL: Block2x2, VL: Block2x2,
               WR: Block2x2, VR: Block2x2) -> Block2x2:
    """
    Compute the S-matrix at a planar interface between two layers.

    Derives the interface scattering matrix from the continuity of tangential
    E and H fields.  For mode amplitudes (c⁺_L, c⁻_R) incident on the
    interface and (c⁻_L, c⁺_R) outgoing, the continuity conditions give:

        [[WL, −WR], [VL, VR]] · [c⁻_L; c⁺_R]
            = [[−WL, WR], [VL, VR]] · [c⁺_L; c⁻_R]

    so  S = left⁻¹ · right  via :meth:`Block2x2.solve`.

    When WL = WR and VL = VR (same medium on both sides) the result is the
    star-product identity ``[[0, I], [I, 0]]`` (perfect transmission,
    zero reflection).

    Parameters
    ----------
    WL : Block2x2
        E-mode matrix of the left layer (columns = eigenvectors of E-field).
    VL : Block2x2
        H-mode matrix of the left layer.
    WR : Block2x2
        E-mode matrix of the right layer.
    VR : Block2x2
        H-mode matrix of the right layer.

    Returns
    -------
    S : Block2x2
        Interface S-matrix; each entry is a Block2x2 (same nesting depth
        as the input mode matrices).
    """
    left  = Block2x2(WL, -WR, VL,  VR)
    right = Block2x2(-WL, WR, VL,  VR)
    return left.solve(right)


def S_prop(lam: torch.Tensor, wvl: torch.Tensor, d: torch.Tensor) -> Block2x2:
    """
    Compute the propagation S-matrix for a homogeneous layer.

    For a layer of (normalised) thickness ``d`` with modal exponents ``lam``,
    the propagation factor for each mode is:

        Xd = exp(lam · k₀ · d),    k₀ = 2π / wvl

    There is no inter-mode coupling during propagation, so the S-matrix is:

        S_prop = [[0,  Xd],
                  [Xd,  0]]

    where 0 and Xd are 2×2 block-diagonal operators (each built from the
    two Nh-sized halves of the ``lam`` vector).

    At zero thickness (d = 0) the result is the star-product identity
    ``[[0, I], [I, 0]]``.

    Parameters
    ----------
    lam : torch.Tensor
        Modal exponents lam = 1j·kz. Shape ``[..., 2Nh]``; the first Nh
        entries correspond to the first polarisation block (e.g. TE) and
        the last Nh to the second block (e.g. TM).
    wvl : torch.Tensor
        Free-space wavelength (same units as ``d``). Scalar or broadcastable
        against the batch dimensions of ``lam``.
    d : torch.Tensor
        Layer thickness (same units as ``wvl``). Scalar or broadcastable.

    Returns
    -------
    S : Block2x2
        Propagation S-matrix; each top-level entry is a Block2x2 of
        ``Block(DIAG, ...)`` sub-entries of shape ``[..., Nh]``.
    """
    k0 = 2 * torch.pi / wvl
    xd = torch.exp(lam * k0 * d)          # [..., 2Nh]
    Nh = xd.shape[-1] // 2
    kw = dict(device=xd.device, dtype=xd.dtype)
    Xd = Block2x2(
        Block(Block.DIAG, xd[..., :Nh]),
        Block.zeros(**kw),
        Block.zeros(**kw),
        Block(Block.DIAG, xd[..., Nh:]),
    )
    Z = Xd.zeros_like()
    return Block2x2(Z, Xd, Xd, Z)


def S_layer(W0: Block2x2, V0: Block2x2, W: Block2x2, V: Block2x2,
            lam: torch.Tensor, d: torch.Tensor, wvl: torch.Tensor) -> Block2x2:
    """
    Compute the S-matrix of a single layer embedded in a homogeneous background.

    Cascades three S-matrices via the Redheffer star product:

        S = S_in ⋆ S_prop ⋆ S_out

    where:

    - ``S_in``  = S_boundary(W₀ = I, V₀, W, V)  — background → layer interface
    - ``S_prop`` = propagation through the layer
    - ``S_out`` = S_boundary reversed — layer → background interface

    Because the background medium is the same on both sides, the right
    interface S-matrix is obtained by block-swapping ``S_in``:

        S_out = [[S_in.d, S_in.c],
                 [S_in.b, S_in.a]]

    This mirror symmetry holds **only** when the same background fills both
    sides.  For asymmetric embeddings call :func:`S_boundary` explicitly.

    Parameters
    ----------
    W0 : Block2x2
        E-mode matrix of the background medium.
    V0 : Block2x2
        H-mode matrix of the background medium.
    W : Block2x2
        E-mode matrix of the layer (columns = E-field eigenvectors).
        Use ``Block2x2.identity()`` for homogeneous layers (W = I).
    V : Block2x2
        H-mode matrix of the layer.
    lam : torch.Tensor
        Modal exponents lam = 1j·kz. Shape ``[..., 2Nh]``.
    d : torch.Tensor
        Layer thickness (same units as ``wvl``).
    wvl : torch.Tensor
        Free-space wavelength (same units as ``d``).

    Returns
    -------
    S : Block2x2
        Full S-matrix of the layer; same nesting depth as the input
        mode matrices.

    Notes
    -----
    call :func:`S_boundary` directly for general interface pairs.
    """
    S_in  = S_boundary(W0, V0, W, V)
    S_p   = S_prop(lam, wvl, d)
    S_out = Block2x2(S_in.d, S_in.c,
                     S_in.b, S_in.a)
    return S_in.star(S_p).star(S_out)
