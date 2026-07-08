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


# ---------------------------------------------------------------------------
# Internal helpers for S_boundary
# ---------------------------------------------------------------------------

def _find_Nh(*entries) -> int | None:
    """Return the first non-SCALAR leaf block size from nested Block2x2 entries."""
    def _scan(e):
        if not hasattr(e, 'a'):           # leaf Block
            return e.n                    # None for SCALAR, int for DIAG/DENSE
        for sub in (e.a, e.b, e.c, e.d):
            n = _scan(sub)
            if n is not None:
                return n
        return None
    for e in entries:
        n = _scan(e)
        if n is not None:
            return n
    return None


def _dense_to_nested_block2x2(M: torch.Tensor, Nh: int) -> Block2x2:
    """Convert a ``[..., 4Nh, 4Nh]`` dense tensor to Block2x2(Block2x2(DENSE, …), …).

    Partitions the 4Nh×4Nh matrix into 4 outer blocks (each 2Nh×2Nh) and then
    each into 4 inner Nh×Nh DENSE blocks.  This reproduces the two-level nesting
    that ``Block2x2.solve`` would produce for DIAG/SCALAR inputs.
    """
    N2 = 2 * Nh
    def _inner(sub):
        return Block2x2(
            Block(Block.DENSE, sub[..., :Nh, :Nh]),
            Block(Block.DENSE, sub[..., :Nh, Nh:]),
            Block(Block.DENSE, sub[..., Nh:, :Nh]),
            Block(Block.DENSE, sub[..., Nh:, Nh:]),
        )
    return Block2x2(
        _inner(M[..., :N2, :N2]),
        _inner(M[..., :N2, N2:]),
        _inner(M[..., N2:, :N2]),
        _inner(M[..., N2:, N2:]),
    )


def _boundary_dense(left: Block2x2, right: Block2x2, Nh: int) -> Block2x2:
    """Direct dense solve of the boundary system (safe for DENSE mode matrices).

    Densifies ``left``/``right`` to ``[..., 4Nh, 4Nh]`` and solves once. Used
    whenever any leaf block is DENSE (eig-derived patterned-layer modes),
    where nested Schur-complement solves can hit singular inner sub-blocks
    (see :func:`S_boundary`).
    """
    L   = left.to_dense(Nh)
    R   = right.to_dense(Nh)
    S_d = torch.linalg.solve(L, R)
    return _dense_to_nested_block2x2(S_d, Nh)


def _all_diag_leaves(WL: Block2x2, VL: Block2x2, WR: Block2x2, VR: Block2x2) -> bool:
    """True iff every leaf Block of WL, VL, WR, VR is SCALAR or DIAG (no DENSE)."""
    return all(
        entry.kind != Block.DENSE
        for op in (WL, VL, WR, VR)
        for entry in (op.a, op.b, op.c, op.d)
    )


def _boundary_diag_fast(WL: Block2x2, VL: Block2x2, WR: Block2x2, VR: Block2x2,
                        Nh: int) -> Block2x2:
    """Exact O(Nh) boundary solve when every leaf block is SCALAR/DIAG.

    With no DENSE leaves anywhere, none of WL/VL/WR/VR mixes harmonics, so
    the full ``4Nh×4Nh`` boundary system decouples into ``Nh`` independent
    ``4×4`` systems — one per Fourier harmonic, mixing only the two
    polarizations within that harmonic. Solving all ``Nh`` systems as one
    batched ``torch.linalg.solve`` (over an added ``Nh`` axis) is O(Nh)
    instead of O(Nh³), and — unlike a nested Schur-complement decomposition —
    is a direct solve of the true per-harmonic system, so it has no
    singular-sub-block failure mode (verified: this matches
    :func:`_boundary_dense` exactly, including at normal incidence where
    ``V.d`` has exact-zero diagonal entries and a naive
    ``Block2x2.solve()`` on DIAG blocks would return NaN).

    Row/column layout of each per-harmonic 4×4 block (from the boundary
    continuity condition in :func:`S_boundary`'s docstring):
    rows = [E-TE, E-TM, H-TE, H-TM], cols(left/right, matching) =
    [c⁻_L-TE, c⁻_L-TM, c⁺_R-TE, c⁺_R-TM] resp. [c⁺_L-TE, c⁺_L-TM, c⁻_R-TE, c⁻_R-TM].
    The solved result stays block-diagonal by harmonic (product of two
    block-diagonal matrices), so it is reassembled directly as a nested
    ``Block2x2(Block2x2(DIAG, ...), ...)`` — the same two-level nesting
    ``S_prop``/``_dense_to_nested_block2x2`` already use, so it composes
    via ``star()`` unchanged.
    """
    def d(blk: Block) -> torch.Tensor:
        return blk.to(Block.DIAG, Nh).data

    left_rows = [
        [ d(WL.a),  d(WL.b), -d(WR.a), -d(WR.b)],
        [ d(WL.c),  d(WL.d), -d(WR.c), -d(WR.d)],
        [ d(VL.a),  d(VL.b),  d(VR.a),  d(VR.b)],
        [ d(VL.c),  d(VL.d),  d(VR.c),  d(VR.d)],
    ]
    right_rows = [
        [-d(WL.a), -d(WL.b),  d(WR.a),  d(WR.b)],
        [-d(WL.c), -d(WL.d),  d(WR.c),  d(WR.d)],
        [ d(VL.a),  d(VL.b),  d(VR.a),  d(VR.b)],
        [ d(VL.c),  d(VL.d),  d(VR.c),  d(VR.d)],
    ]

    def _assemble(rows: list[list[torch.Tensor]]) -> torch.Tensor:
        batch = torch.broadcast_shapes(*(t.shape for r in rows for t in r))
        rows = [[t.expand(batch) for t in r] for r in rows]
        return torch.stack([torch.stack(r, dim=-1) for r in rows], dim=-2)  # [..., Nh, 4, 4]

    L4 = _assemble(left_rows)
    R4 = _assemble(right_rows)
    X4 = torch.linalg.solve(L4, R4)          # batched over Nh: O(Nh), not O(Nh^3)

    def leaf(i: int, j: int) -> Block:
        return Block(Block.DIAG, X4[..., i, j])

    def inner(r0: int, r1: int, c0: int, c1: int) -> Block2x2:
        return Block2x2(leaf(r0, c0), leaf(r0, c1), leaf(r1, c0), leaf(r1, c1))

    return Block2x2(
        inner(0, 1, 0, 1),   # S11
        inner(0, 1, 2, 3),   # S12
        inner(2, 3, 0, 1),   # S21
        inner(2, 3, 2, 3),   # S22
    )


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
    Nh = _find_Nh(WL, VL, WR, VR)
    if Nh is None:
        # All-SCALAR inputs: Schur complement is safe (no singular sub-blocks).
        return left.solve(right)
    if _all_diag_leaves(WL, VL, WR, VR):
        # All-DIAG/SCALAR (e.g. homogeneous/vacuum boundary): the system
        # decouples per harmonic -- solve it as Nh independent 4x4 systems,
        # O(Nh) instead of O(Nh^3). See _boundary_diag_fast for why this is
        # exact (unlike a naive Block2x2.solve() Schur path on DIAG blocks,
        # which returns NaN at normal incidence -- V.d has exact-zero
        # diagonal entries there).
        return _boundary_diag_fast(WL, VL, WR, VR, Nh)
    # Use a direct 4Nh×4Nh dense solve to avoid singular sub-block issues that
    # arise when DENSE mode matrices (e.g. from eigsolver) have rank-deficient
    # inner blocks (V.d) as required by the nested Schur complement.
    return _boundary_dense(left, right, Nh)


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
