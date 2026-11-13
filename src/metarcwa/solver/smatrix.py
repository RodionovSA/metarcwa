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

Affine S-matrices (layers with an internal source, used by the PML
scattered-field formulation) are carried by :class:`AffineS`, whose
:meth:`AffineS.star` extends the Redheffer star product to the source
vectors (``docs/PML.md`` "S-matrices with sources").

All functions use the exp(−j ω t) time convention.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from metarcwa.solver.blockmatrix import Block, Block2x2, _leaf_size, apply_vec


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

    Solve strategy is selected automatically by :meth:`Block2x2.solve` from
    the leaf representation of WL/VL/WR/VR:

    - all-SCALAR → Schur complement (size-agnostic, safe).
    - all-DIAG/SCALAR (homogeneous/vacuum boundaries) → per-harmonic O(n)
      batched solve; robust at normal incidence where V.d has exact-zero
      diagonal entries.
    - any DENSE leaf (patterned/eig layers) → full dense solve; avoids
      singular inner sub-blocks when layer V.d is rank-deficient.

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
    left  = Block2x2(WL, -WR, VL, VR)
    right = Block2x2(-WL, WR, VL, VR)
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
    wvl = torch.as_tensor(wvl)
    if wvl.ndim < lam.ndim:
        wvl = wvl.reshape(*wvl.shape, *([1] * (lam.ndim - wvl.ndim)))
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

    The right interface S-matrix is obtained by block-swapping ``S_in``
    instead of a second :func:`S_boundary` solve:

        S_out = [[S_in.d, S_in.c],
                 [S_in.b, S_in.a]]

    This mirror trick is an identity of :func:`S_boundary` that holds for
    any two media, not only equal ones — see ``docs/matrixexp.md``
    "Accuracy and conditioning" for the general form and a use against two
    different media. Call :func:`S_boundary` explicitly whenever the two
    faces should reference genuinely different backgrounds.

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


def _add(x: torch.Tensor | None, y: torch.Tensor | None) -> torch.Tensor | None:
    """``x + y`` with ``None`` standing for a zero vector."""
    if x is None:
        return y
    if y is None:
        return x
    return x + y


@dataclass(frozen=True)
class AffineS:
    """S-matrix with source vectors: ``(c_L-, c_R+) = S (c_L+, c_R-) + (e_L, e_R)``.

    ``(e_L, e_R)`` are the waves a layer emits through its left / right face
    when nothing comes in (``docs/PML.md`` "Affine S-matrix"). ``S`` is the
    ordinary S-matrix, unaffected by the source.

    Attributes
    ----------
    S : Block2x2
        The S-matrix (entries act on ``2Nh`` mode-amplitude vectors).
    e_L, e_R : torch.Tensor or None
        Source vectors, shape ``[..., 2Nh, ncol]`` (one column per
        excitation); ``None`` means zero (no source).
    """
    S: Block2x2
    e_L: torch.Tensor | None = None
    e_R: torch.Tensor | None = None

    def star(self, o: "AffineS") -> "AffineS":
        """Extended Redheffer star product ``self`` (left) ⋆ ``o`` (right).

        The S-matrix part is the ordinary star product. Solving for the
        waves bouncing between the two parts gives the source part::

            e_L = e_AL + S_A12 (I - S_B11 S_A22)^-1 (e_BL + S_B11 e_AR)
            e_R = e_BR + S_B21 (I - S_A22 S_B11)^-1 (e_AR + S_A22 e_BL)

        With both sources ``None`` this is :meth:`Block2x2.star` (structured
        leaves are kept). Otherwise the two inverses are the ones the star
        product needs anyway, so each is factorized once (densely) and
        applied to the matrix and the vector right-hand sides together.
        """
        A, B = self.S, o.S
        x = _add(o.e_L, None if self.e_R is None else apply_vec(B.a, self.e_R))
        y = _add(self.e_R, None if o.e_L is None else apply_vec(A.d, o.e_L))
        if x is None:                       # then y is None too: no sources
            return AffineS(A.star(B), self.e_L, o.e_R)

        n = _leaf_size(B.a, x.shape[-2])
        P = B.a @ A.d                       # S11^B S22^A
        Q = A.d @ B.a                       # S22^A S11^B
        Za, Zb, zx = _solve_together(P.eye_like() - P, n, (B.a, B.b), x)
        Wc, Wdb, wy = _solve_together(Q.eye_like() - Q, n, (A.c, A.d @ B.b), y)
        S = Block2x2(
            A.a + A.b @ Za @ A.c,  A.b @ Zb,
            B.c @ Wc,              B.d + B.c @ Wdb,
        )
        return AffineS(S, _add(self.e_L, apply_vec(A.b, zx)), _add(o.e_R, apply_vec(B.c, wy)))


def _solve_together(lhs, n: int, mats, vec: torch.Tensor):
    """Solve ``lhs^-1`` against several operators and one vector with a
    single dense factorization. Returns the solved operators (DENSE leaves,
    nested like the inputs) followed by the solved vector."""
    L = lhs.to_dense(n)
    dense = [m.to_dense(n) for m in mats]
    dt = torch.promote_types(L.dtype, vec.dtype)
    cols = [d.to(dt) for d in dense] + [vec.to(dt)]
    batch = torch.broadcast_shapes(L.shape[:-2], *(c.shape[:-2] for c in cols))
    rhs = torch.cat([c.expand(*batch, *c.shape[-2:]) for c in cols], dim=-1)
    X = torch.linalg.solve(L.to(dt).expand(*batch, *L.shape[-2:]), rhs)
    size = L.shape[-1]
    out = [Block2x2.from_dense(X[..., k * size:(k + 1) * size], m) for k, m in enumerate(mats)]
    return (*out, X[..., len(mats) * size:])
