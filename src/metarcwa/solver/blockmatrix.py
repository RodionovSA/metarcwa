# metarcwa/solver/blockmatrix.py
"""
blockmatrix — structured 2D operators for RCWA solvers
=======================================================

Block
-----
Represents a single n×n operator in one of three representations:

  SCALAR  c·I      data [*B]          1 scalar per batch item  — size-agnostic
  DIAG    diag(v)  data [*B, n]       n scalars per batch item
  DENSE   full     data [*B, n, n]    n² scalars per batch item

Arithmetic dispatches to the cheapest representation that is exact:

  SCALAR @ SCALAR → SCALAR     SCALAR + DIAG  → DIAG
  SCALAR @ DIAG   → DIAG       DIAG   + DIAG  → DIAG
  DIAG   @ DIAG   → DIAG       DIAG   + DENSE → DENSE  (unavoidable promotion)
  DIAG   @ DENSE  → DENSE      (row-scaling, not a full matmul)

Memory per batch item vs always-DENSE (float32):

  n=64   DIAG saves    64×, SCALAR saves     4 096×
  n=256  DIAG saves   256×, SCALAR saves    65 536×
  n=1024 DIAG saves 1 024×, SCALAR saves 1 048 576×

For a realistic RCWA run with batch size B=500 and n=256:
  DENSE → 125 MB    DIAG → 500 KB    SCALAR → 2 KB

Kind is promoted only when unavoidable. `n` is a property derived from
`data.shape[-1]`; SCALAR carries no fixed `n` and is compatible with any size.

Block2x2
--------
A 2×2 operator [[a, b], [c, d]] where each entry is a Block. Arithmetic
follows the standard 2×2 block rule, so structured entries stay structured:

  SCALAR identity @ anything  → same kinds in result
  block-diagonal (b=0, c=0)   → inv reduces to two independent Block.inv() calls

Inverse uses the Schur complement of the bottom-right block d:

  S   = a - b · d⁻¹ · c
  M⁻¹ = [[ S⁻¹,         -S⁻¹ · b · d⁻¹             ],
          [ -d⁻¹ · c · S⁻¹,  d⁻¹ + d⁻¹ · c · S⁻¹ · b · d⁻¹ ]]

This is cheap when d is DIAG or SCALAR (d⁻¹ is elementwise).
"""

from __future__ import annotations
import torch
from typing import Protocol, Tuple, Union, runtime_checkable


@runtime_checkable
class Entry(Protocol):
    """Structural protocol satisfied by both Block and Block2x2."""
    def __add__(self, o): ...
    def __sub__(self, o): ...
    def __neg__(self): ...
    def __matmul__(self, o): ...
    def inv(self): ...
    def solve(self, rhs): ...
    def eye_like(self): ...
    def zeros_like(self): ...

class Block:
    """A 2D operator in one of three representations, dispatched internally.

        SCALAR : c·I     data shape [*B]        — size-agnostic (identity at any n)
        DIAG   : diag(d) data shape [*B, n]
        DENSE  : full     data shape [*B, n, n]

    Batch dims [*B] broadcast across operations; only the harmonic size n must match.
    """
    SCALAR = 0
    DIAG   = 1
    DENSE  = 2

    def __init__(self, kind: int, data: torch.Tensor):
        if kind == Block.SCALAR:
            pass                                   # no harmonic axis to validate
        elif kind == Block.DIAG:
            if data.ndim < 1:
                raise ValueError(f"DIAG data must be >=1D, got {tuple(data.shape)}")
        elif kind == Block.DENSE:
            if data.ndim < 2:
                raise ValueError(f"DENSE data must be >=2D, got {tuple(data.shape)}")
            if data.shape[-1] != data.shape[-2]:
                raise ValueError(f"DENSE data not square: {tuple(data.shape)}")
        else:
            raise ValueError(f"unknown kind {kind}")
        self.kind = kind
        self.data = data

    # ---- intrinsic size: derived from data; None for SCALAR ----
    @property
    def n(self) -> int | None:
        return None if self.kind == Block.SCALAR else self.data.shape[-1]
    
    @property
    def shape(self) -> torch.Size:
        return self.data.shape

    def _check_n(self, o: "Block"):
        if self.kind == Block.SCALAR or o.kind == Block.SCALAR:
            return                                 # scalar matches any size
        if self.n != o.n:
            raise ValueError(f"size mismatch: {self.n} vs {o.n}")

    def _sized_n(self, o: "Block") -> int | None:
        return self.n if self.n is not None else o.n   # the sized sibling, if any

    # ---- promotion: upward only; target n required only from SCALAR ----
    def to(self, k: int, n: int | None = None) -> "Block":
        if k <= self.kind:
            return self
        if self.kind == Block.SCALAR:
            if n is None:
                raise ValueError("promoting SCALAR requires a target n")
            diag = self.data.unsqueeze(-1).expand(*self.data.shape, n)
            d = Block(Block.DIAG, diag)
            return d if k == Block.DIAG else d.to(Block.DENSE)
        # DIAG -> DENSE: size is already in the data
        return Block(Block.DENSE, torch.diag_embed(self.data))

    def _scale(self, c: torch.Tensor) -> "Block":        # c·self, c broadcasts over [*B]
        if self.kind == Block.SCALAR:
            return Block(Block.SCALAR, c * self.data)
        if self.kind == Block.DIAG:
            return Block(Block.DIAG, c[..., None] * self.data)
        return Block(Block.DENSE, c[..., None, None] * self.data)

    # ---- additive: promote both to the join kind, then add ----
    def __add__(self, o: "Block") -> "Block":
        self._check_n(o)
        k = max(self.kind, o.kind)
        n = self._sized_n(o)
        return Block(k, self.to(k, n).data + o.to(k, n).data)

    def __neg__(self) -> "Block":
        return Block(self.kind, -self.data)

    def __sub__(self, o: "Block") -> "Block":
        return self + (-o)

    # ---- multiplicative: dispatch per kind, no promotion needed ----
    def __matmul__(self, o: "Block") -> "Block":
        self._check_n(o)
        if self.kind == Block.SCALAR:
            return o._scale(self.data)
        if o.kind == Block.SCALAR:
            return self._scale(o.data)
        if self.kind == Block.DIAG and o.kind == Block.DIAG:
            return Block(Block.DIAG, self.data * o.data)
        if self.kind == Block.DIAG:                # diag @ dense: scale rows
            return Block(Block.DENSE, self.data[..., :, None] * o.data)
        if o.kind == Block.DIAG:                   # dense @ diag: scale cols
            return Block(Block.DENSE, self.data * o.data[..., None, :])
        return Block(Block.DENSE, self.data @ o.data)

    # ---- inverse / solve: cheap for SCALAR & DIAG, solve (not inv) for DENSE ----
    def inv(self) -> "Block":
        if self.kind in (Block.SCALAR, Block.DIAG):
            return Block(self.kind, 1.0 / self.data)
        return Block(Block.DENSE, torch.linalg.inv(self.data))

    def solve(self, rhs: "Block") -> "Block":      # self^{-1} @ rhs
        self._check_n(rhs)
        if self.kind in (Block.SCALAR, Block.DIAG):
            return self.inv() @ rhs                # elementwise inverse, stays cheap
        r = rhs.to(Block.DENSE, self.n)
        r_data = r.data.to(self.data.dtype)
        return Block(Block.DENSE, torch.linalg.solve(self.data, r_data))

    def solve_many(self, *rhs: "Block") -> Tuple["Block", ...]:
        """Solve ``self^{-1} @ r`` for each ``r`` in ``rhs``, sharing one
        factorization of ``self`` when it is DENSE.

        All ``rhs`` are promoted to DENSE and concatenated along the column
        axis into a single ``torch.linalg.solve`` call, so ``self`` is
        LU-factorized once instead of once per ``rhs`` (LAPACK factorizes
        the left-hand side once and reuses it across every column of a
        multi-column right-hand side). No factorization is cached on the
        instance — it lives only for the duration of this call.

        SCALAR/DIAG ``self`` has no factorization to share (its inverse is
        already elementwise, O(n)), so each ``rhs`` is solved independently
        via ``self.inv() @ r``.

        Parameters
        ----------
        *rhs : Block
            Right-hand-side operators; each must share ``self``'s size
            (SCALAR sizes match anything). Their batch shapes must be
            mutually broadcastable (they are concatenated together before
            being broadcast against ``self``).

        Returns
        -------
        tuple of Block
            One solved ``Block`` per input ``rhs``, in the same order.
        """
        for r in rhs:
            self._check_n(r)
        if self.kind in (Block.SCALAR, Block.DIAG):
            inv = self.inv()
            return tuple(inv @ r for r in rhs)
        n = self.n
        dense_rhs = [r.to(Block.DENSE, n).data.to(self.data.dtype) for r in rhs]
        sizes     = [d.shape[-1] for d in dense_rhs]
        # rhs may carry different batch shapes (e.g. a batch-1 TVF field
        # alongside a fully-batched one); broadcast them to a common batch
        # before concatenating along columns (torch.cat needs exact shape
        # match on non-cat dims, unlike torch.linalg.solve's own A-vs-B
        # broadcasting, which is applied afterwards against `self.data`).
        batch = torch.broadcast_shapes(*(d.shape[:-1] for d in dense_rhs))
        dense_rhs = [d.expand(*batch, d.shape[-1]) for d in dense_rhs]
        combined  = torch.cat(dense_rhs, dim=-1)          # [..., n, sum(sizes)]
        solved    = torch.linalg.solve(self.data, combined)
        return tuple(Block(Block.DENSE, s) for s in torch.split(solved, sizes, dim=-1))

    # ---- constructors ----
    @classmethod
    def eye(cls, **kw) -> "Block":
        return cls(cls.SCALAR, torch.ones((), **kw))

    @classmethod
    def zeros(cls, **kw) -> "Block":
        return cls(cls.SCALAR, torch.zeros((), **kw))
    
    # --- neutral elements -------------------------------------------------
    def eye_like(self) -> "Block":
        return type(self).eye(device=self.data.device, dtype=self.data.dtype)

    def zeros_like(self) -> "Block":
        return type(self).zeros(device=self.data.device, dtype=self.data.dtype)

    def __repr__(self) -> str:
        name = {0: "SCALAR", 1: "DIAG", 2: "DENSE"}[self.kind]
        return f"Block({name}, shape={tuple(self.data.shape)})"


class Block2x2:
    """2x2 block operator [[a, b], [c, d]], each entry a Block OR a Block2x2."""

    def __init__(self, a: Entry, b: Entry, c: Entry, d: Entry):
        self.a, self.b, self.c, self.d = a, b, c, d

    # --- linear structure -------------------------------------------------
    def __add__(self, o: "Block2x2") -> "Block2x2":
        return Block2x2(self.a + o.a, self.b + o.b, self.c + o.c, self.d + o.d)

    def __sub__(self, o: "Block2x2") -> "Block2x2":
        return Block2x2(self.a - o.a, self.b - o.b, self.c - o.c, self.d - o.d)

    def __neg__(self) -> "Block2x2":
        return Block2x2(-self.a, -self.b, -self.c, -self.d)

    def __matmul__(self, o: "Block2x2") -> "Block2x2":
        return Block2x2(
            self.a @ o.a + self.b @ o.c,  self.a @ o.b + self.b @ o.d,
            self.c @ o.a + self.d @ o.c,  self.c @ o.b + self.d @ o.d,
        )

    # --- inverse / solve --------------------------------------------------
    def inv(self) -> "Block2x2":
        """Inverse via Schur complement of the d block.

        Prefer solve() for M^-1 @ rhs; use inv only when the materialized
        inverse is reused as an operator.
        """
        di  = self.d.inv()                 # reused below -> materializing is justified
        S   = self.a - self.b @ di @ self.c
        Si  = S.inv()
        dic = di @ self.c
        bdi = self.b @ di
        return Block2x2(
            Si,            -(Si @ bdi),
            -(dic @ Si),   di + dic @ Si @ bdi,
        )

    def _solve_schur(self, rhs: "Block2x2") -> "Block2x2":
        """Solve self @ X = rhs via Schur complement of the d block.

        Safe when d is SCALAR/DIAG (elementwise inverse) or when no inner
        sub-block is rank-deficient.  Called only for all-SCALAR inputs by the
        :meth:`solve` dispatcher; for DIAG/DENSE leaves use the representation-
        aware paths instead.
        """
        a, b, c, d = self.a, self.b, self.c, self.d
        dic = d.solve(c)                       # d^-1 c
        S   = a - b @ dic                      # Schur complement of d

        def col(r1, r2):
            t  = d.solve(r2)                   # d^-1 r2
            x1 = S.solve(r1 - b @ t)
            return x1, t - dic @ x1            # x2 = d^-1 (r2 - c x1)

        x1a, x2a = col(rhs.a, rhs.c)
        x1b, x2b = col(rhs.b, rhs.d)
        return Block2x2(x1a, x1b, x2a, x2b)

    def _solve_perharmonic(self, rhs: "Block2x2", n: int) -> "Block2x2":
        """Solve self @ X = rhs per harmonic — O(n) for all-DIAG/SCALAR leaves.

        Each harmonic index h decouples into an independent M×M system (M =
        leaf-matrix size = 2^nesting-depth).  Stacks all n systems into a single
        batched ``torch.linalg.solve`` call on a ``[..., n, M, M]`` tensor,
        then reassembles a Block2x2 tree with DIAG leaves.

        Robust at normal incidence: avoids the Schur-complement path whose inner
        ``d`` sub-block has exact-zero diagonal entries there.
        """
        L = _perharmonic_matrix(self, n)        # [..., n, M, M]
        R = _perharmonic_matrix(rhs,  n)        # [..., n, M, M]
        X = torch.linalg.solve(L, R)            # [..., n, M, M]
        return _perharmonic_to_block(X, rhs)

    def _solve_dense(self, rhs: "Block2x2", n: int) -> "Block2x2":
        """Solve self @ X = rhs via a single dense [..., M*n, M*n] solve.

        Densifies both operands to ``[..., M*n, M*n]`` (via :meth:`to_dense`),
        calls ``torch.linalg.solve`` once, then reconstructs a Block2x2 tree
        with DENSE leaves matching the structure of ``rhs``.

        Used when any leaf is DENSE (patterned/eig layer): avoids singular
        inner sub-blocks that break the Schur-complement path when the layer's
        V.d block is rank-deficient.
        """
        L = self.to_dense(n)
        R = rhs.to_dense(n)
        X = torch.linalg.solve(L, R.to(L.dtype))
        return _untile(X, rhs)

    def solve(self, rhs: "Block2x2") -> "Block2x2":
        """Solve self @ X = rhs, dispatching on leaf representation.

        Three strategies, selected automatically from the leaf Block kinds:

        - **all-SCALAR** (leaf size n unknown): Schur-complement
          (:meth:`_solve_schur`) — safe and exact, no rank-deficiency risk.
        - **all-DIAG/SCALAR**: per-harmonic batched solve
          (:meth:`_solve_perharmonic`) — O(n) instead of O(n³), and robust at
          normal incidence where DIAG ``d`` sub-blocks have exact-zero diagonals
          that would make the Schur path return NaN.
        - **any DENSE leaf**: full dense solve (:meth:`_solve_dense`) —
          densifies the tree to ``[..., M·n, M·n]``, solves once, and re-nests;
          avoids singular inner sub-blocks from rank-deficient patterned-layer
          ``V.d``.

        Preferred over ``inv() @ rhs``; no explicit inverse is formed.
        """
        n = _find_leaf_n(self)
        if n is None:
            n = _find_leaf_n(rhs)
        if n is None:                                    # all-SCALAR
            return self._solve_schur(rhs)
        if _all_diag_leaves(self) and _all_diag_leaves(rhs):
            return self._solve_perharmonic(rhs, n)       # DIAG/SCALAR
        return self._solve_dense(rhs, n)                 # any DENSE leaf

    # --- Redheffer star product ------------------------------------------
    def star(self, o: "Block2x2") -> "Block2x2":
        """Redheffer star product: self (left) ⋆ o (right).

        Composes the self|o stack. Not commutative — physical left/right
        order must be preserved. Associative, so fold direction is free.
        Inverts only (I - R1 R2), never an S-matrix.
        Convention: a=S11, b=S12, c=S21, d=S22 (reflection on the diagonal).

        Solves rather than inverts (repo convention, ``.solve(rhs)`` over
        ``.inv() @ rhs``): the textbook form needs ``self.b @ (I-P)^-1`` and
        ``o.c @ (I-Q)^-1``, right-multiplications by an inverse that
        ``Entry.solve`` (a left-solve) can't express directly. Instead solve
        for what those right-multiplications are actually used for —
        ``(I-P)^-1 @ o.a``, ``(I-P)^-1 @ o.b``, ``(I-Q)^-1 @ self.c``,
        ``(I-Q)^-1 @ (self.d @ o.b)`` — then left-multiply by ``self.b`` /
        ``o.c`` afterward; algebraically identical, but never materializes
        ``(I-P)^-1``/``(I-Q)^-1`` as an explicit operator, which is the more
        ill-conditioned quantity of the two near a Redheffer near-pole (see
        ``docs/matrixexp.md`` "Accuracy and conditioning").
        """
        P = o.a @ self.d                       # S11^B S22^A
        Q = self.d @ o.a                       # S22^A S11^B
        IP = P.eye_like() - P                  # I - S11^B S22^A
        IQ = Q.eye_like() - Q                  # I - S22^A S11^B
        Za = IP.solve(o.a)                     # (I - S11^B S22^A)^-1 S11^B
        Zb = IP.solve(o.b)                     # (I - S11^B S22^A)^-1 S12^B
        Wc = IQ.solve(self.c)                  # (I - S22^A S11^B)^-1 S21^A
        Wdb = IQ.solve(self.d @ o.b)           # (I - S22^A S11^B)^-1 S22^A S12^B
        return Block2x2(
            self.a + self.b @ Za @ self.c,  self.b @ Zb,
            o.c @ Wc,                        o.d + o.c @ Wdb,
        )

    # --- neutral elements -------------------------------------------------
    def eye_like(self) -> "Block2x2":
        """Multiplicative identity [[I, 0], [0, I]] of matching type/shape."""
        return Block2x2(self.a.eye_like(), self.b.zeros_like(),
                        self.c.zeros_like(), self.d.eye_like())

    def zeros_like(self) -> "Block2x2":
        """Additive zero of matching type/shape."""
        return Block2x2(self.a.zeros_like(), self.b.zeros_like(),
                        self.c.zeros_like(), self.d.zeros_like())

    def to_dense(self, n: int | None = None) -> torch.Tensor:
        """Flatten to a dense ``(..., 2N, 2N)`` tensor where N is the block size.

        Each Block entry is promoted to DENSE; nested Block2x2 entries are
        flattened recursively.  If ``n`` is not given it is inferred from the
        first non-SCALAR leaf block.  Pass ``n`` explicitly when any leaf
        block is SCALAR.

        Parameters
        ----------
        n : int or None
            Harmonic size of each leaf Block.  Required when any entry is a
            SCALAR Block (which carries no size information).

        Returns
        -------
        torch.Tensor
            Dense matrix of shape ``(..., 2N, 2N)`` (or ``(..., 2^k·N, 2^k·N)``
            for k levels of nesting).
        """
        n_eff = n if n is not None else _find_leaf_n(self)

        def _entry(e):
            if hasattr(e, 'a'):                   # nested Block2x2
                return e.to_dense(n_eff)
            if n_eff is None:
                raise ValueError(
                    "n must be provided when entries contain SCALAR Blocks"
                )
            return e.to(Block.DENSE, n_eff).data
        A, B = _entry(self.a), _entry(self.b)
        C, D = _entry(self.c), _entry(self.d)
        # Align batch dims: SCALAR entries produce 2-D tensors while batched
        # DIAG/DENSE entries produce 3-D (or higher).  Unsqueeze to match.
        max_ndim = max(A.ndim, B.ndim, C.ndim, D.ndim)
        def _pad(t):
            while t.ndim < max_ndim:
                t = t.unsqueeze(0)
            return t
        A, B, C, D = _pad(A), _pad(B), _pad(C), _pad(D)
        target = torch.broadcast_shapes(A.shape[:-2], B.shape[:-2], C.shape[:-2], D.shape[:-2])
        def _expand(t):
            return t.expand(*target, *t.shape[-2:])
        A, B, C, D = _expand(A), _expand(B), _expand(C), _expand(D)
        return torch.cat([torch.cat([A, B], dim=-1),
                          torch.cat([C, D], dim=-1)], dim=-2)

    @classmethod
    def from_dense(cls, dense: torch.Tensor, template: "Block2x2") -> "Block2x2":
        """Inverse of :meth:`to_dense`: reconstruct a Block2x2 tree from a
        dense ``(..., 2N, 2N)`` tensor (or ``(..., 2^k*N, 2^k*N)`` for k
        nesting levels), splitting into quadrants matching *template*'s
        nesting shape. Leaf entries become ``Block(DENSE, ...)``.

        Used by callers that compute a dense operator directly (e.g. a
        matrix exponential) and need to re-embed it into the Block2x2
        algebra to compose with structured operators via ``@``/``solve``/
        ``star``.
        """
        return _untile(dense, template)

    # --- misc -------------------------------------------------------------
    @property
    def shape(self) -> Tuple[object, object, object, object]:
        # A nested entry's `shape` is itself a 4-tuple, not a torch.Size.
        return self.a.shape, self.b.shape, self.c.shape, self.d.shape

    @classmethod
    def identity(cls) -> "Block2x2":
        """Matmul identity [[I, 0], [0, I]] at leaf level (Block entries)."""
        return cls(Block.eye(), Block.zeros(), Block.zeros(), Block.eye())  # noqa: F821

    @classmethod
    def star_identity(cls) -> "Block2x2":
        """Star-product identity [[0, I], [I, 0]] at leaf level (Block entries)."""
        return cls(Block.zeros(), Block.eye(), Block.eye(), Block.zeros())  # noqa: F821


# ---------------------------------------------------------------------------
# Module-level helpers used by Block2x2.solve dispatch
# (defined after both classes so isinstance checks are valid)
# ---------------------------------------------------------------------------

def _find_leaf_n(entry) -> int | None:
    """Return the size of the first non-SCALAR leaf Block in *entry*, or None.

    Recurses into nested Block2x2 trees.  Returns ``None`` when every leaf is
    a SCALAR Block (size-agnostic).
    """
    if isinstance(entry, Block):
        return entry.n          # None for SCALAR, int for DIAG/DENSE
    for sub in (entry.a, entry.b, entry.c, entry.d):
        n = _find_leaf_n(sub)
        if n is not None:
            return n
    return None


def _all_diag_leaves(entry) -> bool:
    """True iff every leaf Block in *entry* is SCALAR or DIAG (no DENSE).

    Works for both leaf ``Block`` objects and nested ``Block2x2`` trees.
    """
    if isinstance(entry, Block):
        return entry.kind != Block.DENSE
    return all(_all_diag_leaves(e) for e in (entry.a, entry.b, entry.c, entry.d))


def _perharmonic_matrix(entry, n: int) -> torch.Tensor:
    """Convert a Block/Block2x2 with all-DIAG/SCALAR leaves to ``[..., n, M, M]``.

    For a leaf ``Block``, promotes to DIAG and returns ``[..., n, 1, 1]``.
    For a ``Block2x2``, recursively assembles the four sub-matrices into a
    ``[..., n, 2M, 2M]`` tensor (block-row/column order matching the 2×2
    structure), broadcasting batch dimensions across all four sub-entries.

    The resulting tensor ``T`` satisfies ``T[..., h, :, :]`` = the M×M matrix
    governing harmonic index ``h``, which is independent of all other harmonics
    when every leaf is DIAG (no cross-harmonic coupling).
    """
    if isinstance(entry, Block):
        return entry.to(Block.DIAG, n).data.unsqueeze(-1).unsqueeze(-1)   # [..., n, 1, 1]
    sa = _perharmonic_matrix(entry.a, n)   # [..., n, m, m]
    sb = _perharmonic_matrix(entry.b, n)
    sc = _perharmonic_matrix(entry.c, n)
    sd = _perharmonic_matrix(entry.d, n)
    # Broadcast batch dimensions across all four sub-tensors before cat.
    batch = torch.broadcast_shapes(
        sa.shape[:-3], sb.shape[:-3], sc.shape[:-3], sd.shape[:-3]
    )
    sa = sa.expand(*batch, *sa.shape[-3:])
    sb = sb.expand(*batch, *sb.shape[-3:])
    sc = sc.expand(*batch, *sc.shape[-3:])
    sd = sd.expand(*batch, *sd.shape[-3:])
    return torch.cat(
        [torch.cat([sa, sb], dim=-1),
         torch.cat([sc, sd], dim=-1)], dim=-2,
    )                                                                       # [..., n, 2m, 2m]


def _perharmonic_to_block(x: torch.Tensor, template):
    """Reconstruct a Block/Block2x2 from a ``[..., n, M, M]`` solved tensor.

    Uses *template* (the original rhs ``Block2x2``) to determine the tree
    shape.  Leaf entries are returned as ``Block(DIAG, ...)``.  Slice indices
    mirror :func:`_perharmonic_matrix`'s assembly order.
    """
    if isinstance(template, Block):
        return Block(Block.DIAG, x.squeeze(-1).squeeze(-1))   # [..., n, 1, 1] → [..., n]
    M  = x.shape[-1]
    M2 = M // 2
    return Block2x2(
        _perharmonic_to_block(x[..., :M2, :M2], template.a),
        _perharmonic_to_block(x[..., :M2, M2:], template.b),
        _perharmonic_to_block(x[..., M2:, :M2], template.c),
        _perharmonic_to_block(x[..., M2:, M2:], template.d),
    )


def _untile(dense: torch.Tensor, template):
    """Reconstruct a Block/Block2x2 from a dense ``[..., M*n, M*n]`` tensor.

    Uses *template* (the original rhs ``Block2x2``) to determine the tree
    shape.  Leaf entries become ``Block(DENSE, ...)``.  Splits the dense
    matrix recursively into quadrants matching the 2×2 block structure.
    """
    if isinstance(template, Block):
        return Block(Block.DENSE, dense)
    H2 = dense.shape[-2] // 2
    W2 = dense.shape[-1] // 2
    return Block2x2(
        _untile(dense[..., :H2, :W2], template.a),
        _untile(dense[..., :H2, W2:], template.b),
        _untile(dense[..., H2:, :W2], template.c),
        _untile(dense[..., H2:, W2:], template.d),
    )
    