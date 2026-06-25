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

    def solve(self, rhs: "Block2x2") -> "Block2x2":
        """Solve  self @ X = rhs  via Schur complement of the d block.

        Leaf solve is lu_factor/lu_solve (DENSE) or elementwise (SCALAR/DIAG):
        no explicit inverse is formed. Preferred over inv() @ rhs.
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

    # --- Redheffer star product ------------------------------------------
    def star(self, o: "Block2x2") -> "Block2x2":
        """Redheffer star product: self (left) ⋆ o (right).

        Composes the self|o stack. Not commutative — physical left/right
        order must be preserved. Associative, so fold direction is free.
        Inverts only (I - R1 R2), never an S-matrix.
        Convention: a=S11, b=S12, c=S21, d=S22 (reflection on the diagonal).
        """
        P = o.a @ self.d                       # S11^B S22^A
        Q = self.d @ o.a                       # S22^A S11^B
        D = self.b @ (P.eye_like() - P).inv()  # S12^A (I - S11^B S22^A)^-1
        F = o.c   @ (Q.eye_like() - Q).inv()   # S21^B (I - S22^A S11^B)^-1
        return Block2x2(
            self.a + D @ o.a @ self.c,  D @ o.b,
            F @ self.c,                 o.d + F @ self.d @ o.b,
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
        def _find_n(m):
            for e in (m.a, m.b, m.c, m.d):
                found = _find_n(e) if hasattr(e, 'a') else e.n
                if found is not None:
                    return found
            return None

        n_eff = n if n is not None else _find_n(self)

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
        return torch.cat([torch.cat([A, B], dim=-1),
                          torch.cat([C, D], dim=-1)], dim=-2)

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
    