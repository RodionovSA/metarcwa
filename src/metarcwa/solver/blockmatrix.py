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
        return Block(Block.DENSE, torch.linalg.solve(self.data, r.data))

    # ---- constructors ----
    @classmethod
    def eye(cls, **kw) -> "Block":
        return cls(cls.SCALAR, torch.ones((), **kw))

    @classmethod
    def zeros(cls, **kw) -> "Block":
        return cls(cls.SCALAR, torch.zeros((), **kw))

    def __repr__(self) -> str:
        name = {0: "SCALAR", 1: "DIAG", 2: "DENSE"}[self.kind]
        return f"Block({name}, shape={tuple(self.data.shape)})"
    
class Block2x2:
    """2x2 block operator [[a, b], [c, d]], each entry a Block.

    Structure-blind: every operation is written in terms of Block ops, so it
    runs elementwise when the entries are SCALAR/DIAG and falls back to dense
    only where a DENSE entry forces it.
    """
    def __init__(self, a: Block, b: Block, c: Block, d: Block):
        self.a, self.b, self.c, self.d = a, b, c, d

    def __add__(self, o: "Block2x2") -> "Block2x2":
        return Block2x2(self.a + o.a, self.b + o.b, self.c + o.c, self.d + o.d)

    def __sub__(self, o: "Block2x2") -> "Block2x2":
        return Block2x2(self.a - o.a, self.b - o.b, self.c - o.c, self.d - o.d)

    def __matmul__(self, o: "Block2x2") -> "Block2x2":
        return Block2x2(
            self.a @ o.a + self.b @ o.c,  self.a @ o.b + self.b @ o.d,
            self.c @ o.a + self.d @ o.c,  self.c @ o.b + self.d @ o.d,
        )

    def inv(self) -> "Block2x2":
        """Inverse via Schur complement of the d block."""
        di  = self.d.inv()                 # reused below -> materializing is justified
        S   = self.a - self.b @ di @ self.c
        Si  = S.inv()
        dic = di @ self.c
        bdi = self.b @ di
        return Block2x2(
            Si,            -(Si @ bdi),
            -(dic @ Si),   di + dic @ Si @ bdi,
        )

    @classmethod
    def identity(cls) -> "Block2x2":
        return cls(Block.eye(), Block.zeros(), Block.zeros(), Block.eye())