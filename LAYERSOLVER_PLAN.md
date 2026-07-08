# Plan — LayerSolver `prepare()` / `LayerOperator` split

**Goal:** stop re-running the expensive per-layer precomputation (TVF Newton solve,
convolution matrices, eigendecomposition) on every `solve()` call, without hidden
caches. Addresses ANALYSIS.md items **D1/C2** (TVF precompute claim is false),
**C1** (TVF Hessian batched over N_wvl redundantly), and **D5** (`epsilon_conv`
rebuilt every call) — and makes the documented claim
"`Solver.__init__` is expensive, `solve()` is cheap" actually true.

**Design principle:** no memoization keyed on `id(layer)` or geometry hashes.
Instead, reify the expensive phase as an explicit object (`LayerOperator`) whose
lifetime the *caller* owns. Autograd graphs are then never silently retained across
optimizer steps — in inverse design you rebuild operators each step (which you must,
since the pattern changed); in spectral/thickness sweeps you reuse them explicitly.

---

## Overview of the API after the change

```
LayerSolver (unchanged construction: config, wvl, kx, ky, m_flat, n_flat, tvf)
  ├─ prepare(element)  -> LayerOperator     # EXPENSIVE: TVF field, eps_conv, eig
  ├─ smatrix(op, left) -> Block2x2          # cheap-ish: boundary solves + star
  └─ solve(element, left) -> Block2x2       # thin wrapper = smatrix(prepare(...))
                                            #   kept for backward compatibility

Solver
  ├─ __init__: ... + self._ops = [prepare(e) for e in stack elements]
  └─ solve(): star-compose smatrix(op) over self._ops   # no eig, no TVF, no FFT
```

`LayerOperator` is a frozen dataclass holding the modal solution
(`lam`, `W`, `V`, `thickness`). It is pure data — the S-matrix assembly stays on
`LayerSolver`, which owns the shared context (`W0`, `V0`, `wvl`).

Key property: **thickness enters only at `smatrix()` time** (via `S_prop`), so a
thickness-only optimization or sweep reuses all eigendecompositions for free.

---

## Step 1 — C1: single-slice TVF, field-based `compute_A`

*Independent of the API change; do first, it's the biggest pure-compute win.*

The TVF direction field is wavelength-independent by construction: `TVF.compute()`
works on `real(field).detach()`, globally normalizes the gradient
(`normalize_max_global`), and the A-blocks are quadratic in the field
(`|T|²`, `T*·T`), so both the overall scale *and sign* of the permittivity contrast
`(ε_solid − ε_void)(λ)` cancel. Computing the TVF from the **pattern mask** itself
is therefore equivalent to computing it from any wavelength slice of `eps_grid` —
and more robust (no degeneracy when the real-part contrast crosses zero at some λ).

Today `tvf.compute(epsilon_grid)` receives `[N_wvl, Ny, Nx]` and the Newton solve
builds a Hessian `[N_wvl, flat, flat]` — N_wvl identical copies. After this step it
receives `[1, Ny, Nx]`.

### 1a. `isotropic.py` — `compute_A` takes fields, not a TVF instance

```python
# BEFORE
def compute_A(epsilon_grid, m_flat, n_flat, tvf) -> tuple[Block, Block, Block, Block]:
    Tx, Ty = tvf.compute(epsilon_grid)
    ...

# AFTER
def compute_A(Tx: torch.Tensor, Ty: torch.Tensor,
              m_flat: torch.Tensor, n_flat: torch.Tensor
              ) -> Tuple[Block, Block, Block, Block]:
    """
    Tx, Ty : tangent vector field components, shape [B, Ny, Nx].
             B may be 1 (wavelength-independent field, broadcast downstream)
             or match the batch of epsilon_conv.
    """
    axx = Ty.abs() ** 2
    axy = Tx.conj() * Ty
    ayx = Tx * Ty.conj()
    ayy = Tx.abs() ** 2
    # ... convolution_matrix calls unchanged
```

### 1b. `isotropic.py` — `compute_isotropic` takes precomputed fields

```python
# BEFORE
def compute_isotropic(epsilon_grid, m_flat, n_flat, kx, ky,
                      tvf: TVF | None = None) -> Tuple[Block2x2, Block2x2]:

# AFTER
def compute_isotropic(epsilon_grid, m_flat, n_flat, kx, ky,
                      tvf_fields: tuple[torch.Tensor, torch.Tensor] | None = None
                      ) -> Tuple[Block2x2, Block2x2]:
    ...
    if tvf_fields is None:
        Q = compute_Q0(Kx, Ky, epsilon_conv)
    else:
        Tx, Ty = tvf_fields
        epsilon_inv_conv = Block(Block.DENSE,
                                 convolution_matrix(1.0 / epsilon_grid, m_flat, n_flat))
        Axx, Axy, Ayx, Ayy = compute_A(Tx, Ty, m_flat, n_flat)
        Q = compute_Q(Kx, Ky, epsilon_conv, epsilon_inv_conv, Axx, Axy, Ayx, Ayy)
    return P, Q
```

The `from metarcwa.solver.tvf import TVF` import in `isotropic.py` becomes unused —
remove it (nice side effect: `isotropic.py` no longer needs the TVF package at all).

### 1c. `layersolver/base.py` — `_patterned` invokes TVF on the pattern, batch 1

```python
# inside _patterned, replacing the compute_isotropic call:
if self.tvf is not None:
    # TVF is geometry-only (detached, sign/scale-invariant in the A-blocks):
    # compute once from the pattern mask, [1, Ny, Nx], not per wavelength.
    tvf_fields = self.tvf.compute(pattern[None])
else:
    tvf_fields = None

P, Q = compute_isotropic(
    eps_grid, self.m_flat, self.n_flat,
    self.kx, self.ky, tvf_fields,
)
```

### 1d. Broadcasting check (must verify, likely already works)

A-blocks are now `[1, Nh, Nh]` while `epsilon_conv` is `[N_wvl, Nh, Nh]`.
`compute_Qfact` does `epsilon_conv @ Ayx` and `epsilon_inv_conv.solve(Ayx)`:

- `torch.matmul` broadcasts batch dims → fine.
- `torch.linalg.solve(A, B)` broadcasts batch dims of A and B → fine,
  **provided `Block.__matmul__` / `Block.solve` don't assert equal shapes.**
  Check `blockmatrix.py`; if there's a shape assertion, relax it to
  `torch.broadcast_shapes`.

Guard with a test (Step 4, test T5): old batched path vs new single-slice path must
produce the same S-matrix to `float64` tolerance.

---

## Step 2 — `LayerOperator` + `prepare()`/`smatrix()` on `LayerSolver`

All in `src/metarcwa/solver/layersolver/base.py`.

### 2a. The operator dataclass

```python
from dataclasses import dataclass

@dataclass(frozen=True)
class LayerOperator:
    """Precomputed modal solution of one stack element.

    Produced by :meth:`LayerSolver.prepare` (the expensive step: TVF field,
    convolution matrices, eigendecomposition). Consumed by
    :meth:`LayerSolver.smatrix` (cheap: boundary matching + propagation).

    The caller owns the lifetime: rebuild when the geometry or source changes
    (e.g. every inverse-design step); reuse across repeated solves at fixed
    geometry. ``thickness`` is read at smatrix time, so thickness-only changes
    (in-place parameter updates) do NOT require re-preparing.

    Attributes
    ----------
    lam : torch.Tensor
        Modal exponents lam = 1j·kz, shape ``[..., 2Nh]``.
    W : Block2x2
        E-mode matrix.
    V : Block2x2
        H-mode matrix.
    thickness : torch.Tensor or None
        Layer thickness; ``None`` marks a semi-infinite medium (boundary only).
    """
    lam: torch.Tensor
    W: Block2x2
    V: Block2x2
    thickness: torch.Tensor | None = None
```

(`frozen=True` matches the `HomogeneousLayer`/`PatternedLayer` spec style.)

### 2b. `prepare()` — the dispatcher, returning operators

The three private methods change their return type from `Block2x2` to
`LayerOperator` and **stop calling `S_layer`/`S_boundary`**:

```python
def prepare(self, element: HomogeneousLayer | PatternedLayer | MediumSpec
            ) -> LayerOperator:
    """Expensive per-element precomputation → reusable LayerOperator."""
    if isinstance(element, HomogeneousLayer):
        return self._homogeneous(element)
    elif isinstance(element, PatternedLayer):
        return self._patterned(element)
    elif isinstance(element, MediumSpec):
        return self._medium(element)
    raise TypeError(
        f"element must be HomogeneousLayer, PatternedLayer, or "
        f"MediumSpec, but got {type(element)}"
    )

def _homogeneous(self, layer: HomogeneousLayer) -> LayerOperator:
    medium = layer.medium
    if isinstance(medium, IsotropicMediumSpec):
        lam, V = homogeneous_modes(medium.eps, self.kx, self.ky)
        W = V.eye_like()
    else:
        raise NotImplementedError(...)
    return LayerOperator(lam, W, V, layer.thickness)

def _patterned(self, layer: PatternedLayer) -> LayerOperator:
    # ... eps_grid + tvf_fields + compute_isotropic + eigsolver as today ...
    return LayerOperator(lam, W, V, layer.thickness)

def _medium(self, medium: MediumSpec) -> LayerOperator:
    if isinstance(medium, IsotropicMediumSpec):
        lam, V = homogeneous_modes(medium.eps, self.kx, self.ky)
        W = V.eye_like()
    else:
        raise NotImplementedError(...)
    return LayerOperator(lam, W, V, thickness=None)
```

Note `_medium` no longer takes `left` — side selection is an assembly concern and
moves to `smatrix()`. (Storing `lam` for media costs nothing and will be needed
anyway for diffraction-efficiency postprocessing in `results/`.)

### 2c. `smatrix()` — cheap assembly from an operator

```python
def smatrix(self, op: LayerOperator, left: bool = True) -> Block2x2:
    """Assemble the S-matrix from a prepared operator.

    left : for semi-infinite media only — True places the medium on the
    left (input) side of the interface, False on the right (output) side.
    """
    if op.thickness is None:
        if left:
            return S_boundary(op.W, op.V, self.W0, self.V0)
        return S_boundary(self.W0, self.V0, op.W, op.V)
    return S_layer(self.W0, self.V0, op.W, op.V, op.lam, op.thickness, self.wvl)
```

### 2d. `solve()` — kept as a thin wrapper (backward compatible)

```python
def solve(self, element, left: bool = True) -> Block2x2:
    return self.smatrix(self.prepare(element), left)
```

Signature and behavior identical to today → the existing test suite and notebooks
keep working unmodified.

---

## Step 3 — `Solver` prepares operators in `__init__`

`src/metarcwa/solver/base.py`:

```python
def __init__(self, model: Model, config: Config) -> None:
    # ... unchanged up to LayerSolver construction ...
    self.layersolver = LayerSolver(
        config, self.model_spec.wavelength, kx, ky, m_flat, n_flat, tvf
    )

    # Expensive phase: modal solution of every stack element, done once.
    self._ops = [
        self.layersolver.prepare(self.model_spec.incidence),
        *(self.layersolver.prepare(layer) for layer in self.model_spec.layers),
        self.layersolver.prepare(self.model_spec.transmission),
    ]

def solve(self) -> Block2x2:
    ls = self.layersolver
    S = ls.smatrix(self._ops[0], left=True)
    for op in self._ops[1:-1]:
        S = S.star(ls.smatrix(op))
    return S.star(ls.smatrix(self._ops[-1], left=False))
```

This is what makes the CLAUDE.md / docstring claim true. Two consequences to state
explicitly in the `Solver` docstring:

1. **Inverse design:** rebuild the `Solver` every optimization step. This was
   already required (`model.spec()` resolves the pattern in `__init__`), so
   nothing changes for that workflow — but now the docs say it.
2. **Thickness gradients/updates still flow:** operators hold a *reference* to the
   thickness tensor; `S_prop` reads it inside `solve()`, so autograd through
   thickness works and an in-place `nn.Parameter` update is picked up without
   re-preparing. Pattern/ε changes DO require a rebuild.

---

## Step 4 — Tests

Extend `tests/solver/test_layersolver.py` (new class `TestLayerSolverPrepare`)
plus one equivalence test for Step 1. Use the existing `_make_solver`/`_hom`/`_pat`
helpers.

| # | Test | Asserts |
|---|------|---------|
| T1 | `test_prepare_returns_operator` | `prepare()` on hom/pat/medium returns `LayerOperator`; `thickness is None` iff medium |
| T2 | `test_smatrix_prepare_equals_solve` | `smatrix(prepare(x), left)` bitwise-equals `solve(x, left)` for all three element types and both `left` values |
| T3 | `test_operator_reuse_is_deterministic` | calling `smatrix(op)` twice on one prepared op gives identical S both times |
| T4 | `test_thickness_change_without_reprepare` | prepare a layer, then `dataclasses.replace(op, thickness=d2)`; resulting S equals a fresh `solve()` of a layer with thickness `d2` — documents the thickness-sweep pattern |
| T5 | **(guards Step 1)** `test_tvf_single_slice_matches_batched` | A-blocks from `tvf.compute(pattern[None])` vs `tvf.compute(eps_grid)` with a dispersive `eps_solid` (including one λ where the real contrast is negative) → resulting patterned-layer S-matrices `allclose` in float64 |
| T6 | `test_solver_solve_unchanged` | end-to-end: full-stack S from the refactored `Solver` matches the pre-refactor value on a small fixture (record expected once, or compare `Solver.solve()` against manual `layersolver.solve()` composition) |

All ~30 existing tests must stay green with zero edits — that's the backward-compat
check for Step 2d.

---

## Step 5 (optional, after 1–4 land)

- **D2 — TVF band-mask cache:** `low_pass_mask` + `in_band_idx` recomputed per
  `TVF.compute()`. With Step 1 the TVF runs once per `prepare()` instead of every
  `solve()`, so this is now minor. If still wanted: lazy cache keyed on
  `(D0, D1, device)` in `TVF`, since grid size isn't known at `__init__`.
- **Cache `S_in` on the operator:** `S_layer` re-does its boundary solve
  (O(Nh³) dense solve in `S_boundary`) every `smatrix()` call even though only
  `S_prop` depends on thickness. If thickness sweeps become a hot path, store the
  thickness-independent `S_in` in `LayerOperator` and assemble
  `S_in ⋆ S_prop(d) ⋆ S_out` in `smatrix()`. Measure first.
- **D3/D4 — `HarmonicContext` split** (reusable lattice/source context across
  Solver rebuilds): deferred; independent of this plan and easier once operators
  exist.

## Doc updates (same PR)

- `Solver` and `LayerSolver` docstrings: describe the two-phase API and operator
  lifetime rules (rebuild on geometry change; thickness exempt).
- `CLAUDE.md` architecture table: add `LayerOperator`; the data-flow diagram gains
  `prepare() → LayerOperator → smatrix()`.
- `ANALYSIS.md`: mark D1/C2, C1, D5 as fixed; note D2 downgraded.

## Suggested commit order

1. Step 1 (+ T5) — self-contained, immediately mergeable.
2. Steps 2 + 4 (T1–T4) — LayerSolver API, `solve()` unchanged externally.
3. Step 3 (+ T6) + doc updates — Solver precompute.

## Open decisions (defaults chosen, flag if you disagree)

1. **TVF input = pattern mask** (`pattern[None]`), not `eps_grid[:1]`. Equivalent by
   the sign/scale-invariance argument above, and robust at zero-contrast
   wavelengths. T5 verifies equivalence.
2. **`LayerOperator` lives in `layersolver/base.py`** next to its producer, rather
   than a new module. Move later if `results/` needs to import it without the solver.
3. **`solve()` wrapper stays.** Costs three lines and keeps every notebook/test
   working; deprecate later if it causes confusion.
