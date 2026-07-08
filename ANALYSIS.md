# metarcwa — Architecture & Efficiency Analysis

**Date:** 2026-06-27 · **Revised:** 2026-07-08
**Branch:** redesign
**Method:** Read-only source review (3 parallel analysis agents + direct source verification).
No profiling numbers were collected — agents reviewed source only; no clean GPU benchmark was run.
If real `torch.cuda.max_memory_allocated` / profiler traces are wanted, a follow-up pass can do that.

**Revision note (2026-07-08):** All Tier 1 correctness bugs are fixed, and the
Tier 2 precompute boundary was restructured via the `prepare()`/`LayerOperator` split
(`LAYERSOLVER_PLAN.md`, Steps 1–3). Statuses below reflect the verified state of the source,
priorities were re-ranked accordingly, and several fix recommendations were replaced with
better options that account for the new architecture. D3/D4 was downgraded (its cost argument
mostly evaporated and its original fix was based on an incorrect invariance analysis — see the
rewritten entry).

**Status legend:** ✅ Fixed · 🔶 Partially fixed · ⬜ Open · 💤 Deferred by design

Findings are ordered **expensive-to-change and correctness-critical first** within each tier.
Each entry: `Category · Severity · file:line · what + why · recommendation + tradeoff · effort`.

---

## Tier 1 — Correctness bugs (all confirmed bugs fixed; test/robustness tail remains)

The three HIGH bugs (E1–E3) that made every patterned layer produce wrong mode propagation
are fixed and covered by the operator-level test
`tests/solver/test_isotropic.py::test_uniform_eps_Q_matches_homogeneous_Q`.
What remains open in this tier: the gradcheck (E4), the grazing-mode regularization half of E6,
and the end-to-end S-matrix regression tests (E7).

---

### E1 · HIGH · `solver/layersolver/eigsolver.py` · Extra `1j` factor on modal exponents — ✅ Fixed

**What (historical):** The eigenproblem is `Ω²·w = λ²·w` where `λ² = P·Q` eigenvalues.
The code took `kz = sqrt(lam_sq)` and then set `lam = 1j * kz` — applying `1j` to something
already equal to `λ`. Net effect: patterned layers decayed exponentially where they should
have propagated with unit modulus; the two solvers disagreed for an identical medium; every
inverse-design gradient through patterned layers was wrong.

**Status:** ✅ Fixed. `eigsolver.py:97–102` now takes `lam = sqrt(lam_sq)` directly with branch
selection (propagating: `Im(lam) > 0`; evanescent: `Re(lam) < 0`), matching the
`homogeneous_modes` convention documented in the eigsolver docstring.
**Sergei:** Fixed.

---

### E2 · HIGH · `solver/layersolver/isotropic.py` · TVF A-blocks double-FFT'd — ✅ Fixed

**What (historical):** `compute_A` FFT'd the spatial TVF fields and passed `|Ty_fft|²` to
`convolution_matrix`, which FFTs again — producing the autocorrelation of `Ty` instead of the
Fourier coefficients of `|Ty(x,y)|²` required by `docs/factorization.md` Eq. (14). The identity
`[[Axx]] + [[Ayy]] = I` was violated.

**Status:** ✅ Fixed. `compute_A` forms the outer-product components in real space
(`axx = |Ty|²`, `axy = Tx*·Ty`, …) and lets `convolution_matrix` do the single transform.
Signature later changed to take precomputed `(Tx, Ty)` fields (see C1).
**Sergei:** Fixed.

---

### E3 · HIGH · `solver/layersolver/isotropic.py` · Inverse rule: `Δ` used `[[ε]]⁻¹` instead of `[[1/ε]]⁻¹` — ✅ Fixed

**What (historical):** Li's correction is `Δ = [[ε]] − [[ε⁻¹]]⁻¹`, where `[[ε⁻¹]]` is the
Toeplitz matrix of the Fourier coefficients of `1/ε(r)`. The code used the inverse of the
*direct* convolution matrix — `[[ε]]⁻¹ ≠ [[1/ε]]⁻¹` in general, which is the entire point of
Li's rule. For uniform ε=4 the code added a large spurious `Δ = 3.75I` where `Δ = 0` is exact.

**Status:** ✅ Fixed. `compute_isotropic` builds `epsilon_inv_conv` from `1/epsilon_grid` and
`compute_Qfact` applies `epsilon_inv_conv.solve(A)` (i.e. `−(ε − [[1/ε]]⁻¹)·A` per block).
**Sergei:** Fixed.

---

### E4 · MED · `solver/layersolver/eigsolver.py:214–241` · `Eig.backward` uses `conj(F)` — ✅ Fixed (verified + docstring aligned)

**Status:** ✅ Resolved 2026-07-08. `tests/solver/test_eigsolver.py` gradchecks the eigenvalue
branch (`test_eigval_gradcheck`), gradchecks a gauge-invariant matrix function built from
`(lam, W)` that exercises the eigenvector branch (`test_matrix_func_gradcheck`), and compares
the resulting gradient directly against `torch.linalg.eig`'s own backward on the same
well-separated-spectrum input (`test_backward_matches_torch_linalg_eig`) — all three pass.
The double-`conj(F)` application is therefore correct as-is (no code change); the class and
`backward` docstrings in `eigsolver.py` were rewritten to state the two-step formula the code
actually computes (`F = conj(s)/(|s|²+ε)`, then `conj(F)` applied ⇒ net factor
`s/(|s|²+ε)`) instead of the previous inaccurate single-step description, with no uncertainty
hedging per Sergei's instruction.

**What (historical):** Standard non-symmetric-eig adjoint uses `F_ij = 1/(λ_j − λ_i)`. The docstring said the
Lorentzian replaces this with `conj(λ_j−λ_i)/(|λ_j−λ_i|²+ε)`. Line 233 applies `conj(F)` a
*second* time to the already-conjugated `F`, effectively using `(λ_j−λ_i)/(|.|²+ε)` — may be
intentional for complex eigenvalues (Boeddeker et al. 2020) but is unverified. The implementation
also zeros `diag(F)` but adds no eigenvector gauge-correction term.

**Fix (unchanged, now concrete):** Run gradcheck in complex128 on small well-separated matrices,
with both output grads exercised:

```python
x = torch.randn(4, 4, dtype=torch.complex128, requires_grad=True)
x = x + 4 * torch.diag(torch.arange(4, dtype=torch.complex128))  # separate the spectrum
torch.autograd.gradcheck(lambda m: Eig.apply(m), (x,), check_forward_ad=False)
```

Compare against `torch.linalg.eig`'s own backward on the same input as a second oracle
(temporarily set `broadening_parameter = None`-equivalent small ε). If gradcheck passes, only
align the docstring with what the code computes; if it fails, fix `backward`.
Per instruction below: no hedging docstrings either way — state what the code does.

**Tradeoff:** If correct, doc-only change. If wrong, fixing may change convergence of
gradient-based optimization.
**Effort:** ~1 h. Folds naturally into E7's test additions.
**Sergei:** Not sure here. As stated in the file's top, I use implementation from Torcwa.
So, maybe better to modify docstring. If you fix it, do not add docstrings of uncertanity. Just modify.

---

### E5 · MED · `eigsolver.py` vs `homogeneous.py` · Inconsistent propagating/evanescent classifier — ✅ Fixed (logic) / see A3 (duplication)

**Status:** ✅ Fixed with E1. Both solvers now implement the same rule
(propagating: `Im(lam) > 0`; evanescent: `Re(lam) < 0`, with the `sign==0` fixup).
The rule is however still *duplicated*, not shared — extracting the common
`_branch_select` / `lam_inv` helper is tracked in A3.
**Sergei:** Fixed with E1.

---

### E6 · MED · `homogeneous.py:207–213` / `eigsolver.py:105–112` · Grazing λ≈0 → NaN — ✅ Fixed

**Status:** ✅ Fixed together with A3. The Lorentzian-regularized inverse now lives in
`solver/layersolver/_modes.py::_lam_inv_block` and is used by both `homogeneous_modes` and
`eigsolver`, so `V` stays finite at exact grazing incidence (`test_grazing_V_is_finite`).
The copy-paste warning message (`eigsolver.py:108` saying `"homogeneous_modes: …"`) is also
fixed — `_warn_grazing(lam, tol, source)` takes the caller's name explicitly.

**Better fix (replaces the original `lam = lam/(|lam|+δ)` suggestion):** regularize the
*inverse* where it is consumed, not `lam` itself — this leaves propagation factors
`exp(lam·k0·d)` exact and only touches the ill-defined `V = Q·W·diag(1/λ)`:

```python
# in both solvers, replacing 1.0 / lam in the lam_inv construction:
delta   = 1e-30    # same scale as the Lorentzian sqrt regularizer in homogeneous.py
lam_inv = lam.conj() / (lam.abs() ** 2 + delta)     # == 1/lam away from 0, finite at 0
```

This matches the Lorentzian-regularization style already used twice in this codebase
(`homogeneous.py` sqrt, `Eig.backward`), and is exactly `1/lam` whenever `|lam| ≫ √δ`.
Implement it inside the shared helper proposed in A3 so it exists in one place.

**Tradeoff:** small controlled error only in near-grazing columns of `V`; documented
accuracy-degrading at true grazing (same caveat as before).
**Effort:** ~30 min (do together with A3).
**Sergei:** Fixed. Please add in both places regularization after the warnings.
**Resolved 2026-07-08:** done via `_modes.py` (see A3).

---

### E7 · MED · `tests/` · No patterned==homogeneous S-matrix regression test — ✅ Fixed

**Status:** ✅ Resolved 2026-07-08. Added:
- `test_layersolver.py::TestPatternedEqualsHomogeneous::test_uniform_pattern_equals_homogeneous`
  — checkerboard `eps_solid == eps_void`, parametrized over `tvf ∈ {off, on}` and
  `eps ∈ {1.0, 2.5}`; asserts the full `LayerSolver.solve(...)` S-matrix (not just Q) matches
  the closed-form homogeneous solve. The `eps=1.0` case covers "patterned-vacuum == homogeneous-
  vacuum" (item 2's intent) at the S-matrix level directly, so a separate eigsolver-only variant
  wasn't needed. All 8 combinations (2 tvf × 2 eps, ×2 devices when CUDA is present) pass.
- `test_isotropic.py::TestAPartitionOfUnity::test_A_blocks_partition_of_unity` — real TVF field
  on a checkerboard mask, asserts `[[Axx]] + [[Ayy]] ≈ I` to `atol=1e-6` (item 3).
- `tests/solver/test_eigsolver.py` (new) — E4's gradcheck (item 4; see E4 for detail).

Previously existing coverage (`test_isotropic.py::test_uniform_eps_Q_matches_homogeneous_Q`,
`test_layersolver.py::TestTVFSingleSliceEquivalence`) is unchanged and still passes.

**Better fix (sharper than the original):**
1. `test_uniform_pattern_equals_homogeneous` — use a **checkerboard pattern with
   `eps_solid == eps_void == ε`**, not a constant grid. The layer is then physically
   homogeneous, but the TVF is computed from a non-trivial mask, so it is well-defined
   (a constant grid has zero gradient and the TVF target is degenerate) and the Li correction
   must vanish *exactly* (`Δ = 0`). Assert
   `LayerSolver.solve(patterned) ≈ LayerSolver.solve(homogeneous)` with TVF **on and off**.
2. `test_eigsolver_vacuum_matches_homogeneous` — compare **S-matrices, not raw modes**:
   `torch.linalg.eig` returns eigenpairs in arbitrary order with arbitrary column scaling, so
   comparing `lam`/`W`/`V` element-wise against `homogeneous_modes` will spuriously fail.
   The S-matrix is basis- and permutation-invariant; that is the correct observable.
3. `test_A_blocks_partition_of_unity` — `[[Axx]] + [[Ayy]] ≈ I` on the checkerboard TVF output
   (direct check of the factorization.md constraint; localizes E2-class bugs to `compute_A`).
4. `test_eig_backward_gradcheck` — E4's gradcheck, kept with these tests.

**Effort:** ~2 h.
**Sergei:** You should implement this.

---

## Tier 2 — Interface & precomputation boundaries

The structural work in this tier is done (`LAYERSOLVER_PLAN.md` Steps 1–3). What remains is
deferred by design, not forgotten — see the rewritten D3/D4 for why.

---

### D1/C2 · HIGH · TVF precompute claim was false — ✅ Fixed

**What (historical):** CLAUDE.md and the `Solver` docstring claimed "`__init__` precomputes the
TVF; `solve()` is cheap", but the TVF optimization, convolution-matrix builds, and
eigendecomposition all ran inside every `solve()` call.

**Status:** ✅ Fixed — see `LAYERSOLVER_PLAN.md` Steps 1–3. Took the explicit-operator route
instead of the originally proposed `id(layer)`-keyed cache (rejected: a hidden cache retains
autograd graphs from stale optimizer steps — silent memory growth and wrong-leaf gradients).
- `LayerSolver.prepare(element)` does the expensive work (TVF field, `epsilon_conv`,
  eigendecomposition) and returns a frozen `LayerOperator(lam, W, V, thickness)`.
- `LayerSolver.smatrix(op, left)` is the cheap S-matrix assembly;
  `LayerSolver.solve()` = `smatrix(prepare(...))` for backward compatibility.
- `Solver.__init__` calls `prepare()` once per stack element; `Solver.solve()` only
  star-composes — the documented cost split is now actually true.
- `thickness` is read at `smatrix()` time, so thickness-only optimization reuses every
  eigendecomposition without rebuilding (`test_thickness_change_without_reprepare`).

Verified by `tests/solver/test_solver.py::TestSolverPrecompute` and
`tests/solver/test_layersolver.py::TestLayerSolverPrepare`. Inverse-design loops still rebuild
`Solver` per step when the pattern changes — that recompute is physically required and was
never the target; only the redundancy across repeated solves at fixed geometry was.

---

### C1 · HIGH · TVF Hessian materialized for full wavelength batch — ✅ Fixed

**What (historical):** `tvf.compute(eps_grid)` ran on `[N_wvl, Ny, Nx]`, building an
`[N_wvl, flat, flat]` Newton Hessian — `N_wvl` identical copies, since the TVF field is
wavelength-independent by construction (normalized, detached, quadratic A-blocks).

**Status:** ✅ Fixed — `LAYERSOLVER_PLAN.md` Step 1. `compute_A`/`compute_isotropic` now take
precomputed `(Tx, Ty)` fields; `LayerSolver._patterned` computes them once from the **pattern
mask** (`tvf.compute(pattern[None])`, batch 1) — more robust than `eps_grid[0]`, since it stays
well-defined when the real-part contrast crosses zero at some wavelength. Batch-1 A-blocks
broadcast against the batched `epsilon_conv` (verified: `Block` ops use native torch batch
broadcasting). Equivalence with the old batched path — including a negative-contrast
wavelength — is locked in by
`tests/solver/test_layersolver.py::TestTVFSingleSliceEquivalence`.

---

### D3/D4 · ~~MED-HIGH~~ → LOW · Source/lattice invariants fused into `Solver` — 💤 Deferred (rewritten 2026-07-08)

**What (updated):** `Solver.__init__` still computes reciprocal lattice, harmonic map,
`kx/ky`, `W0/V0`, TVF setup, and the per-element `prepare()` pass in one bundle, so an
inverse-design step rebuilds all of it.

**Why the severity dropped.** The original MED-HIGH rested on "all three groups rebuilt from
scratch every iteration" *when solve() was the expensive call*. After D1/C2 + C1 + D5:

- The **expensive** group (TVF field, `epsilon_conv`, A-blocks, eigendecomposition) is
  geometry-dependent — the pattern changed, so it must rerun. **No context split can save it.**
- The groups a context would save — harmonic map (integer tensors), `kx/ky` (broadcasting),
  `W0/V0` (closed-form DIAG modes), `TVF.__init__` (stores config) — are all trivially cheap.

The redundant per-step work is now low single-digit percent of a step (dominated by eig
forward+backward). **Measure before acting:** time `Solver.__init__` minus the `prepare()`
loop for one representative inverse-design model; if it's <5 % of a step, there is nothing here.

**Correction to the original invariance analysis.** The proposed two-phase fix
("reusable harmonic/source context" + "per-geometry layer pack") was based on a lifetime table
that is wrong for patterned layers: `Kx/Ky` enter `P` and `Q`, so **a source change invalidates
the eigendecompositions and layer packs too**. The true lifetimes are three-tier:

| Tier | Contents | Invalidated by |
|------|----------|----------------|
| geometry-only | TVF field, `epsilon_conv`, A-blocks | pattern change |
| source+lattice | harmonic map, `kx/ky`, `W0/V0` | source or lattice/config change |
| both | `P/Q`, eigendecomposition, `LayerOperator`, S-matrices | either |

A "layer pack reusable across an angle sweep" therefore doesn't exist — only the geometry-only
tier is, and angle/wavelength sweeps are **batched axes** (`[N_wvl, N_θ, N_φ, Nh]`) in this
codebase anyway; the only reason to Python-loop over sources is memory pressure.

**Why deferring is now safe (the ossification argument also weakened):** the seam already
exists. `LayerSolver`'s constructor state *is* the proposed `HarmonicContext`
(config, wvl, kx, ky, m_flat, n_flat, tvf, W0/V0), and `LayerOperator` is the per-element
product. A future extraction is a mechanical move of `Solver.__init__`'s harmonic block into a
factory, doable backwards-compatibly (`Solver(model, config, context=None)` builds one when
absent). No public API needs to break later.

**Better fix options (in preference order):**
1. **Do nothing until a second consumer exists.** `results/` (needs `kx/ky`, `k0`, `W0/V0`,
   harmonics for diffraction efficiencies) and the unbuilt `matrixexp` path are the consumers
   that should shape this boundary. Designing a context API with one consumer risks the wrong
   boundary.
2. **When done:** extract `HarmonicContext` ≈ current `LayerSolver` constructor state; keep
   `Solver(model, config)` working by building the context internally; expose
   `Solver(model, config, context=ctx)` for loops with fixed lattice+source. If the
   geometry-only tier ever matters (memory-limited source loops), cache the TVF field +
   `epsilon_conv` per pattern *explicitly on the caller's side*, mirroring the
   `prepare()`/operator philosophy.
3. **Explicitly rejected:** a `Solver(model, config, reuse_from=old_solver)` shortcut —
   implicit invalidation; a user who changes the wavelength and reuses the context gets
   silently wrong physics.

**Effort:** Structural (~2–3 days) *when triggered*; not now.

---

### D2 · ~~MED~~ → LOW · `solver/tvf/tvf.py:215–217` · `low_pass_mask`/`in_band_idx` re-derived per `TVF.compute()` — ⬜ Open (downgraded)

**What:** unchanged — the boolean mask and flat in-band indices are rebuilt from
`(D0, D1, M, N, device)` on every call, and `low_pass_filter` (`tvf_utils.py:212`) recomputes
the same mask again within the same call.

**Why downgraded:** after Steps 1–3, `TVF.compute()` runs **once per patterned layer per
`Solver` construction** (batch 1), not per `solve()` and not per wavelength. The waste is now
a few small tensor builds per optimization step.

**Fix (adjusted):** lazy cache keyed on `(D0, D1, device)` — the grid size is only known at
`compute()` time, so an `__init__`-time cache (original suggestion) doesn't work as stated.
Fold `low_pass_filter`'s internal mask into the same cache. Only worth bundling with other TVF
work.
**Effort:** ~1 h.

---

### D5 · MED · `epsilon_conv` rebuilt every `solve()` — ✅ Fixed

**Status:** ✅ Fixed via the D1/C2 restructure — `epsilon_conv` (and `eps_inv_conv`) are built
once inside `LayerSolver.prepare()`, called once per element in `Solver.__init__`;
`Solver.solve()` never re-enters `_patterned`. Note the low-level `LayerSolver.solve()`
convenience wrapper still rebuilds on every call *by design* — callers wanting reuse call
`prepare()` once and `smatrix()` repeatedly, as `Solver` does internally.
The remaining `epsilon_conv` inefficiency (redundant re-*factorization* within one `prepare()`)
is B2, which is now the top open perf item.

---

### D7 · LOW · `solver/config.py` · `Config` couples device/dtype with discretization — ⬜ Open (note only)

Unchanged: acceptable as-is; design-aesthetic, not a bug. Consider `PlacementConfig` vs
`DiscretizationConfig` only if the API grows.

---

## Tier 3 — Memory & compute efficiency (now the active tier)

With correctness and the precompute boundary settled, these are the highest-value open items.
Note the framing shift: for B2/B3 the hot path moved from `solve()` to `prepare()` — cost is
now per-`Solver`-construction, i.e. **once per inverse-design step**, which is still the loop
that matters.

---

### C3 · HIGH (peak-memory) · `eigsolver.py:186` · `Eig` saves full eigvec matrices for backward — ✅ Fixed

**Status:** ✅ Resolved 2026-07-08. Added `Config.checkpoint_eig: bool = False`; when set,
`LayerSolver._patterned` wraps the `eigsolver` call in `torch.utils.checkpoint.checkpoint(...,
use_reentrant=False)` exactly as proposed below. Verified `checkpoint(eigsolver, P, Q, ...)`
gives **bit-identical S-matrices and gradients** vs the non-checkpointed path (max grad diff
`0.0` on a synthetic case; see `tests/solver/test_layersolver.py::TestCheckpointEig`:
`test_checkpoint_matches_no_checkpoint`, `test_checkpoint_gradients_match`,
`test_config_checkpoint_eig_roundtrips`). Measured on CUDA with 8 stacked patterned layers
(single wavelength, `Nh=289`, batch=1 throughout — batching wavelength while a layer's
permittivity stays batch-1 hits an unrelated, pre-existing `S_prop` batch-broadcast limitation,
out of scope here): peak memory **2018 MB → 1926 MB (~4.5%)**, reproducible across repeated
trials (`test_checkpoint_reduces_peak_memory`, CUDA-gated). The gap is modest at 8 layers
because the checkpointed activations are only one contributor to peak memory (Block2x2
convolution matrices etc. are unaffected); it should widen with more patterned layers per stack
or larger `N_wvl`/`N_θ`/`N_φ` batches, per the original per-batch-element estimate below.

**What (historical):** `Eig.forward` saves `eigval` and `eigvec` (`[..., 2N, 2N]`) via `save_for_backward`
for every batch element; with `Kx/Ky` carrying the full `[N_wvl, N_θ, N_φ, Nh]` batch, all
eigvec stacks are held simultaneously for the backward pass. One stack at `N_wvl=100`,
`Nh=625` ≈ 1.2 GB in complex64, per patterned layer.

**New consideration from the prepare() split:** the eigendecomposition now happens in
`Solver.__init__`, so the saved-for-backward eigvecs live from **construction** until backward
(or until the `Solver` is dropped). A user who keeps one `Solver` alive for repeated
`solve()` calls under `torch.no_grad()` is fine, but in a training step the retention window
is the whole step — checkpointing is *more* attractive now, not less.

**Fix (unchanged, placement updated):** wrap the eigsolver call inside
`LayerSolver._patterned`:

```python
lam, W, V = torch.utils.checkpoint.checkpoint(
    eigsolver, P, Q, self.config.eigsolver_stable, use_reentrant=False)
```

Recompute happens once per patterned layer in backward. Gate behind a
`Config.checkpoint_eig: bool = False` switch so small-batch compute-bound runs keep the fast
path.

**Tradeoff:** +1 forward `eigsolver` per patterned layer in backward; net win only when
memory-bound. Measure `torch.cuda.max_memory_allocated` before/after on a realistic case.
**Effort:** ~2 h.

---

### B2 · MED · `isotropic.py` `compute_Qfact`/`compute_P` · `epsilon_conv` re-factorized up to 8× per patterned layer — ✅ Fixed

**Status:** ✅ Resolved 2026-07-08. Added `Block.solve_many(*rhs)` (`blockmatrix.py`): promotes
all `rhs` to DENSE, concatenates them along the column axis (broadcasting mismatched batch
shapes first — needed since the TVF `A`-blocks can be batch-1 per C1 while `epsilon_conv` is
fully batched), and issues a **single** `torch.linalg.solve` call, which LAPACK factorizes once
and reuses across every RHS column; SCALAR/DIAG `self` falls back to a per-rhs elementwise
inverse (no factorization to share there). `compute_P` and `compute_Qfact` now call
`epsilon_conv.solve_many(...)` / `epsilon_inv_conv.solve_many(...)` once each instead of 4
individual `.solve()` calls. Verified: (1) `tests/solver/test_blockmatrix.py::TestSolveMany`
includes a monkeypatched call-count regression guard (3 separate `.solve()` → 3
`torch.linalg.solve` calls; one `solve_many(...)` on the same 3 rhs → 1 call); (2)
`test_isotropic.py::TestComputeP/TestComputeQfact::test_matches_naive_four_solves` assert the
refactored functions produce bit-identical output to the original 4-solve formulation; (3)
measured end-to-end on one patterned-layer `prepare()` (checkerboard, `Nh_half=3`): **1**
`torch.linalg.solve` call for `compute_P` alone (down from 4), and **3** total with TVF on
(1 `compute_P` + 1 `compute_Qfact` + 1 unrelated TVF-Newton-step solve — down from what would
have been 9 before this fix). All existing `compute_P`/`compute_Qfact`/`compute_isotropic`
tests (incl. `test_uniform_eps_Q_matches_homogeneous_Q` and the TVF single-slice equivalence
test) pass unmodified.

**What (historical):** `compute_Qfact` (4× `epsilon_inv_conv.solve`) and `compute_P`
(4× `epsilon_conv.solve`) each trigger `Block.solve` → `torch.linalg.solve` → a fresh LU
factorization of the same DENSE `[..., Nh, Nh]` matrix. LU is `O(Nh³)` — the same order as
the eig itself. Post-refactor this is once per `prepare()` instead of per `solve()`, but that
is still every inverse-design step.

**Better fix (concrete, minimal API change):** stack the right-hand sides so each operator is
factorized once. `compute_P`'s four solves share `epsilon_conv` and DIAG RHS (`Kx`, `Ky`):

```python
# compute_P: one batched solve instead of four
rhs   = torch.stack([Kx_dense, Ky_dense], dim=0)         # or cat along columns
sols  = torch.linalg.solve(eps_dense.unsqueeze(0), rhs)   # single LU, two solves
```

or, cleaner and reusable: add `Block.solve_many(*rhs)` (uses
`torch.linalg.lu_factor` + `lu_solve` under the hood) and call it from both `compute_P` and
`compute_Qfact`. Avoid caching LU on the `Block` instance (hidden state on a value type);
thread it explicitly.

**Tradeoff:** zero accuracy cost; removes ~6–7 redundant `O(Nh³)` factorizations per patterned
layer per step.
**Effort:** ~0.5–1 day.

---

### A4/C4 · MED · `solver/smatrix.py:121–124` · `S_boundary` force-densifies all inputs — ⬜ Open

**What:** unchanged — every boundary (including all-DIAG homogeneous/vacuum ones) pays a dense
`[..., 4Nh, 4Nh]` `torch.linalg.solve`, bypassing the cheap `Block2x2.solve` Schur path, because
eig-derived DENSE mode matrices can have singular inner sub-blocks. Post-refactor this runs
inside `smatrix()` — i.e. on **every `Solver.solve()` call**, making it the largest per-solve
cost for stacks dominated by homogeneous layers.

**Fix (unchanged):** branch on input structure — if all leaf blocks are SCALAR/DIAG, use the
Schur path (full-rank by construction for DIAG mode matrices); fall through to the dense solve
only for DENSE eig-mode blocks.
**Tradeoff:** branching logic + validation that the DIAG Schur path is non-singular.
**Effort:** ~1 day + careful testing.

---

### B3 · ~~MED~~ → LOW-MED · `homogeneous.py:207–210` + `eigsolver.py:105–106` · `zero_mask.any()` host sync — ✅ Fixed

**Status:** ✅ Fixed together with A3. The grazing check now lives in
`_modes.py::_warn_grazing`, gated by module-level `WARN_GRAZING: bool = False` (default off),
called identically from both `homogeneous_modes` and `eigsolver` — one gate, not two/four
call sites. Production runs pay no host↔device sync unless the flag is explicitly enabled
(e.g. at problem setup, per the original recommendation).

**Fix (unchanged):** gate behind a debug/validation flag (`Config` or module-level), default
off in production; grazing incidence should be checked at problem setup, not in the hot path.
Implement inside the shared A3 helper so it's one gate, not four.
**Effort:** ~30 min (with A3).

---

### B5 · LOW · `eigsolver.py:236` · `inv(XH)` in `Eig.backward` — use `solve` — ⬜ Open

Unchanged: replace `torch.linalg.inv(XH) @ (grad_eigval + tmp)` with
`torch.linalg.solve(XH, grad_eigval + tmp)` — avoids materializing the inverse, better
conditioned. (Line moved from 225 → 236 after the E1 fix.)
**Effort:** ~15 min.

---

### B4 · LOW · `blockmatrix.py:266–267` · Redheffer star materializes `(I−P).inv()` — 💤 Wait

Unchanged: the inverse is reused for both `D` and `F`, so materialization is partially
justified. Revisit only if profiling after C3/B2 shows `star()` matters.

---

### C5/C6/B6 · INFO · Things that are (still) fine as-is

- **C5 (precision):** complex64 default via `Config.dtype = float32`; complex128 opt-in. OK.
- **C6 (W0/V0 cache):** vacuum modes cached as DIAG blocks in `LayerSolver.__init__`. OK.
- **B6 (no Fourier loops):** everything vectorized; only loops are over layers (inherently
  sequential Redheffer) and TVF Newton steps (default 1). OK.

---

## Tier 4 — Architecture cleanups (quick, low risk)

---

### A3 · MED · Duplicated branch-select + `lam_inv` logic in both mode solvers — ✅ Fixed

**Status:** ✅ Fixed 2026-07-08. Extracted into `solver/layersolver/_modes.py`:
`_branch_select(lam, tol)` (E5 rule), `_lam_inv_block(lam, N, delta)` (E6's Lorentzian
inverse), and `_warn_grazing(lam, tol, source)` gated by module-level `WARN_GRAZING` (B3).
Both `homogeneous_modes`/`homogeneous_kz` (`homogeneous.py`) and `eigsolver`
(`eigsolver.py`) now call these instead of duplicating the logic; `homogeneous_kz` was
refactored to branch-select in `lam`-space (`lam = 1j·kz`) so it reuses the same E5 rule as
`eigsolver`, converting back to `kz = -1j·lam` at the end — verified behavior-preserving
against the full `test_homogeneous.py`/`test_isotropic.py`/`test_layersolver.py` suites
(200/200 pass) plus the new `test_grazing_V_is_finite` and the updated grazing-warning tests
(`test_grazing_mode_emits_runtime_warning` now enables the gate via monkeypatch;
`test_no_warning_when_gate_disabled` covers the new default-off behavior).

**Fix:** extract into `solver/layersolver/_modes.py`:
`_branch_select(lam, tol)` (E5 rule), `_lam_inv_block(lam, N, delta)` (with E6's Lorentzian
inverse), and the gated grazing warning (B3). Call from both solvers.
**Effort:** ~1–2 h including E6/B3 pieces; do before or together with E6.

---

### A1 · MED · `isotropic.py:41`, `homogeneous.py:34` import `_REAL_TO_COMPLEX` from `model.base` — ⬜ Open

Unchanged (verified still present): the solver layer imports from the model package, breaking
the one-way model→solver rule. Move `_REAL_TO_COMPLEX` (+ `to_complex`/`to_real` from
`model/utils.py`) into a shared `metarcwa/_dtypes.py`.
**Effort:** ~1 h.

---

### A5 · LOW · `solver/utils.py` · `matrix_solve` stub returns `None` — ⬜ Open

Unchanged (verified still present): full docstring, body does nothing, no callers.
Delete the file.
**Effort:** ~10 min.

---

### A2 · LOW · `model/utils.py` grab-bag — ⬜ Open

Unchanged: split into `_dtypes.py` (with A1), `nn_helpers.py`, `adapters.py`. Wait until A1
is done since they share `_dtypes.py`.
**Effort:** ~1–2 h.

---

### A6 · LOW · `ModelSpec` manually re-aggregates sub-spec fields — ⬜ Open

Unchanged: `model/base.py:148–159` copies 5+5 fields into a flat frozen dataclass; every new
field threads through three places. Store `stack_spec`/`source_spec` directly, or
auto-generate. Lower priority than Tiers 1–3; becomes relevant when `results/` grows the spec.
**Effort:** ~half day.

---

### A7 · LOW · `Solver.__init__` mutates caller's `Model` in place — ⬜ Open

Unchanged (`solver/base.py`, `model.to(...)` in `__init__`). Minimum: document the in-place
mutation in the `Solver.__init__` docstring. Structural option (operate on a copy) only if it
bites someone.
**Effort:** quick (doc).

---

## Summary table (revised 2026-07-08)

| ID | Category | Severity | Status | Effort left |
|----|----------|----------|--------|-------------|
| E1 | Correctness | HIGH | ✅ Fixed | — |
| E2 | Correctness | HIGH | ✅ Fixed | — |
| E3 | Correctness | HIGH | ✅ Fixed | — |
| E5 | Correctness | MED | ✅ Fixed (logic; dedup → A3) | — |
| D1/C2 | Interface/Perf | HIGH | ✅ Fixed (prepare/operator split) | — |
| C1 | Memory | HIGH | ✅ Fixed (single-slice TVF) | — |
| D5 | Interface/Perf | MED | ✅ Fixed (via D1/C2) | — |
| E6 | Correctness | MED | ✅ Fixed (regularization via A3) | — |
| E7 | Correctness | MED | ✅ Fixed (S-matrix regression + partition-of-unity tests) | — |
| E4 | Correctness | MED | ✅ Fixed (gradcheck verified; docstring aligned) | — |
| C3 | Memory | HIGH | ✅ Fixed (`checkpoint_eig` config flag) | — |
| B2 | Compute | MED | ✅ Fixed (`Block.solve_many`) | — |
| A4/C4 | Arch/Memory | MED | ⬜ Open | ~1 d |
| A3 | Arch | MED | ✅ Fixed (`_modes.py` helper; E6+B3 folded in) | — |
| B3 | Compute | LOW-MED (↓) | ✅ Fixed (gated via A3) | — |
| B5 | Compute | LOW | ⬜ Open | ~15 min |
| A1 | Arch | MED | ⬜ Open | ~1 h |
| A5 | Arch | LOW | ⬜ Open | ~10 min |
| A2 | Arch | LOW | ⬜ Open | ~1–2 h |
| A6 | Arch | LOW | ⬜ Open | ~0.5 d |
| A7 | Arch | LOW | ⬜ Open | quick |
| D2 | Interface | LOW (↓) | ⬜ Open (downgraded) | ~1 h |
| D3/D4 | Interface | LOW (↓ from MED-HIGH) | 💤 Deferred — revisit at `results/`/`matrixexp` | 2–3 d when triggered |
| B4 | Compute | LOW | 💤 Wait for profiling | — |
| D7 | Arch | LOW | ⬜ Note only | — |

**Suggested order of work (revised):**
1. **Tier 1 tail** — A3 helper extraction carrying E6 regularization + B3 debug gate
   (~2 h), then E7 S-matrix regression tests + E4 gradcheck (~3 h). Closes out correctness.
2. **C3** gradient checkpointing behind a config flag — measure peak memory before/after (~2 h).
3. **B2** single-LU solves in `compute_P`/`compute_Qfact` (~0.5–1 d).
4. **Quick wins:** B5 inv→solve, A1 `_dtypes.py`, A5 delete stub (~1.5 h total).
5. **A4/C4** DIAG fast path in `S_boundary` (~1 d) — now the dominant per-`solve()` cost.
6. A2/A6/A7 at leisure.
7. **Deferred:** D3/D4 (revisit when `results/` or `matrixexp` lands, or if profiling shows
   context rebuild >5 % of an inverse-design step), D2 (bundle with other TVF work), B4
   (profile first).
