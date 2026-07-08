# metarcwa — Architecture & Efficiency Analysis

**Date:** 2026-06-27  
**Branch:** redesign  
**Method:** Read-only source review (3 parallel analysis agents + direct source verification).
No profiling numbers were collected — agents reviewed source only; no clean GPU benchmark was run.
If real `torch.cuda.max_memory_allocated` / profiler traces are wanted, a follow-up pass can do that.

Findings are ordered **expensive-to-change and correctness-critical first**.
Each entry: `Category · Severity · file:line · what + why · recommendation + tradeoff · effort`.

---

## Tier 1 — Correctness bugs (patterned-layer path is currently wrong)

These three confirmed bugs mean every patterned layer produces wrong mode propagation.
They are independent of each other and each is a small, local fix.

---

### E1 · HIGH · `solver/layersolver/eigsolver.py:97-101` · Extra `1j` factor on modal exponents

**What:** The eigenproblem is `Ω²·w = λ²·w` where `λ² = P·Q` eigenvalues.
The homogeneous convention (shared per the docstring) defines `lam = 1j·kz`, so `lam²`
are the `PQ` eigenvalues. The code takes `kz = sqrt(lam_sq)` and then sets `lam = 1j * kz`
— applying `1j` to something already equal to `lam`. Net effect: `lam = 1j·sqrt(lam²) = 1j·lam`.

**Proof (vacuum, normal incidence):** `PQ = −I` → `lam_sq = −1` → `kz = sqrt(−1) = 1j` →
`lam = 1j · 1j = −1` (real, decaying). `S_prop` then applies `exp(lam·k0·d) = exp(−k0·d)` — real
exponential decay — whereas `homogeneous_modes` returns `lam = +1j` for the same case,
producing `exp(+1j·k0·d)` (unit-modulus phase). **The two solvers disagree for an identical medium.**

**Side effects:** `is_ev = kz.real.abs() < tol` (line 98) classifies propagating modes (`Re(kz)=0`)
as evanescent and vice-versa, so the branch correction also flips sign for the wrong population.

**Why it matters:** Every patterned layer mis-propagates (exponential attenuation instead of phase),
energy conservation is violated, and every inverse-design gradient through patterned layers is wrong.

**Fix:** The eigenvalues are already `λ²`. Replace lines 97–101 with:
```python
# lam^2 from PQ are the modal exponents squared; take sqrt with branch selection
# Propagating: Im(lam) > 0 (phase); evanescent: Re(lam) < 0 (decay)
# Match homogeneous_kz convention: lam = 1j*kz, kz defined with Re(kz)>0 propagating
lam = torch.sqrt(lam_sq)                         # [..., 2N], initial branch
is_ev  = lam.imag.abs() < tol                    # True for purely propagating
sign   = torch.where(is_ev, torch.sign(lam.real), torch.sign(lam.imag))
sign   = torch.where(sign == 0, torch.ones_like(sign), sign)
lam    = lam * sign
```
Then add a regression test (see E7). The exact branch rule must be cross-validated against
`homogeneous_modes` output on a uniform patterned grid.

**Tradeoff:** None — the current code is wrong.  
**Effort:** ~1–2 h including test.

**Sergei:** Fixed.

---

### E2 · HIGH · `solver/layersolver/isotropic.py:116-122` · TVF A-blocks double-FFT'd (autocorrelation, not convolution)

**What:** `tvf.compute()` returns spatial-domain fields `Tx, Ty` shaped `[B, Ny, Nx]`.
`compute_A` FFTs them (`Tx_fft = fft2(Tx)`) and passes `|Ty_fft|²` to `convolution_matrix`,
which FFTs *again*. The result is the Toeplitz matrix of `FFT(|FFT(Ty)|²)` = the **autocorrelation**
of `Ty`, not the Fourier coefficients of `|Ty(x,y)|²` required by `docs/factorization.md` Eq. (14).

**Proof:** For a uniform field `Ty=c`, `|Ty_fft|²` has a single spike at DC of amplitude
`(Ny·Nx)²·|c|²`; the correct `|Ty(r)|²=|c|²` has a uniform real-space value → flat Fourier spectrum.
The resulting `Axx`, `Ayy` are wrong, and the identity `[[Axx]] + [[Ayy]] = I`
(`docs/factorization.md:236–244`) is violated.

**Fix:** Form the outer-product components in real space before passing to `convolution_matrix`:
```python
Tx, Ty = tvf.compute(epsilon_grid)   # spatial, [B, Ny, Nx]
axx = Ty.abs() ** 2                  # |Ty|^2  (not FFT first)
axy = Tx.conj() * Ty                 # Tx*·Ty
ayx = Tx * Ty.conj()                 # Tx·Ty*
ayy = Tx.abs() ** 2                  # |Tx|^2
Axx = Block(Block.DENSE, convolution_matrix(axx, m_flat, n_flat))
# ... etc.
```
Delete the `fft2` calls at lines 116–117; `convolution_matrix` handles the transform.

**Tradeoff:** None — current code is wrong; the fix eliminates one redundant FFT pair (smaller allocation, faster).  
**Effort:** ~15 min.

**Sergei:** Fixed.

---

### E3 · HIGH · `solver/layersolver/isotropic.py:162-165` · Inverse rule missing: `Δ` uses `[[ε]]⁻¹` instead of `[[1/ε]]⁻¹`

**What:** Li's factorization correction (`docs/factorization.md:227`, `docs/rcwa_core.md:201`) is:

    Δ = [[ε]] − [[ε⁻¹]]⁻¹

where `[[ε⁻¹]]` is the Toeplitz matrix of the Fourier coefficients of `1/ε(r)`.
The code computes `epsilon_conv.solve(Ayx)` = `[[ε]]⁻¹ · Ayx` — using the *inverse* of the
direct convolution matrix, not the convolution of the *reciprocal* permittivity.
`[[ε]]⁻¹ ≠ [[1/ε]]⁻¹` in general; this is the entire point of Li's rule.

**Proof (uniform ε=4):** Correct: `[[1/ε]] = (1/4)I` → `[[1/ε]]⁻¹ = 4I = [[ε]]` → `Δ=0`.
Code: `epsilon_conv.solve(A) = (1/4)I·A`, so effective `Δ = 4I − 0.25I = 3.75I` per A-block.
A large spurious correction is added for a pattern where the correction should be zero.

**Note:** `compute_P` (lines 231–234) correctly uses `[[ε]]⁻¹` via `epsilon_conv.solve` — that
application is right. Only `Δ` in `compute_Qfact` is wrong.

**Fix:** Build the reciprocal-permittivity convolution and use its inverse in `Δ`:
```python
# In compute_isotropic, after building epsilon_conv:
eps_inv_grid = 1.0 / epsilon_grid
eps_inv_conv = Block(Block.DENSE, convolution_matrix(eps_inv_grid, m_flat, n_flat))
# Pass eps_inv_conv to compute_Qfact alongside epsilon_conv.
# In compute_Qfact:
delta = epsilon_conv - eps_inv_conv.inv()   # [[ε]] − [[1/ε]]^{-1}
a_fact = -delta @ Ayx
b_fact =  delta @ Ayy
c_fact = -delta @ Axx
d_fact =  delta @ Axy
```
This builds one additional `Nh×Nh` DENSE block per patterned layer (the `eps_inv_conv`) and
one dense inverse — similar cost to the existing `epsilon_conv`.

**Tradeoff:** +1 DENSE `Nh×Nh` allocation and +1 `inv` per patterned layer. Unavoidable; the math requires it.  
**Effort:** ~30 min.

**Sergei:** Fixed.
---

### E4 · MED · `solver/layersolver/eigsolver.py:211-226` · `Eig.backward` uses `conj(F)` — gradcheck needed

**What:** Standard non-symmetric-eig adjoint uses `F_ij = 1/(λ_j − λ_i)`. The class docstring
and comment say the Lorentzian replaces `1/(λ_j−λ_i)` with `conj(λ_j−λ_i)/(|λ_j−λ_i|²+ε)`.
Line 222 applies `conj(F)` a *second* time to an already-complex `F`, effectively using
`(λ_j−λ_i)/(|.|²+ε)` instead of the conjugate form — may be intentional for complex eigenvalues
(see Boeddeker et al. 2020) but is unverified. Additionally, the implementation zeros `diag(F)`
but does not add the eigenvector normalization/gauge correction term that is needed when
`grad_eigvec ≠ 0` with non-unitary eigenvector matrices.

**Fix:** Run `torch.autograd.gradcheck(Eig.apply, ...)` on small well-separated complex matrices
with both `grad_eigval` and `grad_eigvec` nonzero. Fix only if gradcheck fails.

**Tradeoff:** If correct, no change. If wrong, fixing may change convergence of gradient-based optimization.  
**Effort:** ~1 h to gradcheck + fix if needed.

**Sergei:** Not sure here. As stated in the file's top, I use implementation from Torcwa.
So, maybe better to modify docstring. If you fix it, do not add docstrings of uncertanity. Just modify. 
---

### E5 · MED · `eigsolver.py:98` vs `homogeneous.py:96` · Inconsistent propagating/evanescent classifier

**What:** Two different thresholds and axes:
- `homogeneous_kz:96`: `is_evan = kz.imag.abs() > tol` (tol=1e-12) — evanescent if Im large
- `eigsolver:98`: `is_ev = kz.real.abs() < tol` (tol=1e-12) — inverted logic; also wrong for
  the post-E1 fix (kz is already `lam` after the fix, not `sqrt(lam_sq)`)

For lossy media with complex `kz`, the two paths can select opposite decay directions.

**Fix:** After fixing E1, extract a shared `_branch_select(lam, tol)` helper and use it in both
solvers. Define one consistent rule aligned with `docs/rcwa_core.md` convention.  
**Effort:** ~30 min (folds naturally into E1 fix).
**Sergei:** Fixed with E1.
---

### E6 · MED · `homogeneous.py:208`/`eigsolver.py` · Grazing λ=0 → NaN; eig path warns nowhere

**What:** The `sign==0` fixup at `homogeneous.py:102` and `eigsolver.py:100` corrects the branch
sign but does not prevent `1/lam → Inf` in `lam_inv` when `lam≈0`. `homogeneous_modes` issues a
`RuntimeWarning` when `|lam|<tol` (line 208). `eigsolver` has no equivalent warning, so the same
NaN columns in `V` appear silently in the patterned path.

**Fix:** Add the same `RuntimeWarning` in `eigsolver`, aligned with the same `tol` parameter.
Optionally add a regularizer `lam = lam / (|lam| + δ)` with tiny `δ` to clip the NaN for
robustness; mark it as accuracy-degrading at true grazing.  
**Effort:** ~30 min.
**Sergei:** Fixed. Please add in both places regularization after the warnings. 
---

### E7 · MED · `tests/solver/test_layersolver.py:235` · No patterned==homogeneous regression test

**What:** `TestLayerSolverPatterned` only checks shape, absence of NaN, and off-diagonal
nonzero — not that the physics is right. A uniform patterned layer with constant `ε=ε₀`
must reproduce the `HomogeneousLayer` S-matrix exactly (after E1–E3 fixes). This test would
have caught all three high-severity bugs.

**Fix:** Add:
1. `test_uniform_pattern_equals_homogeneous`: constant eps grid → `S_patterned ≈ S_homogeneous`
   for both TVF off and TVF on (`[[Axx]]+[[Ayy]] ≈ I` separately).
2. `test_eigsolver_vacuum_matches_homogeneous`: `eigsolver` on vacuum PQ returns same `lam, V`
   as `homogeneous_modes`.
3. `test_eig_backward_gradcheck`: `Eig.apply` passes `gradcheck`.

**Effort:** ~1–2 h.
**Sergei:** You should implement this. 
---

## Tier 2 — Interface & precomputation boundaries (expensive to change later)

Design decisions whose cost grows with time. Fix the API shape before it ossifies.

---

### D1/C2 · HIGH · `solver/base.py:64-75` + `solver/layersolver/base.py:175-181` + `solver/layersolver/isotropic.py:285` · TVF precompute claim is false

**What:** CLAUDE.md (line 49 when written) and the `Solver` docstring state "`__init__`
precomputes the TVF; `solve()` is cheap." In reality:
- `__init__` only *constructs* the `TVF` object (stores config, `M`, `N`, `D0/D1`).
- The actual TVF optimization — Newton solve, FFTs, Hessian materialization, convolution-matrix
  builds — runs inside `compute_A` (`isotropic.py:115`), called from `_patterned`
  (`layersolver/base.py:178-181`), which runs **every `solve()` call**.
- TVF depends only on fixed geometry (it's normalized and detached: `tvf.py:206`), so this
  work is **repeated identically** on every spectral sweep step.

**Why it matters:** This is falsely advertised as cheap. The Newton Hessian alone allocates
`[B, flat, flat]` where `B=N_wvl` and `flat ≈ 4Nh` (e.g. `flat≈2500` for `M=N=12`),
making this one of the largest transient allocations — repeated every `solve()`.

**Fix:** Cache `(Tx, Ty)` (or the A-blocks) per patterned layer in `LayerSolver.__init__`,
keyed on `id(layer)` or an explicit geometry hash. On `solve()` reuse cached A-blocks if
geometry is unchanged. For inverse-design loops (pattern changes per step), the cache must be
invalidated — gate behind an explicit `static_geometry=True` flag on `Solver`, or always
recompute (but then fix the documentation to be honest about cost).

**Tradeoff (caching):** Stores 4 DENSE `[..., Nh, Nh]` A-blocks per patterned layer. For a
10-layer stack at `Nh=625` in complex64, that's `4·625²·8·10 ≈ 125 MB` — significant but
much less than the per-call Newton Hessian allocation for large wavelength batches.  
**Effort:** Structural (~1–2 days for full caching + invalidation).

**Status:** Fixed — see `LAYERSOLVER_PLAN.md` Steps 1–3. Took the explicit-operator route
instead of an `id(layer)`-keyed cache (rejected in the plan: a hidden cache would retain
autograd graphs from a stale optimizer step). `LayerSolver.prepare(element)` now does the
expensive work (TVF, `epsilon_conv`, eigendecomposition) and returns a `LayerOperator`;
`LayerSolver.smatrix(op)` is the cheap S-matrix assembly. `Solver.__init__` calls `prepare()`
once per stack element and `Solver.solve()` only calls `smatrix()` — the CLAUDE.md claim
"`__init__` expensive, `solve()` cheap" is now actually true, verified by
`tests/solver/test_solver.py::TestSolverPrecompute::test_solve_is_deterministic_across_calls`
and `test_ops_precomputed_at_init`. The caller (an inverse-design loop) still rebuilds
`Solver` every step when the pattern changes, same as before this fix — geometry-dependent
recompute was never eliminated, only the *redundant re-computation across repeated `solve()`
calls at fixed geometry* was.

---

### C1 · HIGH · `solver/tvf/optimizers.py:196` + `isotropic.py:285` · TVF Hessian materialized for full wavelength batch (N_wvl-fold redundant)

**What:** `tvf.compute(eps_grid)` is called with `eps_grid` shaped `[N_wvl, Ny, Nx]` (all
wavelengths batched). The TVF optimization (`optimizers.py:196–217`) builds a dense Hessian `H`
of shape `[N_wvl, flat, flat]` where `flat ≈ 4Nh`. Since the TVF field is normalized and
detached from wavelength-dependent material parameters (`tvf.py:206`, `compute()` takes
`real(eps_grid).detach()` for the mask), the tangent field is **identical across all wavelengths**.
The Newton solve does `N_wvl` copies of the same computation.

**Fix:** Compute TVF once on a single-slice geometry (e.g. `eps_grid[0]` or the bare binary
pattern mask), then broadcast `Tx, Ty` to the wavelength batch for `compute_A`. The `[1, Ny, Nx]`
TVF input reduces the Hessian to `[1, flat, flat]` — an `N_wvl`-fold reduction in the largest
transient allocation.

**Tradeoff:** Zero accuracy cost (the field is wavelength-independent by design). Complexity:
`compute_A` must broadcast `Tx[0]`/`Ty[0]` to match `eps_grid`'s batch dim.  
**Effort:** Quick-to-moderate (~2–4 h).

**Status:** Fixed — see `LAYERSOLVER_PLAN.md` Step 1. `compute_A`/`compute_isotropic` now take
precomputed `(Tx, Ty)` fields instead of a `TVF` instance; `_patterned` calls
`tvf.compute(pattern[None])` (batch 1) instead of on the full `eps_grid` batch. Verified equal
to the old per-wavelength-batched result (`tests/solver/test_layersolver.py::TestTVFSingleSliceEquivalence`),
including at a wavelength with negative solid/void contrast.

---

### D3/D4 · MED-HIGH · `solver/base.py:53-75`, `solver/layersolver/base.py:95-105` · Source/lattice invariants fused into `Solver` (full rebuild per inverse-design step)

**What:** `Solver.__init__` computes in one bundle: reciprocal lattice vectors, harmonic index
map `(m_flat, n_flat)`, `kx/ky` arrays `[N_wvl, N_θ, N_φ, Nh]`, vacuum modes `W0/V0`, and
the TVF+patterned geometry. All of these have different invariance lifetimes:
- **Lattice-invariant:** reciprocal lattice, harmonic map, truncation mask — only change if
  `Lattice` or `Config.m/n/nx/ny` change.
- **Source-invariant:** `kx/ky`, `W0/V0` — change only when `Source` (wavelength, angles) changes.
- **Geometry-dependent:** `TVF`, `epsilon_conv`, `A-blocks` — change when pattern changes.

In the stated primary use case (gradient-based inverse design with fixed lattice+source, varying
pattern), **all three groups are rebuilt from scratch every iteration** even though only the last
group needs updating.

**Fix:** Split `Solver` into a reusable "harmonic/source context" (lattice→reciprocal→kx,ky→W0/V0,
harmonic map) constructed once per (lattice+source+config), and a per-geometry "layer pack" that
holds TVF+A-blocks+epsilon_conv. The `solve()` method only needs the geometry-dependent part.

**Tradeoff:** A two-phase API (`HarmonicContext` + `Solver(ctx, geometry)`) is more complex but
correctly reflects the problem structure. This is the kind of boundary that is cheap to get right
now and expensive to retrofit later.  
**Effort:** Structural (~2–3 days).

---

### D2 · MED · `solver/tvf/tvf.py:215-217` · `low_pass_mask` and `in_band_idx` re-derived on every `TVF.compute()`

**What:** Lines 215–217 recompute `low_pass_mask` (shape `[D0, D1]` boolean) and `in_band_idx`
(flat indices) from `(D0, D1, M, N, device)` on every call. These depend only on `(D0, D1, M, N)`,
all set at construction. `low_pass_filter` (`tvf_utils.py:212`) independently recomputes the same
mask again within the same call.

**Fix:** Compute and cache `low_pass_mask`, `in_band_idx`, and the `low_pass_filter` grid in
`TVF.__init__`. Move `low_pass_filter`'s mask into the cached version. Check device matches on
first use and `.to(device)` if needed.

**Tradeoff:** ~3 small tensors stored per TVF instance (negligible memory). Gain: clean separation
of initialization work from compute work.  
**Effort:** Quick win (~1 h).

---

### D5 · MED · `solver/layersolver/isotropic.py:277` · `epsilon_conv` rebuilt every `solve()`

**What:** `convolution_matrix(epsilon_grid, m_flat, n_flat)` at line 277 runs on every call to
`_patterned`, which runs every `solve()`. `epsilon_grid` and `m_flat/n_flat` are fixed for a
fixed pattern. The FFT + index gather = `O(Ny·Nx·log(Ny·Nx))` + `O(Nh²)` allocation per call.

**Fix:** Cache `epsilon_conv` (and `eps_inv_conv` from E3) per patterned layer alongside the
A-blocks (folds into D1/D3 refactor). For the inverse-design case where ε changes, the cache
must be invalidated (same logic as TVF cache).

**Tradeoff:** Stores one DENSE `[..., Nh, Nh]` (or `[Nh, Nh]` if batch collapsed) per patterned
layer. Shares the lifecycle with the A-block cache (D1). If geometry changes per step (optimization),
both must re-run — the value is in spectral sweeps at fixed geometry.  
**Effort:** Structural (folds into D1/D3 refactor).

**Status:** Fixed via the D1/C2 fix — `epsilon_conv` (and `eps_inv_conv`) are built once inside
`LayerSolver.prepare()`, called once per element in `Solver.__init__`; `Solver.solve()` never
re-enters `_patterned`, so `epsilon_conv` is not rebuilt across repeated `solve()` calls at fixed
geometry. Note `LayerSolver.solve()` (the low-level `prepare()+smatrix()` convenience wrapper,
distinct from `Solver.solve()`) still rebuilds it on every call by design — callers who want reuse
should call `LayerSolver.prepare()` once and `LayerSolver.smatrix()` repeatedly, same as `Solver`
does internally.

---

### D7 · LOW · `solver/config.py:125-134` · `Config` couples device/dtype with discretization

**What:** `Config` bundles `dtype`/`device` (runtime placement, causes in-place `model.to()` in
`Solver.__init__`) with `m/n/nx/ny/truncation` (math discretization) and
`factorization/eigsolver_stable/modesolver` (algorithm switches). A device change looks identical
to a truncation change to callers, though they have very different implications.

**Note:** Acceptable to keep as-is; this is a design aesthetic issue, not a bug. Consider
separating if the API grows (e.g. a `PlacementConfig` vs `DiscretizationConfig`). Low priority.  
**Effort:** Structural if changed; note only for now.

---

## Tier 3 — Memory & compute efficiency

Address after correctness bugs and boundaries are settled.

---

### C3 · HIGH (peak-memory) · `solver/layersolver/eigsolver.py:174` · Eig saves full eigvec matrices for backward across entire batch

**What:** `Eig.forward` saves both `eigval` and `eigvec` (shape `[..., 2N, 2N]`) via
`save_for_backward` for every batch element. Because `Kx/Ky` carry the full
`[N_wvl, N_θ, N_φ, Nh]` batch, `Omega2_dense` has shape `[N_wvl, N_θ, N_φ, 2N, 2N]` and
**all these eigvec stacks are held simultaneously in memory for the backward pass**. With multiple
patterned layers, each layer's eigvec stack is retained concurrently.

**Fix:** Wrap each patterned layer's `eigsolver` call in `torch.utils.checkpoint`:
```python
lam, W, V = torch.utils.checkpoint.checkpoint(eigsolver, P, Q, stable_eig_grad)
```
This drops the saved eigvecs and re-runs `torch.linalg.eig` in the backward pass.

**Tradeoff:** +1 full `eigsolver` call per patterned layer in backward. At `N_wvl=100`,
`N_θ=1`, `Nh=625` (25×25 harmonics), one eigvec stack ≈ `100·(2·625)²·8 ≈ 1.2 GB` per layer
in complex64 — clearly worth checkpointing when multiple layers are present. Net win only when
memory-bound; pure-compute-bound small batches get slower.  
**Effort:** Quick win (~2 h to add + test that backward still works).

---

### B2 · MED · `solver/layersolver/isotropic.py:162-165, 231-234` · `epsilon_conv` re-factorized up to 8× per patterned layer per call

**What:** `compute_Qfact` (4 calls to `epsilon_conv.solve`) and `compute_P` (4 calls to
`epsilon_conv.solve`) each trigger `Block.solve` → `torch.linalg.solve` → implicit LU
factorization of the same DENSE `[..., Nh, Nh]` matrix up to 8 times per layer per `solve()`.
LU factorization is `O(Nh³)` — the same work as one of the solve calls themselves.

**Fix:** Factor `epsilon_conv` once explicitly and reuse:
```python
# torch.linalg.lu_factor + lu_solve, or stack RHSs into one solve
eps_LU, eps_P = torch.linalg.lu_factor(epsilon_conv.d)
# Then: eps_solve = lambda B: torch.linalg.lu_solve(eps_LU, eps_P, B)
```
Or simpler: concatenate all RHS column matrices into one `[..., Nh, 8Nh]` solve per operator.
Also applies to `eps_inv_conv` added in E3.

**Tradeoff:** Zero accuracy cost; saves ~7 redundant `O(Nh³)` factorizations per patterned layer
per call. Requires threading the factored form through `compute_P`/`compute_Qfact` or
refactoring `Block` to expose a cached-LU solve path.  
**Effort:** Moderate (~0.5–1 day).

---

### A4/C4 · MED · `solver/smatrix.py:121-124` · `S_boundary` force-densifies all inputs, bypassing `Block` abstraction

**What:** For every boundary (including homogeneous/vacuum, where all inputs are DIAG or SCALAR),
`S_boundary` calls `.to_dense(Nh)` on both mode matrices and runs `torch.linalg.solve` on the
full `[..., 4Nh, 4Nh]` system (line 124). The existing `Block2x2.solve` Schur path (which works
cheaply for DIAG inputs) is bypassed — the comment (lines 118–120) explains this is because
the Schur path can hit singular inner sub-blocks for eig-derived DENSE modes. But this means
the all-DIAG homogeneous boundary — the common case in most stacks — also pays the full dense cost.

**Fix:** Branch on input structure: if all leaf blocks are SCALAR or DIAG, route to the Schur
path (or a direct closed-form boundary for the vacuum case); only fall through to the dense
`4Nh×4Nh` solve for DENSE eig-mode blocks where the Schur path is unreliable.

**Tradeoff:** Adds branching logic; the Schur path needs to be validated as non-singular for
the DIAG case (it is, since DIAG mode matrices are full-rank). Saves a `4Nh×4Nh` dense solve
and allocation on every homogeneous boundary per call.  
**Effort:** Moderate (~1 day + careful testing).

---

### B3 · MED · `solver/layersolver/homogeneous.py:207-210` · `zero_mask.any()` host sync on every homogeneous boundary

**What:** `any()` on a CUDA tensor triggers a host↔device sync; `.item()` (line 210) syncs again.
This is called for every homogeneous layer, medium boundary, and vacuum precompute — several
times per `solve()`. Each sync stalls the GPU pipeline.

**Fix:** Gate behind a debug flag:
```python
if torch.is_grad_enabled() or check_grazing:   # or a solver-level debug flag
    if zero_mask.any(): ...
```
Default: skip in production; the behavior is documented as "not auto-handled" anyway.

**Tradeoff:** Loses the warning. Users should check at problem setup, not in the hot path.  
**Effort:** Quick win (~30 min).

---

### B5 · LOW · `solver/layersolver/eigsolver.py:225` · `inv(XH)` in `Eig.backward` — use `solve` instead

**What:** Line 225: `torch.linalg.inv(XH)` materializes the full eigenvector-matrix inverse,
then matmuls. Replace with `torch.linalg.solve(XH, grad_eigval + tmp)` to avoid forming the
explicit inverse (more numerically stable and avoids allocating the dense inverse matrix).

**Tradeoff:** Negligible accuracy/speed cost. The backward already dominates compute.  
**Effort:** Quick win (~15 min).

---

### B4 · LOW · `solver/blockmatrix.py:266-267` · Redheffer star materializes explicit `Block2x2.inv`

**What:** `Block2x2.star` at line 266–267 forms `(I−P).inv()` explicitly and then matmuls twice.
For the DENSE case this materializes a dense inverse. Where the inverse is only used as `inv @ rhs`,
`.solve(rhs)` is preferable.

**Tradeoff:** In `star`, `(I−P).inv()` is reused in forming both `D` and `F`, so the
materialization is partially justified. The net win (one avoided inverse) is small relative to
the eig cost. Address after E1–E3 and B2.  
**Effort:** Moderate; wait until eig is the confirmed bottleneck.

---

### C5/C6/B6 · INFO · Things that are fine as-is

- **C5 (precision):** complex64 is already the default eig path (`Config.dtype = float32` →
  `_REAL_TO_COMPLEX` → complex64). complex128 is opt-in via `Config.dtype = float64`. No change needed.
- **C6 (W0/V0 cache):** `LayerSolver._prepare_vacuum` caches vacuum modes as DIAG blocks (tiny).
  Memory cost is negligible; keep cached.
- **B6 (no Fourier loops):** No Python loops over harmonic indices anywhere (harmonics, conv,
  TVF Newton all vectorized). The only Python loops are over layers (small, unavoidably sequential
  due to Redheffer associativity) and TVF Newton steps (default 1). No action needed.

---

## Tier 4 — Architecture cleanups (quick, low risk)

---

### A1 · MED · `isotropic.py:39`, `homogeneous.py:34` · `solver` imports `_REAL_TO_COMPLEX` from `model.base` (leaky boundary)

**What:** The one-way model→solver data-flow rule (documented in CLAUDE.md) is broken:
`solver/layersolver/isotropic.py:39` and `homogeneous.py:34` both import `_REAL_TO_COMPLEX` from
`metarcwa.model.base`, pulling in the entire model package from the numerics layer.

**Fix:** Move `_REAL_TO_COMPLEX` (and `to_complex`/`to_real` helpers from `model/utils.py:10-54`)
into a shared `metarcwa/_dtypes.py` module imported by both sides.  
**Effort:** Quick win (~1 h including updating all import sites).

---

### A3 · MED · Duplicated `lam_inv` block + branch-sign logic in both mode solvers

**What:** The `lam_inv` Block2x2 construction (DIAG, `1/lam[:N]` + `1/lam[N:]`) and the
branch-sign fixup `torch.where(sign==0, ones, sign)` are verbatim duplicates in
`homogeneous.py:218-224` and `eigsolver.py:112-118`. The sign convention is the historically
bug-prone zone (per CLAUDE.md and commit history), and duplication means a future fix must be
applied in two places.

**Fix:** Extract `_lam_inv_block(lam, Nh, device, dtype)` and `_branch_select(kz, tol)` into
`solver/layersolver/_modes.py`. Call from both solvers. Folds naturally with E1/E5 fixes.  
**Effort:** Quick win (~1 h).

---

### A5 · LOW · `solver/utils.py:23` — `matrix_solve` is a documented stub returning `None`

**What:** `matrix_solve` has a full docstring but the body is `pass`. Any caller gets `None`
silently. The real solve logic is in `Block.solve` (`blockmatrix.py:168`).

**Fix:** Delete `solver/utils.py` or move it to `_deprecated.py`. No known callers (the function
is not imported anywhere). Quick win; avoids trapping a future contributor.  
**Effort:** ~10 min.

---

### A2 · LOW · `model/utils.py` is a grab-bag of four unrelated concerns

**What:** `model/utils.py` mixes: dtype maps + converters (consumed by solver — see A1),
`register`/`CallableModule` (model-internal nn helpers), and `from_metashapes`/`from_dispertorch`
adapter functions (user-facing glue). Adding a feature to any one concern requires understanding
all four.

**Fix:** Split into `_dtypes.py` (shared with solver, folds into A1), `nn_helpers.py`,
`adapters.py`. Mechanical refactor; no logic changes.  
**Effort:** ~1–2 h, but wait until A1 is done since they share `_dtypes.py`.

---

### A6 · LOW · `ModelSpec` manually re-aggregates `StackSpec` + `SourceSpec` fields

**What:** `model/base.py:148-159` copies 5 stack fields and 5 source fields into a flat frozen
dataclass. `Solver.__init__` reads them back individually. Every new field in a sub-spec must
be threaded through three places (sub-spec, ModelSpec, Solver reader).

**Fix:** Store `stack_spec` and `source_spec` as direct fields on `ModelSpec`, or auto-generate
the flattening via `__init_subclass__`. The current design is workable but becomes painful when
the spec grows.  
**Effort:** Structural (~half day); lower priority than Tier 1–3.

---

### A7 · LOW · `Solver.__init__` mutates caller's `Model` in place

**What:** `solver/base.py:50` calls `model.to(dtype=..., device=...)`, which `nn.Module.to`
applies in-place. A user constructing two Solvers with different devices on the same Model
object gets surprising results. Ownership is undocumented.

**Fix:** Document the in-place mutation explicitly in the `Solver.__init__` docstring (quick
win), or operate on a `model.to(...)` copy (structural but avoids surprises).  
**Effort:** Quick win (doc) / structural (copy).

---

## Summary table

| ID | Category | Severity | Effort | Group |
|----|----------|----------|--------|-------|
| E1 | Correctness | **HIGH** | 1–2 h | Tier 1 |
| E2 | Correctness | **HIGH** | 15 min | Tier 1 |
| E3 | Correctness | **HIGH** | 30 min | Tier 1 |
| E7 | Correctness | MED | 1–2 h | Tier 1 |
| E4 | Correctness | MED | 1 h | Tier 1 |
| E5 | Correctness | MED | 30 min | Tier 1 |
| E6 | Correctness | MED | 30 min | Tier 1 |
| D1/C2 | Interface/Perf | **HIGH** | Structural | Tier 2 |
| C1 | Memory | **HIGH** | 2–4 h | Tier 2 |
| D3/D4 | Interface | MED-HIGH | Structural | Tier 2 |
| D2 | Interface | MED | 1 h | Tier 2 |
| D5 | Interface/Perf | MED | Structural | Tier 2 |
| C3 | Memory | HIGH | 2 h | Tier 3 |
| B2 | Compute | MED | 0.5–1 d | Tier 3 |
| A4/C4 | Arch/Memory | MED | 1 d | Tier 3 |
| B3 | Compute | MED | 30 min | Tier 3 |
| B5 | Compute | LOW | 15 min | Tier 3 |
| A1 | Arch | MED | 1 h | Tier 4 |
| A3 | Arch | MED | 1 h | Tier 4 |
| A5 | Arch | LOW | 10 min | Tier 4 |
| A2 | Arch | LOW | 1–2 h | Tier 4 |
| A6 | Arch | LOW | 0.5 d | Tier 4 |
| A7 | Arch | LOW | quick | Tier 4 |

**Suggested order of work:**
1. Fix E1 + E2 + E3 in one pass (~1 h total), then add E7 regression tests (~1 h).
2. Run existing tests — they should now catch any regression.
3. E4 gradcheck (~1 h); E5+E6 branch cleanup (~1 h).
4. C1 quick TVF wavelength-batch fix (~2–4 h) — biggest pure-compute win, independent of structure.
5. B3 host-sync removal + B5 inv→solve + A1 import fix + A3 helper extract (quick wins, ~2 h total).
6. D1/C2/D3/D4/D5 structural API refactor — plan separately after correctness is confirmed.
7. C3 gradient checkpointing — measure actual peak memory first.
8. Remaining Tier 3–4 items at leisure.
