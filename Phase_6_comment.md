# Phase 6 — Source vectors and extended star product: comments

## What was done
- `solver/blockmatrix.py`: `apply_vec(entry, v)` / `solve_vec(entry, v)` — operator × dense vector
  (`[..., rows, ncol]`) for `Block`/`Block2x2` without densifying SCALAR/DIAG leaves (solve densifies).
- `solver/smatrix.py`: `AffineS(S, e_L, e_R)` with `.star()` — the extended Redheffer star of
  docs/PML.md (`None` = zero source, reduces to the plain star).
- New `solver/scattered.py`:
  - `layer_source(...)` → `SourceTerms(f [*B,4Nh,4], lam_bg)`. `g = FFT(Δε · e^{j k∥·r})` (phase along
    PML axes only; pixel `i` at `x = i·px/nx`, the FFT convention of `convolution_matrix`),
    `f = (−jK̃x[[ε]]⁻¹F_z, −jK̃y[[ε]]⁻¹F_z, −F_y, F_x)`. Columns: forward-s, forward-p (referenced to
    the left face), backward-s, backward-p (referenced to the right face).
  - `modal_affine` (eig operators): closed-form integrals in the eigenbasis with a stable
    `phi1` / `exp_integral` (exact at γ = λ).
  - `transfer_affine` (matexp operators): per slice, the exponential of the augmented system
    `[[A, f], [0, Γ]]` gives `T` and the exact particular solution in one `matrix_exp`
    (size 4Nh+4); slices are recombined by repeated squaring with the phase bookkeeping
    (`e^{μ k0 L}` scales the right block's forward columns / the left block's backward columns).
  - `layer_affine` adds forward+backward columns → 2 columns (s, p) per layer.
  - `check_structure_in_physical_region`: warns in `prepare()` if `ε − ε_void ≠ 0` inside the PML.
- `Config.pml_formulation: "total" | "scattered"` (default `"total"`; Phase 7 switches it).
  With `"scattered"`, oblique incidence along a PML axis is allowed.
- `run()`: under `"scattered"` → `_run_scattered`: background solve, per-layer affine S, extended
  star. `ModalSolution` gained `e_L`, `e_R` (`[..., 2Nh, 2]`, half-space mode bases) and `background`.

## Decision worth reviewing: TVF in the source
With TVF the plan's source `[[Δε]]·E_inc` (Laurent) is *not* consistent with the total-field
formulation: the TVF rule describes D–E of the **total** field (normal D of the total field is
continuous, not of E_s). I therefore add the TVF correction applied to E_inc to the u-part of the
source: `f_u += Qfact · (φ E_x, φ E_y)`, `Qfact = Q − Q0`, `φ` = Fourier coefficients of
`e^{j k∥·r}`. `Qfact` is kept on the operator (`ModalOperator.qfact`, `TransferOperator.qfact`,
new optional fields; only set under PML + TVF) via `compute_isotropic(..., return_qfact=True)`.
Without this, scattered vs total differed by ~1e-2 with TVF; with it they agree to ~1e-14.

## Checks run (scratch scripts, single-threaded, not the test suite)
- Normal incidence, isolated ridge (ε=4) on a film + substrate, x-PML, cell 2, m=12:
  scattered-field Jones (background + e_L/e_R at (0,0)) vs the Phase 4 total-field Jones,
  for {eig, matexp} × {no TVF, TVF} × {R, T}: max difference 1.3e-15 … 4.7e-14.
- Modal closed form vs brute-force augmented matrix exponential (one layer): ~3e-13.
- Extended star associativity (random affine S): ~1e-15; with e = 0 it is the plain star.
- `phi1` vs mpmath over |x| ∈ [1e-9, 30], complex: ≤ 2.5e-14 relative.
- Oblique incidence (θ = 0.3/0.5, several φ): runs, finite. (Physical validation is Phase 7.)
- Gradient of |e_L|² w.r.t. ε_solid at oblique incidence: finite.

## Notes / open points
- **Pre-existing crash**: intermittent segfault / glibc heap abort in multithreaded CPU runs of a
  PML + TVF + matexp model (torch 2.13.0+cu130). It reproduces on the original `PML-feature` code
  (5 of 10 runs of a total-field solve), and disappears with `OMP_NUM_THREADS=1`. Not caused by
  these changes, but worth investigating (maybe a torch bug in a batched LAPACK/matrix_exp kernel).
- `tests/solver/test_pml_stretch.py::test_oblique_along_pml_axis_raises` now expects `ValueError`
  (message unchanged) and sets `pml_formulation="total"`. Not run.
- The extended star solves `(I − S_B11 S_A22)` again for the vectors (separate dense solves),
  so composition costs ~1.5× the plain star. Could be fused later.
- `modal_affine` solves with the dense `Φ` (4Nh) per run; caching its LU in `prepare()` would save
  that. Under PML the default is matexp anyway.
- Phase 6 tests from PLAN.md (associativity, `phi1`, brute force, gradcheck) correspond to the checks
  above but are not written as pytest tests.
