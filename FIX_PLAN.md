# PML branch: fixes and improvements plan

Follow-up to the review of `PML-feature-Claude`. Ordered from small to large; each step is one
commit and is checked with small probe scripts (the full pytest suite is not run).

**Status: all steps done** (item 8 dropped, see below). Only `tests/solver/test_pml_scattered.py`
was run from the test suite.

## Step 1 — Bugs

1. **Uniform stack + PML crashes `Observables`** under the scattered formulation (`e_L`/`e_R` are
   `None` when no layer has a source). Fix: a missing source is a zero scattered field.
2. **`dataclasses.replace(op, thickness=...)` gives a wrong result**: the background solve reads
   thicknesses from `model_spec`, the layers from the operators. Fix: the background takes the
   thicknesses of `prepared.ops`.
3. **Physical-region mask from `s == 1`** is empty for a profile with `s(0) = 1` only to round-off.
   Fix: `PMLSpec` carries the mask (`physical_x`, `physical_y`) from the same distance test as
   `PML.pad`.
4. **"Structure inside the PML" warning always fires for soft patterns** (`|Δε| > 0`). Fix: relative
   tolerance.

## Step 2 — Robustness

5. `reprepare()` repeats the PML-overlap check for the re-prepared layers.
6. `scattering_pattern` raises for a lossy half-space (no far field there).
7. Resonance guard on the scattered matexp path (same warning as `star_power`).
8. ~~PML-overlap check also under the total-field formulation.~~ Dropped: under the total-field
   formulation a structure continuing through the PML (edge padding, e.g. a waveguide core) is
   valid by design, so a warning there would be wrong.

## Step 3 — Duplication, caching, repeated work

9. One helper for the regularized ε grid, shared by `LayerSolver._patterned` and `scattered.py`.
10. `Observables` caches the scattered fields per face (and the specular Jones matrices).
11. `[[ε]]` of a patterned layer is kept on the operator under PML (next to `qfact`) instead of
    being rebuilt on every `run()`.
12. `_affine_slice`: one factorization for the S-matrix and the source vector.
13. `AffineS.star`: one factorization of `(I − S_B11 S_A22)` / `(I − S_A22 S_B11)` for the matrix
    and vector right-hand sides.

Also done here: the LU of the gap matrix `Φ` of eigenmode layers is computed in `prepare()` and
reused for every source vector (`SourceData.phi_lu`).

## Step 4 — Shared basis for uniform media under PML

14. When `K̃x`, `K̃y` commute (rectangular truncation, or a PML on one axis only),
    `P Q = (K̃x² + K̃y² − ε) ⊗ I₂`, so every uniform medium shares the eigenvectors `X` of
    `K̃x² + K̃y²` and `λ² = μ − ε`. One `Nh` eigendecomposition in `LayerSolver` replaces one `2Nh`
    eigendecomposition per homogeneous layer and half-space. Otherwise the current path is kept.

## Step 5 — Docs, hygiene, tests

15. PLAN.md / `Phase_N_comment.md`: `_branch_select_direction` (removed on purpose), crash notes.
16. `docs/PML.md`: "Proposed API" heading, the "~50×" figure, the "S-matrices are not affected" wording.
17. `solver/base.py` docstrings mention the scattered path; `README.md` mentions the PML.
18. `tests/solver/test_pml_scattered.py`: small, fast tests for phases 5–8 and the fixes above
    (only this file is run).
