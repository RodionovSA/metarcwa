# Phase 7 — Validation at oblique incidence (milestone M2): comments

## What was done
- `Config.pml_formulation` now defaults to `"scattered"` (PLAN.md). `"total"` remains for validation.
  At normal incidence both give the same S-matrix, so `Observables` results of the Phase 4 tests
  are unchanged; `run()` additionally does the (cheap) background + source work.
- Validation (numbers below). The runnable version is `tests/pml_scattered_validation_bench.py`,
  added in the Phase 8 commit because it uses the Phase 8 `Observables` API
  (`jones()` under the scattered formulation, `background_jones()`).

Test structure: dielectric ridge (ε = 4, width 0.5, height 0.3) in air on glass (ε = 2.25),
λ = 0.8, uniform along y, PML along x, float64, rectangular truncation, no TVF unless noted.
Quantity at oblique incidence: the specular scattering amplitude per unit length
`L·(J − J_bg)` (Jones matrix of background + scattered field projected on `e^{j k∥·r}` over the
cell, minus the background), which must be independent of the cell size `L`.

| Check | Result |
|---|---|
| Normal incidence, scattered vs total, {eig, matexp} × {no TVF, TVF} × {R, T} (ridge + film) | ≤ 5e-14 |
| Tilt along the periodic axis y (θ = 0.3, 0.7; both formulations valid) | ≤ 2e-14 |
| Ridge shifted by 0.37 along x at θ = 0.4: specular amplitude invariant | 3.7e-3 abs. (cell-window effect) |
| θ = 0.4 along x, PML cell L = 2 … 8: `L·dJ` (|·| ≈ 0.25–0.85) | stable to ~1e-2 |
| PML parameters at L = 4 (width 0.75/1.25, σ_max 1/2, a_max 4/8) | variation ~1e-2 |
| Periodic supercell (no PML) Lp = 4, 8, 16, 24: `Lp·(J₀₀ − J_bg)` | converges (Lp=16→24 change ≤ 1e-3) |
| PML L = 8/12/16 vs periodic Lp = 24 | max diff 3e-3 / 3e-3 / 2.5e-3 (w = 2), 2e-3 (w = 3) |

So at oblique incidence along the PML axis the PML scattered-field result converges to the same
limit as an (independent) large periodic supercell, at the 1e-3 level for amplitudes ~0.8.

## Not done
- Mie theory / published aperiodic-FMM comparison (PLAN.md): not done. The periodic-supercell
  limit was used as the independent reference instead.
- No pytest tests added (per instruction).

## Notes
- The remaining ~1e-3–1e-2 spread with L comes mostly from the observable: the projection over the
  cell window truncates the slowly decaying lateral tails of the scattered field. Wider PMLs reduce
  it (w = 3 at L = 16 is best). A physical-region-only projection is offered in Phase 8.
- All runs used `OMP_NUM_THREADS` ≤ 4; see the crash note in `Phase_6_comment.md`.
