# Phase 8 — Observables and documentation: comments

## What was done
- `observables/base.py` (`Observables`, scattered-field formulation only, detected via
  `modal.background is not None`):
  - `jones` / `reflection` / `transmission` (order (0, 0)): background + scattered field projected on
    `e^{j k∥·r}`, averaged over the cell (`region="cell"`, default — equals the total-field (0, 0)
    order at normal incidence) or integrated over the physical region only and divided by the cell
    area (`region="physical"`).
  - `background_jones(kind)`, `scattered_coefficients(kind)` (harmonic coefficients on a face),
    `scattered_field(kind)` (real-space grid `[..., 4, 2, Ny, Nx]`: Ex, Ey, Ux, Uy × s, p),
    `scattered_power(kind, region="physical")`, `extinction(region="physical")`, both normalized to
    the incident power on one cell (`× cell area` = cross section).
  - The periodic (non-PML) and total-field paths are unchanged.
- `docs/PML.md`: status, explicit source vector `f`, TVF correction in the source, matexp
  augmented-exponential source, structure-in-PML warning, pipeline table, new "Observables" section,
  open questions / validation status.
- `CLAUDE.md`: model/solver/observables tables (`model/pml.py`, `solver/pml.py`, `background.py`,
  `scattered.py`, `AffineS`, `apply_vec`), PML domain concept, PML gotchas (incl. the crash), docs table
  (`PML.md`, which was missing), examples list.
- `examples/pml_oblique_incidence.ipynb` (new, executed, outputs stored): scattered vs total at
  normal incidence; PML vs periodic supercell at θ = 0.4; angle sweep of scattered power and
  extinction (batched angles); scattered field on the substrate face.
- `tests/pml_scattered_validation_bench.py` (new, not collected by pytest): the Phase 7 validation,
  runnable with `OMP_NUM_THREADS=1 python tests/pml_scattered_validation_bench.py`.

## Checks run
- Bench script output reproduces all Phase 7 numbers through the public API.
- Energy balance, lossless ridge: `extinction = scattered_power(R) + scattered_power(T)` to ~1e-3 at
  normal / moderate angles; the residual comes from the finite cell and is largest near grazing
  (relative ~1e-2 at L = 8, ≤ 4e-3 at L = 24 for θ = 1.2). With a lossy ridge extinction exceeds the
  scattered power (absorption), as expected. Scattered powers converge with L to ~4 digits.

## Decisions worth reviewing
- **Specular projection over the cell (default) vs the physical region.** "cell" keeps exact
  agreement with the total-field formulation at normal incidence, but includes the (analytically
  continued) field inside the PML; "physical" is more principled for isolated structures. Both
  converge to the same limit.
- **Normalization**: powers per incident power on one cell, so `L · P` (2D) or `A · P` (3D) is a cross
  section. Amplitudes: `A · (J − J_bg)` is the cell-independent specular scattering amplitude.
- **Extinction** uses only the two faces (no lateral boundaries); documented as such. If guided
  modes in films matter, it underestimates.
- **Notebook execution**: `nbclient` is not installed, so the notebook was executed by a small
  script that runs the cells and stores stdout/figures. Re-running it in Jupyter should give the
  same outputs.

## Not done
- Near / far fields (need `FieldSolution`); far-field angular spectrum of the scattered field.
- pytest tests for the new observables (per instruction).
