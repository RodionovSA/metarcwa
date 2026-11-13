# PML implementation plan

Implementation plan for the lateral PML described in [docs/PML.md](docs/PML.md). The work is split into phases. Each phase ends with passing tests and leaves the existing (periodic) behaviour unchanged.

Two milestones:

- **M1 — PML at normal incidence** (phases 0–4): RCWA + PML with the total field. Useful on its own and the reference for M2.
- **M2 — oblique incidence** (phases 5–8): scattered-field formulation with source vectors in the S-matrix algebra.

---

## Phase 0 — Preparatory refactors (no behaviour change)

Goal: make the patterned-layer path accept arbitrary `Kx`, `Ky` blocks.

- `layersolver/isotropic.py::compute_Q0`: replace `d = -a` with `d = Ky @ Kx`.
- `compute_isotropic`: accept `Kx`, `Ky` as `Block`s (DIAG today, DENSE under PML) instead of building them from `kx`, `ky` internally.
- Check that `Block` / `Block2x2` operations used by `compute_P` / `compute_Q*` (`solve_many`, `eye_like`, `@`) work with DENSE `Kx`, `Ky`.

Tests: the full existing suite passes unchanged; a unit test that `compute_P` / `compute_Q0` with `Kx`, `Ky` passed as DENSE (diagonal content) match the DIAG result.

## Phase 1 — Model API

Goal: the user-facing `PML` object and geometry handling, without touching the solver yet.

- New `model/pml.py`: `PML(width, profile=StretchAbsorbProfile(), method="stretch")` as an `nn.Module` (parameters via `register`), pluggable profiles `s(d)` (`StretchAbsorbProfile`, `PolynomialProfile`, or any `nn.Module`), and a frozen `PMLSpec` with $s_x$, $s_y$ sampled on the grid. Export `PML` and the profiles from `metarcwa/__init__.py`.
- `model/stack.py`: `Stack(..., pml=None)`. With a PML, require a rectangular lattice and validate the widths against the cell size.
- `model/layer.py::Layer.spec`: edge-pad the pattern into the PML region. Warn (or raise) when the structure reaches the PML.
- `ModelSpec` carries the `PMLSpec` (`None` when no PML).

Tests: profile values ($s = 1$ outside, continuity at the PML entrance and cell edge, $\mathrm{Im}\,s > 0$ inside); edge padding on simple masks; non-rectangular lattice raises; `Model.to()` moves PML parameters.

## Phase 2 — Stretched wavevector operators

Goal: $\tilde K_x$, $\tilde K_y$ in the patterned-layer path.

- New `solver/pml.py::stretched_k(kx, ky, pml_spec, m_flat, n_flat, factorization)` returning DENSE `Block`s. `[[1/s_x]]` comes from `convolution_matrix` applied to a grid that depends on $x$ only, which gives the $\delta_{nn'}$ structure automatically. `factorization="inverse"` uses $[[s_x]]^{-1}$.
- `solver/base.py::build_layersolver`: with a PML, set the Bloch phase to 0 along PML axes, build $\tilde K$ once and store it in the `LayerSolver` context.
- `LayerSolver._patterned`: pass $\tilde K$ to `compute_isotropic`.
- `Config`: `pml_factorization: "laurent" | "inverse"`. (`pml_formulation` is added in M2, when there is a second formulation to choose.)
- Oblique incidence along a PML axis raises `NotImplementedError` until M2; tilt along a non-PML axis is allowed.

Tests: with $s \equiv 1$, $\tilde K = K$ and results match the periodic solver bit-for-bit; $\tilde K_x e_0 = 0$ for $k_{x0} = 0$; $\tilde K_x \tilde K_y = \tilde K_y \tilde K_x$ for rectangular truncation; Laurent and inverse agree for a smooth profile as $N_h$ grows.

## Phase 3 — Homogeneous layers, half-spaces, background

Goal: everything that currently uses the closed-form `homogeneous_modes`.

- Numerical modes of uniform media: with uniform ε, $P = Q_0/\varepsilon$ exactly, so `compute_P`/`compute_Q0` (ε as a SCALAR block) + `eigsolver`, per element; works for both truncations. *Done this way.* Follow-up optimization: for rectangular truncation the eigenvectors are shared by all uniform media ($\lambda^2 = \mu_i - \varepsilon$, $\mu_i$ = eigenvalues of $\tilde K_x^2 + \tilde K_y^2$), and come from two 1D eigenproblems. *Shared basis done* (`LayerSolver._prepare_uniform_basis`, one $N_h$ eigendecomposition whenever $\tilde K_x$, $\tilde K_y$ commute); the two-1D-eigenproblem form is still open.
- `LayerSolver._homogeneous`, `_medium`, `_prepare_vacuum`: use the shared basis under PML.
- Decide the gap-medium question (stretched vs unstretched `W0`/`V0`) by experiment: both must give the same total S-matrix for a stack; keep the cheaper one. *Decided:* both agree to $\sim 10^{-12}$ relative, so the unstretched (DIAG) vacuum gap is kept.
- `matexpsolver`: slice-count bound from $\tilde K$ (e.g. a norm bound) instead of $\max(k_x^2 + k_y^2)$.
- Branch selection under PML. The discretized stretched operator is not exactly passive: truncation error can give a nearly lossless mode a tiny gain ($\mathrm{Im}\,\lambda^2 > 0$), so neither root $\pm\lambda$ is both decaying and forward. `_branch_select` then falls back on its generic tolerance ($|\mathrm{Re}\,\lambda| \le 10^{-4}|\lambda|$) and returns either a slightly growing forward mode or a decaying *backward* mode. Seen for patterned layers with both factorizations (e.g. circular truncation, $M = 6$–$10$, $|\mathrm{Re}\,\lambda/\lambda| \sim 10^{-6}$–$10^{-4}$). Harmless in finite layers (the layer S-matrix does not depend on the labelling; $|X| - 1 \sim 10^{-4}$), but in the half-spaces the branch defines the outgoing wave, so a backward label is a physics error. Fix for this ambiguous case: when $\mathrm{Im}\,\lambda^2$ has the gain sign and $|\mathrm{Re}\,\lambda|$ is at the discretization-error level, select by the sign of $\mathrm{Im}\,\lambda$ (phase direction = energy-flow direction for a nearly lossless mode). *Implemented as `_branch_select_direction` and later removed on purpose:* a scan of 270 configurations found no gain-quadrant mode in uniform half-spaces, so it was only a safeguard; the issue is confined to patterned layers, where the labelling is harmless. Half-spaces use `_branch_select`.

Tests: with $s \equiv 1$ the numerical basis reproduces the closed form; a uniform stack with PML at normal incidence gives the Fresnel / TMM zeroth-order R and T (the zeroth harmonic is an exact mode); `eig` and `matexp` agree under PML; half-space branch selection: every mode is forward ($\mathrm{Im}\,\lambda > 0$ for nearly lossless modes) in a case with a guided mode whose discretized $\mathrm{Im}\,\lambda^2$ has the gain sign.

## Phase 4 — Validation at normal incidence (milestone M1)

Goal: show the PML works physically, using the total-field formulation.

- Minimal observables: zeroth-order (specular) R and T, which are well-defined at normal incidence.
- Convergence studies: supercell size, PML width, $a_\text{max}$, $\sigma_\text{max}$, $N_h$; Laurent vs inverse factorization.
- Slab waveguide: effective index of the guided mode vs the analytic dispersion relation.
- Isolated scatterer: result stabilizes as the supercell grows (no more array coupling); cross-check against fmmax.
- New notebook in `examples/` once the results are clean.

## Phase 5 — Background solver

Goal: the incident field $\mathbf E_\text{inc}$ at any angle.

- New `solver/background.py`: 1-harmonic transfer-matrix solve of the background stack (each patterned layer filled with `medium_void`), for s and p. Output: up/down amplitudes in every layer, each referenced to the face where the wave enters the layer.
- Runs in `run()`, since it depends on the source only.
- *Done:* `solver/background.py::solve_background` → `BackgroundField` (wired into `run()` in phase 6).

Tests: Fresnel coefficients for a single interface; agreement with the existing solver for a stack of homogeneous layers at oblique incidence.

## Phase 6 — Source vectors and extended star product

Goal: the scattered-field formulation in the S-matrix algebra.

- Source term $f$: FFT of $\Delta\varepsilon\,e^{j\mathbf k_\parallel\cdot\mathbf r}$ (a vector per layer and angle), combined with the $\mathbf E_\text{inc}$ amplitudes, including the $[S_z]$ elimination contribution to the $s$-equations.
- Layer source vector: modal projection $\Phi^{-1}f$, closed-form integrals with a stable $\varphi_1$, then through the two boundary factors.
- Affine S-matrix: an `(S, e_L, e_R)` container (e.g. in `smatrix.py`) with an extended `star` that composes $S$ as today and propagates the source vectors. $e$ has two columns (s, p).
- `run()`: compose the affine S-matrices; `ModalSolution` gains `e_L`, `e_R`.
- *Done:* `solver/scattered.py`, `smatrix.AffineS`, `blockmatrix.apply_vec/solve_vec`, `Config.pml_formulation` (default `"total"` until phase 7). With TVF the source also carries the TVF correction applied to $\mathbf E_\text{inc}$ (`Qfact`), so that scattered = total at normal incidence exactly (see `Phase_6_comment.md`).

Tests:
- Extended star is associative, and reduces to the plain star when $e = 0$.
- $\varphi_1$ is accurate near $\gamma = \lambda$.
- Layer source vector vs brute force: the matrix exponential of the augmented system $\bigl(\begin{smallmatrix}A & f\\ 0 & \gamma\end{smallmatrix}\bigr)$ gives the exact particular solution.
- Gradients finite through the source path (small `gradcheck` in `float64`).

## Phase 7 — Validation at oblique incidence (milestone M2)

- Normal incidence: scattered-field result equals the total-field result from phase 4.
- Oblique incidence: convergence towards a large periodic supercell as the cell grows; an independent reference for a single scatterer (Mie theory for a sphere, or published aperiodic-FMM results).
- Switch the default of `pml_formulation` to `"scattered"`.
- *Done:* default switched; normal incidence and tilt along the periodic axis agree with the total field to ~1e-14; tilt along the PML axis converges to the large periodic supercell limit at ~1e-3 (`Phase_7_comment.md`, `tests/pml_scattered_validation_bench.py`). Mie / aperiodic-FMM comparison not done.

## Phase 8 — Observables and documentation

- Define the PML observables: total R/T (background specular + scattered), scattered power, extinction; near/far field later, together with `FieldSolution`.
- Update `docs/PML.md` with the decisions taken (gap medium, observables), the `CLAUDE.md` docs table and architecture notes, and add an example notebook for oblique illumination.
- *Done:* `Observables` gains `background_jones`, `scattered_coefficients`, `scattered_field`, `scattered_power`, `extinction` and the specular `jones` under the scattered formulation (`region="cell" | "physical"`); `docs/PML.md` "Observables"; `CLAUDE.md`; `examples/pml_oblique_incidence.ipynb` (see `Phase_8_comment.md`). Near/far fields remain with `FieldSolution`.

---

## Out of scope for now

- Oblique lattices with PML.
- Route B (uniaxial PML material), which needs anisotropic and magnetic media.
- Dipole and other embedded sources (they reuse the phase 6 source-vector machinery, but need `FieldSolution`).
- The Hugonin–Lalanne nonlinear coordinate map.
