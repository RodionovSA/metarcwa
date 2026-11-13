# metarcwa

PyTorch-based Rigorous Coupled-Wave Analysis (RCWA) solver for 3D periodic/metasurface
electromagnetic structures. Fully GPU-accelerated and autograd-differentiable, making it
suitable for inverse design and gradient-based optimization. The numerical backend is pure
PyTorch (no numpy/scipy/jax). Python 3.12+, `src/` layout, package `metarcwa`.

---

## Key commands

```bash
# Dev install (canonical workflow)
uv sync --all-extras --dev

# As a dependency
uv add git+https://github.com/RodionovSA/metarcwa.git
pip install git+https://github.com/RodionovSA/metarcwa.git

# Optional extras (separate git repos)
uv sync --extra metashapes   # geometry / rasterization  (adapter: model.adapters.from_metashapes)
uv sync --extra dispertorch  # dispersion / material models (adapter: model.adapters.from_dispertorch)
uv sync --extra ridb         # refractive-index database

# Tests
pytest   # pythonpath=["src"] set in pyproject.toml; suites in tests/{model,solver,observables}
         # tests/*_bench.py, tvf_memory_profile.py are benchmarks, not tests
         # (tests/pml_scattered_validation_bench.py: PML oblique-incidence validation)

# Config-driven solve
config = Config.from_yaml("config.yaml")
```

- `examples/` — curated, validated notebooks: `uniform_layer` / `interface` (checked against
  TMM/Fresnel), `1d_grating`, `2d_grating`, `compare_factorization`, `compare_solvers`,
  `convergence_test`, `pml_isolated_structures` (PML, normal incidence), `pml_oblique_incidence`
  (PML scattered-field formulation).
- `notebooks/` — scratch/dev notebooks (`example`, `model_tests`, `test_solver`, `tvf_playground`, …).

> Lint / type-check: none configured. No CI (`.github/` absent).

---

## Architecture

### Data flow (one-way: model → solver)

```
Model (structure + source)
  └─ .spec(nx, ny) ──► ModelSpec  (immutable snapshot)
       └─ Solver(model, config)  ≡ prepare(model, config) ──► PreparedStack
            │    LayerSolver.prepare(element) ──► LayerOperator  (one per stack element)
            └─ Solver.run() ≡ run(prepared) ──► ModalSolution(S11, S21, prepared)
                 └─ Observables(solution) ──► jones / reflection / transmission per order
```

`solver/base.py` is a functional core (`build_layersolver`, `prepare`, `reprepare`, `run`)
with `Solver` as a thin wrapper. `Solver.__init__` (= `prepare`) is the expensive step under
the default `modesolver="eig"`: it moves the model to `config.device/dtype`, precomputes
harmonics + TVF, and solves every stack element's modes once, caching a `LayerOperator` per
element in a frozen `PreparedStack`. `Solver.run()` is cheap: Redheffer star-product composition
of the cached operators (via `LayerSolver.smatrix`), no eigendecomposition; it keeps only
`S11`/`S21` (left excitation). (`modesolver="matexp"` inverts this split — see Gotchas.)

Re-preparing after changes:
- **Whole-stack geometry / source / lattice change** → new `Solver` (re-`prepare`).
- **One or a few layers' pattern changes** → `solver.reprepare(layer_indices)`: reuses the
  `LayerSolver` context and all other layers' operators (one eig per step in inverse design).
- **Thickness only** → nothing; operators hold a reference to the thickness tensor and read it
  at `smatrix()` time, so autograd / in-place `nn.Parameter` updates just work.

`Solver.run(fields=True)` / `FieldSolution` / `FieldObservables` are **not implemented**
(raise `NotImplementedError`).

### `src/metarcwa/model/` — problem description

| File | Responsibility |
|------|----------------|
| `base.py` | `Model` + `ModelSpec`: top-level container; `.spec()` produces solver snapshot |
| `stack.py` | `Stack` + `StackSpec`: ordered layer sequence + incidence/transmission media |
| `layer.py` | `Layer`, `HomogeneousLayer`, `PatternedLayer`: per-layer permittivity + thickness |
| `medium.py` | `Medium`, `IsotropicMedium`: material / permittivity definitions for half-spaces |
| `source.py` | `Source`: wavelength, incidence angles (theta/phi). No polarization — s/p amplitudes are applied downstream by `Observables`, not needed for the S-matrix itself |
| `lattice.py` | `Lattice`: 2D Bravais lattice vectors `a1`, `a2`; `Lattice.rectangular(px, py)`, `Lattice.hexagonal(a, orientation=...)` |
| `pml.py` | `PML(width, profile, method="stretch")` lateral PML (rectangular lattice only), profiles `StretchAbsorbProfile` / `PolynomialProfile`; `PMLSpec(sx, sy, physical_x, physical_y)` stretch + physical-region masks on the grid; `PML.pad` edge-pads patterns into the PML (`Stack(pml=...)`) |
| `adapters.py` | `from_metashapes(shape, soft=, softness=)` → `shape_fn`; `from_dispertorch(model)` → `eps_fn` (optional-extra bridges) |
| `nn_helpers.py` | `register` (tensor → param/buffer), `CallableModule` (wrap a callable + its deps as an `nn.Module`) |

### `src/metarcwa/solver/` — numerics

| File/pkg | Responsibility |
|----------|----------------|
| `base.py` | Functional core `build_layersolver` / `prepare` / `reprepare` / `run` + `PreparedStack`, `ModalSolution` (`S11`, `S21`), `FieldSolution` (stub); `Solver` is a thin class wrapper (`.run()`, `.reprepare(layer_indices)`) |
| `config.py` | `Config` + `Factorization` dataclasses; grid/truncation/dtype/device/solver switches |
| `harmonics.py` | Harmonic index map `(m,n)`, truncation, in-plane wavevectors `kx`/`ky` |
| `smatrix.py` | `S_boundary`, `S_prop`, `S_layer`: per-interface/layer S-matrix builders; `AffineS(S, e_L, e_R)` + extended `.star()` for layers with sources (PML scattered field) |
| `pml.py` | `stretched_k`: dense stretched `K̃x = [[1/s_x]] Kx`, `K̃y` (Laurent or inverse) |
| `background.py` | `solve_background` → `BackgroundField`: 1-harmonic solve of the background stack (patterned → `medium_void`) for unit s/p incidence = `E_inc` per layer; built from `homogeneous_modes` / `S_boundary` / `S_prop` / `.star` with size-1 DIAG leaves |
| `polarization.py` | `sp_vectors` / `sp_matrix`: the s/p tangential-E basis (with the `rho → 0` fallback), shared by `background.py` and `Observables` |
| `scattered.py` | PML scattered-field source: `layer_source` (FFT of `Δε e^{jk∥·r}` → `f`), `modal_affine` (closed form, stable `phi1`), `transfer_affine` (augmented `matrix_exp` per slice + phase-shifted squaring), `layer_affine` |
| `blockmatrix.py` | `Block` / `Block2x2`: structured operator algebra + Redheffer star product; `apply_vec` / `solve_vec` (operator × dense vector) |
| `convolution.py` | Fourier convolution matrix helpers |
| `layersolver/base.py` | `LayerSolver`: `.prepare(element)` builds a `LayerOperator` (expensive: TVF/convolution/eigendecomp, unless `modesolver="matexp"` — see gotchas); `.smatrix(op)` delegates to `op.smatrix(background)` (cheap, unless `matexp`); `.run(element)` = both (there is no `.solve()`); precomputes vacuum modes `W0`/`V0`/`background` |
| `layersolver/operator.py` | `LayerOperator`: structural `Protocol` (`thickness`, `.smatrix()`, `.transfer()`), not one concrete type — matches the `Entry` Protocol pattern in `blockmatrix.py`. `ModalOperator` is the `(lam, W, V, thickness)` implementation used by `eigsolver`/`homogeneous_modes`. `Background` bundles `W0`/`V0`/`wvl`. |
| `layersolver/homogeneous.py` | `homogeneous_modes`: closed-form modes (no eigensolver) |
| `layersolver/_modes.py` | Shared by both mode solvers: `_branch_select`, regularized `1/λ` (`_lam_inv_block`), `_regularize_eps` (grazing), opt-in `_warn_grazing` |
| `layersolver/isotropic.py` | `compute_isotropic`: builds patterned-layer eigenproblem matrices (`P`, `Q` — shared by both modesolvers) |
| `layersolver/eigsolver.py` | `eigsolver`: numerical eigendecomp with stable autograd gradient |
| `layersolver/matexpsolver.py` | `TransferOperator`: the other `LayerOperator` implementation, for `modesolver="matexp"`. No eigendecomposition — slices the layer, exponentiates `A=[[0,P],[Q,0]]` per slice (`torch.linalg.matrix_exp`), converts to an S-matrix referenced to a per-layer "gap medium" (`Config.matexp_gap`, default the layer's mean permittivity — not vacuum, which reduces a fictitious-sub-slab resonance artifact of slicing), recombines via `star_power` (`O(log n)` Redheffer star products), then sandwiches back to the stack's vacuum background. See `docs/matrixexp.md`. |
| `tvf/tvf.py` | `TVF`: Tangent Vector Field for Li/FFF factorization |
| `tvf/tvf_utils.py` | TVF field math: periodic gradients, Fourier loss, normalization |
| `tvf/optimizers.py` | `make_optimizer`: Newton + other TVF direction-field optimizers |

### `src/metarcwa/observables/base.py` — post-processing

| Class | Responsibility |
|-------|----------------|
| `Observables(modal_solution)` | s/p Jones matrices per diffraction channel: `.jones(kind, in_order, out_order)` → `[..., 2, 2]`; `.reflection(pol)` / `.transmission(pol)` → (co, cross) amplitudes; `.scatter(excitation, ...)` for arbitrary `(a_s, a_p)`. Returns amplitudes, not power — the caller applies `|·|²` and the transmission Poynting-flux factor (see `examples/interface.ipynb`). Under a PML only order `(0, 0)`; with the scattered-field formulation also `background_jones`, `scattered_coefficients` / `scattered_field` (grid), `scattered_power`, `extinction`, `scattering_pattern` (angular dσ/dΩ, or dσ'/dθ for a one-axis PML) (`region="cell" \| "physical"`), see `docs/PML.md` "Observables". |
| `FieldObservables` | Field-level API (`intensity`, `absorption_map`) — **stub**, raises `NotImplementedError` |

`src/metarcwa/_dtypes.py` — `to_complex` / `to_real` / `_REAL_TO_COMPLEX`, shared by `model` and
`solver` (lives at package root so `solver` never imports `model` utils).

---

## Conventions

- **Language / deps**: Python 3.12+, `torch>=2.1`, `pyyaml`. No numpy/scipy/jax.
- **Dev deps**: `ipykernel`, `matplotlib`, `pytest>=9.0.3`.
- **Docstrings**: NumPy-style (Parameters / Returns / Attributes sections).
- **Type hints**: modern union syntax (`A | B`, `dict[str, ...]`).
- **Config objects**: `@dataclass` with `__post_init__` validation, `to_dict`/`from_dict`,
  `to_yaml`/`from_yaml` round-trip (`Config`, `Factorization`).
- **Module header**: file-path comment at top of each module (`# metarcwa/solver/config.py`).
- **Naming**: `snake_case` functions/variables, `PascalCase` classes.
- **Tensors**: all ops batched over wavelength/angle; everything is differentiable by design.

---

## Domain concepts (must-know)

- **Harmonics**: Bloch–Fourier expansion over `(m,n)` pairs, truncated to `N_h` harmonics
  (`circular` ellipse or `rectangular` grid; set in `Config.nx/ny` + `Config.m/n`).
- **Field vector**: `ψ = (Sx, Sy, Ux, Uy)` transverse components; mode matrices `W` (E) and
  `V` (H). For homogeneous layers `W=I` (implicit).
- **Two interchangeable patterned-layer paths**, selected by `Config.modesolver`:
  - `"eig"` (default) — `compute_isotropic` → `eigsolver` → `ModalOperator(lam, W, V, thickness)`.
  - `"matexp"` — same `compute_isotropic` `P`/`Q`, no eigendecomposition — `TransferOperator`
    (see `layersolver/matexpsolver.py`, `docs/matrixexp.md`).
  - `homogeneous_modes` (closed form, no eig) also produces a `ModalOperator`, for
    `HomogeneousLayer`/`MediumSpec` regardless of `modesolver` (which only affects `_patterned`).
  - `LayerOperator` itself is a structural `Protocol` (`operator.py`), not one dataclass — both
    `ModalOperator` and `TransferOperator` satisfy it (`.smatrix()`, `.transfer()`, `thickness`).
    `thickness=None` marks a semi-infinite medium (boundary only).
- **S-matrix composition**: Redheffer star product `Block2x2.star()`. **Not matrix multiply.**
  Associative, not commutative. Convention: `S11/S22` = reflection, `S12/S21` = transmission.
- **Lateral PML** (`docs/PML.md`): complex coordinate stretch at the edges of a rectangular supercell for
  aperiodic problems; `K → K̃` (dense), Bloch phase 0 along PML axes, `modesolver` defaults to `"matexp"`
  (eig vectors ill-conditioned in large cells), uniform media solved numerically (`P = Q0/ε` + `eigsolver`).
  Illumination: `Config.pml_formulation="scattered"` (default) splits `E = E_inc + E_s`; `run()` solves the
  background (`background.py`), builds per-patterned-layer source vectors (`scattered.py`) and composes
  `AffineS`; `ModalSolution` gains `e_L`, `e_R` (`[..., 2Nh, 2]`, s/p columns) and `background`.
  `"total"` only without tilt along a PML axis (validation).
- **TVF / FFF**: Tangent Vector Field gives the anisotropic Fourier-space factorization of
  `D`–`E` at discontinuities (Li 1996 rules). Treated as fixed geometry; detached from autograd.
- **`Block`/`Block2x2`**: lazy structured matrices (SCALAR → DIAG → DENSE promotion only).
  Avoids allocating dense `N_h × N_h` arrays until forced. Prefer `.solve(rhs)` over `.inv()@rhs`.

---

## Gotchas / non-obvious notes

**Conventions that bite:**
- `exp(−jωt)` time convention throughout. Branch selection for `λ = 1j·kz`:
  propagating → `Re(kz) > 0`; evanescent → `Im(kz) > 0` (decaying +z). Getting this wrong
  silently gives growing evanescent modes. Shared by both mode solvers via
  `layersolver/_modes.py::_branch_select` (classifies on `Re(λ)`, tolerance-based).
- **All wavevectors are k0-normalized** (dimensionless). `kx0`/`ky0` in `Source` are k0-normalized;
  multiply by `k0` only for physical units.
- **Axis ↔ lattice mapping** (easy to flip): eps grid shape is `[…, Ny, Nx]`; axis `−1` = Nx = a1
  direction = `m`/`Gx`; axis `−2` = Ny = a2 direction = `n`/`Gy`. `Config.nx` = a1 resolution,
  `Config.ny` = a2 resolution (`harmonics.harmonic_wavevectors`).

**Numerical stability:**
- **Grazing incidence** (`kx² + ky² = ε` exactly → `λ = 0`, e.g. TIR critical angle): handled by
  `Config.grazing_eps_reg` (default `1e-8`), a tiny imaginary part added to every permittivity in
  mode solving (`_modes._regularize_eps`). Setting it to `0` can make the boundary solve singular.
  The grazing `RuntimeWarning` (`_modes._warn_grazing`) is opt-in via `_modes.WARN_GRAZING`
  (it forces a GPU sync).
- **Lorentzian regularization** of `1/λ` (`_modes._lam_inv_block`, `δ=1e-30`) and custom `Eig`
  autograd in `eigsolver.py` (`broadening_parameter=1e-10`) give finite gradients near degeneracy
  at the cost of a small controlled error.
- `eigsolver_stable=False` is faster but **can produce NaN gradients**. `checkpoint_eig=True`
  trades one extra eig per patterned layer in backward for lower peak memory.
- Default dtype is **float32** — use `float64` for accuracy-sensitive runs.
- **dtype/device of `shape_fn` output**: patterns are created fresh in `Layer.spec()`, so
  `Model.to()` can't cast them; `LayerSolver._patterned` casts `pattern` to `config.dtype` /
  solver device. Keep that cast — without it a float32 mask under float64 + TVF crashes with a
  complex64/complex128 matmul mismatch.

**TVF-specific:**
- Fourier loss **must** divide raw `fft2` output by `D0·D1`; omitting this makes the penalty
  ~`(D0·D1)²` too large and collapses the field to DC (`tvf.md`, `tvf/tvf_utils.py`).
- Use **forward differences**, not central — central differences have a checkerboard null space
  (`tvf.md` "Forward-difference", `_grad_forward_periodic`).
- Newton optimizer default `steps=1` is the **exact global minimum** (loss is quadratic in Fourier
  coeffs) — do not add steps.

**Setup quirks:**
- `medium_void` and `medium_solid` must be the **same `Medium` subclass**; mixing
  isotropic/anisotropic is unsupported (checked in `Layer.__init__`). Only `IsotropicMedium`
  exists today.
- **Wrap callables in `CallableModule`** (`model/nn_helpers.py`) — `eps_fn` / `shape_fn` must be
  `nn.Module`s so their tensors/parameters move with `Model.to()` and show up in
  `model.parameters()`. `register()` only accepts float32/float64 values.
- Reciprocal lattice uses an **explicit 2D cross-product formula** (not `linalg.solve`) — keep
  consistent if editing lattice math (commit `735639b`).
- Both `modesolver="eig"` and `"matexp"` are implemented (`docs/matrixexp.md`). Under `"matexp"`
  the `Solver.__init__`-expensive / `run()`-cheap split **inverts**: `prepare()` skips the
  eigendecomposition (cheap), and the sliced `matrix_exp` moves into `smatrix()`/`run()` instead,
  since it depends on `thickness` and must stay responsive to
  `dataclasses.replace(op, thickness=...)`.
- GPU eigendecomposition is a known perf bottleneck (synchronizes CPU↔GPU) [verify] —
  `modesolver="matexp"` avoids it entirely (no `eig` call), at the cost of the slicing overhead
  above and ~4x the dense memory per patterned layer (`4Nh×4Nh` system matrix vs `2Nh×2Nh` for `Ω²`).
- Historically bug-prone areas: sign conventions, TVF scaling, homogeneous layer shapes.

**PML-specific:**
- Under the scattered formulation the structure (`ε − ε_void`) must stay out of the PML (`prepare()` warns).
- Patterned-layer operators carry `op.source` (`SourceData`: `d_eps`, `eps_conv`, `qfact`, `phi_lu`) only
  under the scattered formulation; `run()` builds the source vectors from it. With TVF the source carries
  `Qfact = Q − Q0` applied to `E_inc`; without it scattered ≠ total at finite `Nh`. Keep it when touching
  `compute_isotropic` (`return_parts=True`).
- The background solve takes layer thicknesses from the prepared operators (not `model_spec`), so
  `dataclasses.replace(op, thickness=...)` stays consistent.
- Source phase convention: pixel `i` at `x = i·px/nx` (FFT convention of `convolution_matrix`); the
  background/scattered fields share it.

---

## Detailed docs (`docs/`)

| File | Read for |
|------|----------|
| `rcwa_core.md` | **START HERE** — Maxwell → coupled first-order system; `P`/`Q` operators; where TVF correction enters |
| `factorization.md` | Li (1996) Laurent vs inverse rule; FFF tensor `[[A_ij]]`; constraint `[[A_xx]]+[[A_yy]]=I` |
| `tvf.md` | TVF construction (Jones/Pol/Normal/Jones_direct); Newton optimizer; all loss terms; "Numerical Choices" table |
| `eigenproblem.md` | Patterned-layer eig path; custom stable-gradient `Eig` autograd class; branch selection |
| `homogeneous.md` | Closed-form modes; `λ²=kx²+ky²−ε`; Lorentzian sqrt regularization; grazing incidence |
| `smatrix.md` | S- vs T-matrix (stability rationale); boundary/layer/Redheffer-star derivations; impl notes |
| `blockmatrix.md` | `Block`/`Block2x2` system (SCALAR/DIAG/DENSE); memory savings; Schur-complement inverse |
| `PML.md` | Lateral PML: coordinate stretching, `K̃`, factorization, scattered-field formulation, affine S-matrices, PML observables |
| `matrixexp.md` | Matrix-exponential patterned-layer solver (`modesolver="matexp"`) — system matrix, T→S conversion, slicing for stability, cost-model inversion |
