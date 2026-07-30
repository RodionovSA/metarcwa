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
uv sync --extra metashapes   # geometry / rasterization
uv sync --extra dispertorch  # dispersion / material models

# Tests
pytest   # pythonpath=["src"] set in pyproject.toml; suites in tests/

# Interactive usage — root notebooks are the de-facto demo/integration tests
# example.ipynb, model_tests.ipynb, homogeneous_tests.ipynb, test_solver.ipynb, tvf_playground.ipynb

# Config-driven solve
config = Config.from_yaml("config.yaml")
```

> Lint / type-check: none configured [verify]. No CI (.github/ absent) [verify].

---

## Architecture

### Data flow (one-way: model → solver)

```
Model (structure + source)
  └─ .spec(nx, ny) ──► ModelSpec  (immutable snapshot)
       └─ Solver(model, config)
            └─ LayerSolver.prepare(element) ──► LayerOperator  (one per stack element)
                 └─ .solve() / LayerSolver.smatrix(op) ──► Block2x2  (full-stack S-matrix)
```

`Solver.__init__` is the expensive step (under the default `modesolver="eig"`):
it precomputes harmonics, the TVF, and solves the modal eigenproblem for every
stack element (`LayerSolver.prepare`), caching the result as a `LayerOperator`
per element. `Solver.run()` is cheap: pure Redheffer star-product composition
of the cached operators via `LayerSolver.smatrix`, no eigendecomposition.
(`modesolver="matexp"` inverts this split — see Gotchas.) Rebuild the `Solver` whenever the
pattern/geometry changes (already required, since `model.spec()` resolves the
pattern in `__init__`); a `LayerOperator`'s `thickness` is read at `smatrix()`
time, so thickness-only changes don't require rebuilding.

### `src/metarcwa/model/` — problem description

| File | Responsibility |
|------|----------------|
| `base.py` | `Model` + `ModelSpec`: top-level container; `.spec()` produces solver snapshot |
| `stack.py` | `Stack` + `StackSpec`: ordered layer sequence + incidence/transmission media |
| `layer.py` | `Layer`, `HomogeneousLayer`, `PatternedLayer`: per-layer permittivity + thickness |
| `medium.py` | `Medium`, `IsotropicMedium`: material / permittivity definitions for half-spaces |
| `source.py` | `Source`: wavelength, incidence angles (theta/phi). No polarization — s/p amplitudes are applied downstream to the computed S-matrix, in the future `results` layer, not needed for the S-matrix itself |
| `lattice.py` | `Lattice`: 2D Bravais lattice vectors `a1`, `a2` |

### `src/metarcwa/solver/` — numerics

| File/pkg | Responsibility |
|----------|----------------|
| `base.py` | `Solver`: top-level driver; precomputes harmonics + TVF + one `LayerOperator` per stack element; `run()` only star-composes |
| `config.py` | `Config` + `Factorization` dataclasses; grid/truncation/dtype/device/solver switches |
| `harmonics.py` | Harmonic index map `(m,n)`, truncation, in-plane wavevectors `kx`/`ky` |
| `smatrix.py` | `S_boundary`, `S_prop`, `S_layer`: per-interface/layer S-matrix builders |
| `blockmatrix.py` | `Block` / `Block2x2`: structured operator algebra + Redheffer star product |
| `convolution.py` | Fourier convolution matrix helpers |
| `layersolver/base.py` | `LayerSolver`: `.prepare(element)` builds a `LayerOperator` (expensive: TVF/convolution/eigendecomp, unless `modesolver="matexp"` — see gotchas); `.smatrix(op)` delegates to `op.smatrix(background)` (cheap, unless `matexp`); `.solve()` = both; precomputes vacuum modes `W0`/`V0`/`background` |
| `layersolver/operator.py` | `LayerOperator`: structural `Protocol` (`thickness`, `.smatrix()`, `.transfer()`), not one concrete type — matches the `Entry` Protocol pattern in `blockmatrix.py`. `ModalOperator` is the `(lam, W, V, thickness)` implementation used by `eigsolver`/`homogeneous_modes`. `Background` bundles `W0`/`V0`/`wvl`. |
| `layersolver/homogeneous.py` | `homogeneous_modes`: closed-form modes (no eigensolver) |
| `layersolver/isotropic.py` | `compute_isotropic`: builds patterned-layer eigenproblem matrices (`P`, `Q` — shared by both modesolvers) |
| `layersolver/eigsolver.py` | `eigsolver`: numerical eigendecomp with stable autograd gradient |
| `layersolver/matexpsolver.py` | `TransferOperator`: the other `LayerOperator` implementation, for `modesolver="matexp"`. No eigendecomposition — slices the layer, exponentiates `A=[[0,P],[Q,0]]` per slice (`torch.linalg.matrix_exp`), converts to an S-matrix, recombines via `star_power` (`O(log n)` Redheffer star products). See `docs/matrixexp.md`. |
| `tvf/tvf.py` | `TVF`: Tangent Vector Field for Li/FFF factorization |
| `tvf/tvf_utils.py` | TVF field math: periodic gradients, Fourier loss, normalization |
| `tvf/optimizers.py` | `make_optimizer`: Newton + other TVF direction-field optimizers |

`src/metarcwa/results/base.py` — **stub only**, not implemented.

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
- **TVF / FFF**: Tangent Vector Field gives the anisotropic Fourier-space factorization of
  `D`–`E` at discontinuities (Li 1996 rules). Treated as fixed geometry; detached from autograd.
- **`Block`/`Block2x2`**: lazy structured matrices (SCALAR → DIAG → DENSE promotion only).
  Avoids allocating dense `N_h × N_h` arrays until forced. Prefer `.solve(rhs)` over `.inv()@rhs`.

---

## Gotchas / non-obvious notes

**Conventions that bite:**
- `exp(−jωt)` time convention throughout. Branch selection for `λ = 1j·kz`:
  propagating → `Re(kz) > 0`; evanescent → `Im(kz) > 0` (decaying +z). Getting this wrong
  silently gives growing evanescent modes (`homogeneous.py:97–103`, `eigsolver.py:99–101`).
- **All wavevectors are k0-normalized** (dimensionless). `kx0`/`ky0` in `Source` are k0-normalized;
  multiply by `k0` only for physical units.
- **Axis ↔ lattice mapping** (easy to flip): eps grid shape is `[…, Ny, Nx]`; axis `−1` = Nx = a1
  direction = `m`/`Gx`; axis `−2` = Ny = a2 direction = `n`/`Gy`. `Config.nx` = a1 resolution,
  `Config.ny` = a2 resolution (`harmonics.py:95–97`).

**Numerical stability:**
- **Grazing incidence** (`kx² + ky² = ε` exactly → `λ = 0`): column of `V` becomes NaN; emits
  `RuntimeWarning` and is **not** auto-handled — avoid in inputs (`homogeneous.py:193,209–212`).
- **Lorentzian regularization** in `homogeneous.py` (default `δ=1e-30`) and custom `Eig` autograd
  in `eigsolver.py` (`broadening_parameter=1e-10`) give finite gradients near degeneracy at the
  cost of a small controlled error.
- `eigsolver_stable=False` is faster but **can produce NaN gradients** (`config.py:118–122`).
- Default dtype is **float32** — use `float64` for accuracy-sensitive runs (`config.py:91–93`).

**TVF-specific:**
- Fourier loss **must** divide raw `fft2` output by `D0·D1`; omitting this makes the penalty
  ~`(D0·D1)²` too large and collapses the field to DC (`tvf.md`, `tvf_utils.py:341–387`).
- Use **forward differences**, not central — central differences have a checkerboard null space
  (`tvf.md` "Forward-difference", `_grad_forward_periodic`).
- Newton optimizer default `steps=1` is the **exact global minimum** (loss is quadratic in Fourier
  coeffs) — do not add steps.

**Setup quirks:**
- `medium_void` and `medium_solid` must be the **same `Medium` subclass**; mixing
  isotropic/anisotropic is unsupported (`layer.py:45–53`).
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
| `matrixexp.md` | Matrix-exponential patterned-layer solver (`modesolver="matexp"`) — system matrix, T→S conversion, slicing for stability, cost-model inversion |
