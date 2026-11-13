# MetaRCWA

Rigorous Coupled-Wave Analysis (RCWA) for 3D periodic structures and metasurfaces, written in pure PyTorch:
it runs on the GPU, and the whole pipeline (geometry → S-matrix → R/T) is differentiable with autograd, so it can be used for inverse design.

## Features

- **Two patterned-layer solvers** (`Config.modesolver`):
  - `"eig"`: modal eigendecomposition with a stable autograd gradient near degeneracies.
  - `"matexp"`: sliced matrix exponential with no `eig` call. It avoids the GPU eigensolver bottleneck.
- **Closed-form homogeneous layers**: uniform layers skip the eigensolver entirely.
- **Fast Fourier factorization**: Laurent/inverse rules (Li 1996) with Tangent Vector Field (TVF) normal-vector fields (`Jones`, `Pol`, `Normal`, `Jones_direct`).
- **Redheffer star-product S-matrices**: stable for thick and lossy layers. Lazy SCALAR/DIAG/DENSE block algebra avoids dense `N_h × N_h` allocations.
- **Batched sweeps**: wavelength × θ × φ, plus batched geometry.
- **Lateral PML for isolated structures**: `Stack(pml=PML(width=...))` turns a rectangular supercell into an aperiodic problem (single scatterers, finite arrays, waveguides). A scattered-field formulation handles any angle of incidence.
- **Observables**: s/p Jones matrices, reflection/transmission amplitudes per diffraction order. With a PML: specular amplitudes, scattered power, extinction and the angular scattering pattern.
- **Config**: `Config` dataclass with YAML round-trip (`Config.from_yaml("config.yaml")`).

## Installation

```bash
uv add git+https://github.com/RodionovSA/metarcwa.git
# or
pip install git+https://github.com/RodionovSA/metarcwa.git
```

Optional extras:

| Extra | Provides |
|-------|----------|
| `metashapes` | Shape primitives and rasterization (`from_metashapes`) |
| `dispertorch` | Dispersion models / material library (`from_dispertorch`) |
| `ridb` | Refractive-index database |

```bash
uv add "metarcwa[metashapes,dispertorch] @ git+https://github.com/RodionovSA/metarcwa.git"
```

Development:

```bash
git clone https://github.com/RodionovSA/metarcwa.git && cd metarcwa
uv sync --all-extras --dev
pytest
```

Requires Python 3.12+ and PyTorch 2.1+.

## Quickstart

A silicon disk array (400 nm period, 100 nm thick) on glass, at normal incidence:

```python
import torch
from metarcwa import (Model, Stack, Layer, Lattice, Source, IsotropicMedium,
                      Solver, Config, Factorization, Observables)
from metarcwa.model.nn_helpers import CallableModule

const = lambda eps: CallableModule(lambda wl: torch.full_like(wl, eps))
air, si, glass = (IsotropicMedium(const(e)) for e in (1.0, 3.8**2, 1.46**2))

def disk(lattice, nx, ny):                      # pattern in [0, 1], shape [ny, nx]
    y, x = torch.meshgrid(torch.linspace(-.5, .5, ny), torch.linspace(-.5, .5, nx), indexing="ij")
    return (x**2 + y**2 < 0.3**2).float()

lattice = Lattice.rectangular(px=400.0, py=400.0)              # nm
layer   = Layer(si, 100.0, medium_void=air, shape_fn=CallableModule(disk))
model   = Model(Stack(air, [layer], glass, lattice),
                Source(torch.linspace(500., 900., 41), theta=0.0, phi=0.0))

config = Config(dtype=torch.float32, device="cpu", nx=128, ny=128, m=8, n=8,
                factorization=Factorization(method="Jones"))
obs = Observables(Solver(model, config).run())
rs, rp = obs.reflection(pol="s")                # (co, cross) amplitudes, order (0, 0)
R = rs.abs()**2                                 # [N_wl, N_theta, N_phi]
```

`Solver(...)` does the expensive work: it builds the harmonics and the TVF and solves each layer's modes.
`run()` only composes the cached per-layer S-matrices, so a thickness-only change does not need a new `Solver`.
With `modesolver="matexp"` this split is reversed: setup is cheap and `run()` does the work.

## Examples & docs

- `examples/`: uniform layer and interface (validated against TMM/Fresnel), 1D/2D gratings, factorization and solver comparisons, convergence, PML for isolated structures (normal and oblique incidence).
- `docs/`: theory and implementation notes. Start with `rcwa_core.md`, then `factorization.md`, `tvf.md`, `eigenproblem.md`, `matrixexp.md`, `smatrix.md`, `blockmatrix.md`, `PML.md`.

Conventions: `exp(−jωt)` time dependence; all wavevectors are k0-normalized; the permittivity grid is `[…, Ny, Nx]`, with `Nx` along `a1`.

## License

MIT
