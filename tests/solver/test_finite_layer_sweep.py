# tests/solver/test_finite_layer_sweep.py
# Regression tests for batched wavelength sweeps through a finite layer.
#
# Neither of these shape bugs was exercised by the existing suite: every
# existing finite-layer fixture uses a scalar/1-element wavelength, and every
# multi-wavelength fixture uses an empty layer list (bare interface, boundary
# S-matrix only, no S_prop / no eigensolver dispatch). Sweeping N_wl>1 through
# a *finite* layer hits both bugs:
#   1. S_prop (smatrix.py) broadcasting `lam` [N_wl, N_theta, N_phi, 2Nh]
#      against a bare 1-D `wvl` [N_wl].
#   2. compute_isotropic (layersolver/isotropic.py) broadcasting a 3-D
#      epsilon_conv [N_wl, Nh, Nh] against 5-D Kx/Ky (eigensolver path only).

import pytest
import torch

from metarcwa.model.base import Model
from metarcwa.model.stack import Stack
from metarcwa.model.layer import Layer
from metarcwa.model.medium import IsotropicMedium
from metarcwa.model.lattice import Lattice
from metarcwa.model.source import Source
from metarcwa.model.utils import CallableModule
from metarcwa.solver.base import Solver
from metarcwa.solver.config import Config


def _const_eps(val: complex):
    """Return a CallableModule that returns a constant complex eps."""
    return CallableModule(lambda wvl: torch.full_like(wvl, val, dtype=torch.complex128))


def _uniform_mask():
    """All-ones shape_fn: forces the eigensolver path on a physically-uniform
    layer (same medium as solid and void everywhere)."""
    return CallableModule(lambda lattice, nx, ny: torch.ones((ny, nx)))


def _make_model(method: str, wavelength: torch.Tensor) -> Model:
    """One ε=2.5 slab between ε=1 (incidence) and ε=2.25 (transmission),
    swept over `wavelength`.

    method="homogeneous": no shape_fn -> closed-form homogeneous_modes.
    method="eigen": all-ones shape_fn + matching medium_void -> same physical
    layer, routed through compute_isotropic + eigsolver instead.
    """
    incidence    = IsotropicMedium(_const_eps(1.0 + 0j))
    transmission = IsotropicMedium(_const_eps(2.25 + 0j))
    solid        = IsotropicMedium(_const_eps(2.5 + 0j))
    if method == "homogeneous":
        layer = Layer(solid, thickness=0.3)
    elif method == "eigen":
        void  = IsotropicMedium(_const_eps(2.5 + 0j))
        layer = Layer(solid, thickness=0.3, medium_void=void, shape_fn=_uniform_mask())
    else:
        raise ValueError(method)
    lattice = Lattice.rectangular(1.0, 1.0)
    stack   = Stack(incidence, [layer], transmission, lattice)
    source  = Source(wavelength=wavelength)
    return Model(stack, source).to(dtype=torch.float64)


class TestWavelengthSweepThroughFiniteLayer:

    def test_homogeneous_path_runs_and_is_wavelength_dependent(self):
        """Regression for the S_prop wvl/lam broadcast bug: a multi-point
        wavelength sweep through a finite homogeneous layer must run without
        error and must not collapse to a single (wavelength-independent)
        value -- that would indicate wvl silently broadcast wrong."""
        wl = torch.linspace(0.5, 1.5, 6, dtype=torch.float64)
        solver = Solver(_make_model("homogeneous", wl), Config(m=1, n=1, dtype=torch.float64))
        sol = solver.run()
        Nh = solver.layersolver.m_flat.shape[0]
        S11 = sol.S11.to_dense(Nh) if hasattr(sol.S11, "a") else sol.S11.to(sol.S11.DENSE, Nh).data
        assert S11.shape[0] == 6
        assert not torch.isnan(S11).any()
        r00 = S11[:, 0, 0]
        assert r00.abs().std().item() > 1e-8, "reflection is suspiciously wavelength-independent"

    def test_homogeneous_and_eigen_paths_agree(self):
        """Regression for the compute_isotropic epsilon_grid/kx broadcast
        bug: a physically-uniform layer solved via the closed-form
        homogeneous path and via the eigensolver (forced with an all-ones
        mask) must give the same S-matrix for every wavelength in the sweep,
        not just the diagonal of a spurious [N_wl, N_wl] outer product."""
        wl = torch.linspace(0.5, 1.5, 6, dtype=torch.float64)
        cfg = Config(m=1, n=1, dtype=torch.float64, factorization=None)

        solver_h = Solver(_make_model("homogeneous", wl), cfg)
        solver_e = Solver(_make_model("eigen", wl), cfg)
        Nh = solver_h.layersolver.m_flat.shape[0]

        sol_h = solver_h.run()
        sol_e = solver_e.run()

        def dense(entry):
            return entry.to_dense(Nh) if hasattr(entry, "a") else entry.to(entry.DENSE, Nh).data

        S11_h, S11_e = dense(sol_h.S11), dense(sol_e.S11)
        S21_h, S21_e = dense(sol_h.S21), dense(sol_e.S21)

        assert S11_h.shape[0] == 6 and S11_e.shape[0] == 6
        assert torch.allclose(S11_h, S11_e, atol=1e-8)
        assert torch.allclose(S21_h, S21_e, atol=1e-8)

    def test_matexp_matches_homogeneous_and_eigen(self):
        """Same slab as `test_homogeneous_and_eigen_paths_agree`, solved a
        third way: `Config.modesolver="matexp"` (no eigendecomposition --
        sliced matrix exponential instead, see docs/matrixexp.md). Must
        agree with the closed-form homogeneous path across the wavelength
        sweep, same as the eig path does."""
        wl = torch.linspace(0.5, 1.5, 6, dtype=torch.float64)
        cfg_eig    = Config(m=1, n=1, dtype=torch.float64, factorization=None,
                            modesolver="eig")
        cfg_matexp = Config(m=1, n=1, dtype=torch.float64, factorization=None,
                            modesolver="matexp")

        solver_h = Solver(_make_model("homogeneous", wl), cfg_eig)
        solver_m = Solver(_make_model("eigen", wl), cfg_matexp)
        Nh = solver_h.layersolver.m_flat.shape[0]

        sol_h = solver_h.run()
        sol_m = solver_m.run()

        def dense(entry):
            return entry.to_dense(Nh) if hasattr(entry, "a") else entry.to(entry.DENSE, Nh).data

        S11_h, S11_m = dense(sol_h.S11), dense(sol_m.S11)
        S21_h, S21_m = dense(sol_h.S21), dense(sol_m.S21)

        assert S11_m.shape[0] == 6
        assert torch.allclose(S11_h, S11_m, atol=1e-8)
        assert torch.allclose(S21_h, S21_m, atol=1e-8)
