# tests/solver/test_pml_validation.py
# Physical validation of the lateral PML at normal incidence (milestone M1):
# specular observables of a uniform stack, and slab-waveguide effective
# indices against the analytic dispersion relation. See docs/PML.md.

import math

import pytest
import torch
from torch.testing import assert_close

from metarcwa import (Config, Factorization, IsotropicMedium, Lattice, Layer, Model,
                      Observables, PML, Solver, Source, Stack)
from metarcwa.model.nn_helpers import CallableModule
from metarcwa.solver.base import build_layersolver

from tests.solver.test_pml_uniform import N_SUB, WVL, _tmm_normal, _uniform_model
from tests.solver.test_pml_stretch import _config


# ---------------------------------------------------------------------------
# Observables under PML
# ---------------------------------------------------------------------------

class TestObservables:

    @pytest.mark.parametrize("tilt, pml", [
        ({}, PML(width=0.25)),
        (dict(theta=0.4, phi=torch.pi / 2), PML(width=(0.25, 0.0))),
    ])
    @pytest.mark.parametrize("kind", ["reflection", "transmission"])
    def test_specular_jones_matches_periodic(self, tilt, pml, kind):
        """Uniform stack: the specular order is an exact mode under PML, so
        its s/p Jones matrix equals the periodic one (normal incidence, and
        tilt along a non-PML axis)."""
        cfg = _config(m=4, n=4, nx=32, ny=32)
        J0 = Observables(Solver(_uniform_model(None, **tilt), cfg).run()).jones(kind)
        J1 = Observables(Solver(_uniform_model(pml, **tilt), cfg).run()).jones(kind)
        assert_close(J1, J0, atol=1e-12, rtol=0)

    def test_specular_power_matches_tmm(self):
        obs = Observables(Solver(_uniform_model(PML(width=0.25)), _config(m=4, n=4, nx=32, ny=32)).run())
        rs, _ = obs.reflection("s")
        ts, _ = obs.transmission("s")
        R, T = _tmm_normal(WVL)
        assert abs(abs(rs.reshape(-1)[0]) ** 2 - R) < 1e-6
        assert abs(N_SUB * abs(ts.reshape(-1)[0]) ** 2 - T) < 1e-6

    def test_non_specular_order_raises(self):
        obs = Observables(Solver(_uniform_model(PML(width=0.25)), _config(m=2, n=2, nx=16, ny=16)).run())
        with pytest.raises(ValueError, match="specular order"):
            obs.jones("reflection", out_order=(1, 0))


# ---------------------------------------------------------------------------
# Large PML cells: the default (matexp) solver stays stable
# ---------------------------------------------------------------------------

def _ridge_C(L, M):
    """L * (t00 - t00_bg) for an isolated ridge (width 0.5, height 0.3,
    index 2) on a substrate (index 1.5), uniform along y, PML along x, at
    normal incidence. Independent of L for an isolated ridge, up to the
    slow convergence of the cell-averaged observable."""
    def medium(eps):
        return IsotropicMedium(CallableModule(lambda wl: torch.full_like(wl, eps, dtype=torch.complex128)))

    def t00(ridge):
        def stripe(lat, nx, ny):
            x = ((torch.arange(nx, dtype=torch.float64) + 0.5) / nx - 0.5) * L
            return ((x.abs() < 0.25) & ridge).double()[None, :].expand(ny, nx)
        cfg = Config(dtype=torch.float64, nx=128 * L, ny=4, m=M, n=0,
                     truncation="rectangular", factorization=Factorization())
        layer = Layer(medium_solid=medium(4.0), medium_void=medium(1.0),
                      shape_fn=CallableModule(stripe), thickness=0.3)
        stack = Stack(incidence=medium(1.0), layers=[layer], transmission=medium(2.25),
                      lattice=Lattice.rectangular(L, 1.0), pml=PML(width=(2.0, 0.0)))
        ts, _ = Observables(Solver(Model(stack, Source(wavelength=1.0)), cfg).run()).transmission("s")
        return ts.reshape(-1)[0]

    return L * (t00(True) - t00(False))


def test_large_pml_cell_stable_with_default_solver():
    """With eig, a 16-wavelength PML cell gives erratic results (patterned-
    layer eigenvectors become numerically singular; here an error ~6e-3);
    the default matexp solver agrees with an 8-wavelength cell."""
    assert abs(_ridge_C(16, 128) - _ridge_C(8, 64)) < 2e-3


# ---------------------------------------------------------------------------
# Slab waveguide: patterned-layer modes vs the analytic dispersion relation
# ---------------------------------------------------------------------------

def _slab_neff_analytic(n1, n2, d, wvl, pol):
    """Fundamental (even) mode of a symmetric slab, by bisection.

    TE: kappa tan(kappa d/2) = gamma;  TM: kappa tan(kappa d/2) = (n1/n2)^2 gamma.
    """
    k0 = 2 * math.pi / wvl
    ratio = 1.0 if pol == "TE" else (n1 / n2) ** 2

    def f(n):
        kappa, gamma = k0 * math.sqrt(n1**2 - n**2), k0 * math.sqrt(n**2 - n2**2)
        return kappa * math.tan(kappa * d / 2) - ratio * gamma

    lo = max(n2, math.sqrt(max(n1**2 - (math.pi / (k0 * d)) ** 2, 0.0))) + 1e-12
    hi = n1 - 1e-12
    for _ in range(200):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if f(mid) > 0 else (lo, mid)
    return (lo + hi) / 2


def _slab_kz(n1, n2, d, lx, M, pml, wvl=1.0):
    """Modal kz (= lam / j) of a z-invariant slab layer, uniform along y.

    The core is rasterized exactly (pixel 0.00625 divides d), so the result
    is limited by harmonic truncation, not by staircasing.
    """
    def medium(eps):
        return IsotropicMedium(CallableModule(lambda wl: torch.full_like(wl, eps, dtype=torch.complex128)))

    def stripe(lat, nx, ny):
        x = ((torch.arange(nx, dtype=torch.float64) + 0.5) / nx - 0.5) * lx
        return (x.abs() < d / 2).double()[None, :].expand(ny, nx)

    cfg = Config(dtype=torch.float64, nx=round(lx / 0.00625), ny=4, m=M, n=0,
                 truncation="rectangular", factorization=Factorization(),
                 modesolver="eig")                 # the layer eigenvalues are the waveguide modes
    layer = Layer(medium_solid=medium(n1**2), medium_void=medium(n2**2),
                  shape_fn=CallableModule(stripe), thickness=1.0)
    stack = Stack(incidence=medium(1.0), layers=[layer], transmission=medium(1.0),
                  lattice=Lattice.rectangular(lx, 1.0), pml=pml)
    spec = Model(stack, Source(wavelength=wvl)).to(torch.float64).spec(cfg.nx, cfg.ny)
    return build_layersolver(spec, cfg).prepare(spec.layers[0]).lam.reshape(-1) / 1j


def _guided(kz, n1, n2, k):
    """The k largest-index guided modes (n2 < Re kz < n1, nearly lossless)."""
    g = kz[(kz.real > n2) & (kz.real < n1) & (kz.imag.abs() < 1e-3)]
    return g[g.real.argsort(descending=True)][:k]


class TestSlabWaveguide:

    def test_strong_guide_te_tm_match_analytic(self):
        """Strongly confined slab (n 2.0 / 1.45, d 0.4): TE0 and TM0 effective
        indices match the analytic dispersion relation."""
        n1, n2, d = 2.0, 1.45, 0.4
        kz = _slab_kz(n1, n2, d, lx=4.0, M=40, pml=PML(width=(1.0, 0.0)))
        te, tm = _guided(kz, n1, n2, 2)
        assert abs(te.real - _slab_neff_analytic(n1, n2, d, 1.0, "TE")) < 2e-5
        assert abs(tm.real - _slab_neff_analytic(n1, n2, d, 1.0, "TM")) < 2e-5

    def test_pml_isolates_weak_guide_in_small_cell(self):
        """Weakly guided slab (n 1.6 / 1.45, d 0.3, decay length 0.45) in a
        cell of 3: the periodic array is biased by coupling between copies,
        while the PML gives the isolated waveguide."""
        n1, n2, d = 1.6, 1.45, 0.3
        exact = _slab_neff_analytic(n1, n2, d, 1.0, "TE")
        te_pml = _guided(_slab_kz(n1, n2, d, lx=3.0, M=20, pml=PML(width=(0.75, 0.0))), n1, n2, 1)[0]
        te_per = _guided(_slab_kz(n1, n2, d, lx=3.0, M=20, pml=None), n1, n2, 1)[0]
        assert abs(te_pml.real - exact) < 2e-5
        assert abs(te_per.real - exact) > 1e-4
        assert 0 < te_pml.imag < 1e-4          # small artificial PML loss
