# tests/observables/test_observables.py
# Physics regression tests for Observables: s/p-basis reflection/
# transmission Jones matrices, cross-checked against independently
# computed Fresnel formulas for a single dielectric interface.

import math

import pytest
import torch

from metarcwa.model.base import Model
from metarcwa.model.stack import Stack
from metarcwa.model.medium import IsotropicMedium
from metarcwa.model.lattice import Lattice
from metarcwa.model.source import Source
from metarcwa.model.nn_helpers import CallableModule
from metarcwa.solver.base import Solver
from metarcwa.solver.config import Config
from metarcwa.observables import Observables

# Numerical-agreement tolerance vs. independently-computed Fresnel formulas.
# Matches the project's own established cross-check tolerance for "should
# match a reference" comparisons (see tests/block_structure_bench.py's
# correctness check, tol_atol=tol_rtol=1e-6) -- residuals at the ~1e-8..1e-9
# level were confirmed during development to be a base-solver characteristic
# (present even when Sxx/Sxy/Syx/Syy are read directly and the s/p rotation
# is done by hand with plain torch.linalg.solve, bypassing Observables and
# Block2x2 entirely), not an artifact of the rotation implementation itself.
ATOL = 1e-6


def _const_eps(val: complex):
    """Return a CallableModule that returns a constant complex eps."""
    return CallableModule(lambda wvl: torch.full_like(wvl, val, dtype=torch.complex128))


def _make_interface(eps1: complex, eps2: complex, theta: float, phi: float) -> Model:
    """A single dielectric interface: no finite layers, just an incidence/
    transmission boundary. theta/phi in radians."""
    incidence = IsotropicMedium(_const_eps(eps1))
    transmission = IsotropicMedium(_const_eps(eps2))
    lattice = Lattice.rectangular(1.0, 1.0)
    stack = Stack(incidence, [], transmission, lattice)
    source = Source(wavelength=1.0, theta=theta, phi=phi)
    return Model(stack, source).to(dtype=torch.float64, device="cpu")


def _fresnel_kz(eps1: float, eps2: float, theta: float, phi: float):
    """Independently compute kz1, kz2, and the standard s/p Fresnel
    reflection coefficients for a single isotropic interface, using plain
    Python complex arithmetic -- no solver internals involved."""
    n1 = math.sqrt(eps1)
    kx0 = n1 * math.sin(theta) * math.cos(phi)
    ky0 = n1 * math.sin(theta) * math.sin(phi)
    kz1 = complex(eps1 - kx0**2 - ky0**2) ** 0.5
    kz2 = complex(eps2 - kx0**2 - ky0**2) ** 0.5
    r_s = (kz1 - kz2) / (kz1 + kz2)
    r_p = (eps2 * kz1 - eps1 * kz2) / (eps2 * kz1 + eps1 * kz2)
    return kz1, kz2, r_s, r_p


class TestReflectionFresnelAgreement:

    def test_normal_incidence_matches_fresnel(self):
        """At normal incidence: rs (co-pol, s-excited) matches the scalar
        Fresnel r=(n1-n2)/(n1+n2); rp (co-pol, p-excited) matches -r (the
        well-known sign convention under p=k x s "physical transport" basis
        transport -- k flips on reflection, s doesn't, so p's tangential
        projection flips sign even though |rp|=|rs|); both cross-pol terms
        are ~0. Exercises the rho->0 (kx=ky=0) regularization path in
        Observables._rotation. This sign convention was confirmed as the
        intended one (not a bug) during development."""
        eps1, eps2 = 1.0 + 0j, 2.5 + 0j
        model = _make_interface(eps1, eps2, theta=0.0, phi=0.0)
        solver = Solver(model, Config(m=1, n=1, factorization=None, dtype=torch.float64))
        obs = Observables(solver.run())

        n1, n2 = math.sqrt(eps1.real), math.sqrt(eps2.real)
        r_analytic = (n1 - n2) / (n1 + n2)

        rs_s, rp_s = obs.reflection("s")   # excite pure s
        rs_p, rp_p = obs.reflection("p")   # excite pure p

        assert rs_s.item() == pytest.approx(r_analytic, abs=ATOL)      # co-pol
        assert rp_p.item() == pytest.approx(-r_analytic, abs=ATOL)     # co-pol, sign-flipped
        assert rp_s.item() == pytest.approx(0.0, abs=ATOL)             # cross-pol
        assert rs_p.item() == pytest.approx(0.0, abs=ATOL)             # cross-pol
        assert abs(rs_s.item()) == pytest.approx(abs(rp_p.item()), abs=ATOL)  # |rs|=|rp|

    def test_oblique_incidence_matches_fresnel(self):
        """At oblique incidence, exciting with pure s (or p) must reproduce
        the standard Fresnel r_s (or r_p), with the cross-polarized
        component ~0 (isotropic, unpatterned stack -> no depolarization)."""
        eps1, eps2 = 1.0 + 0j, 2.5 + 0j
        theta, phi = math.radians(30.0), math.radians(20.0)
        model = _make_interface(eps1, eps2, theta, phi)
        solver = Solver(
            model, Config(m=2, n=2, factorization=None, truncation="rectangular", dtype=torch.float64)
        )
        obs = Observables(solver.run())

        _, _, r_s, r_p = _fresnel_kz(eps1.real, eps2.real, theta, phi)

        rs_s, rp_s = obs.reflection("s")   # excite pure s
        rs_p, rp_p = obs.reflection("p")   # excite pure p

        assert rs_s.item() == pytest.approx(r_s, abs=ATOL)     # co-pol
        assert rp_s.item() == pytest.approx(0.0, abs=ATOL)     # cross-pol
        assert rp_p.item() == pytest.approx(r_p, abs=ATOL)     # co-pol
        assert rs_p.item() == pytest.approx(0.0, abs=ATOL)     # cross-pol


class TestEnergyConservation:

    def test_lossless_interface_r_plus_t_equals_one(self):
        """For a lossless interface, R+T=1 for each incident polarization.
        R = |rs|^2+|rp|^2 (reflection stays in the incidence medium, no kz
        ratio); T = (Re(kz2)/Re(kz1))*(|ts|^2+|tp|^2) -- the SAME kz-ratio
        applies uniformly to both s and p because Observables' rotation is
        built from genuinely unit-norm 3D s/p vectors (not a convention
        that folds an eps ratio into one polarization's transmission)."""
        eps1, eps2 = 1.0 + 0j, 2.5 + 0j
        theta, phi = math.radians(30.0), math.radians(20.0)
        model = _make_interface(eps1, eps2, theta, phi)
        solver = Solver(
            model, Config(m=2, n=2, factorization=None, truncation="rectangular", dtype=torch.float64)
        )
        sol = solver.run()
        obs = Observables(sol)

        ls = sol.prepared.layersolver
        ops = sol.prepared.ops
        i0 = ((ls.m_flat == 0) & (ls.n_flat == 0)).nonzero(as_tuple=True)[0].item()
        kz1 = (ops[0].lam[..., i0] / 1j).reshape(()).real.item()
        kz2 = (ops[-1].lam[..., i0] / 1j).reshape(()).real.item()
        kz_ratio = kz2 / kz1

        for pol in ("s", "p"):
            rs, rp = obs.reflection(pol)
            ts, tp = obs.transmission(pol)
            R = rs.abs().item() ** 2 + rp.abs().item() ** 2
            T = kz_ratio * (ts.abs().item() ** 2 + tp.abs().item() ** 2)
            assert R + T == pytest.approx(1.0, abs=ATOL)
