# tests/solver/test_pml_scattered.py
# PML scattered-field formulation (PLAN.md phases 5-8): background solver,
# source vectors / affine S-matrices, PML observables. Small 2D problems
# (uniform along y, PML along x) so the whole file runs in seconds.
# See docs/PML.md.

import dataclasses
import math
import warnings

import pytest
import torch
from torch.testing import assert_close

from metarcwa import (Config, Factorization, IsotropicMedium, Lattice, Layer, Model,
                      Observables, PML, Solver, Source, Stack)
from metarcwa.model.nn_helpers import CallableModule
from metarcwa.solver.background import solve_background
from metarcwa.solver.blockmatrix import Block, Block2x2, apply_vec, solve_vec
from metarcwa.solver.layersolver.base import LayerSolver
from metarcwa.solver.polarization import sp_matrix
from metarcwa.solver.scattered import exp_integral, phi1
from metarcwa.solver.smatrix import AffineS

DT = torch.float64
CDT = torch.complex128
L, WVL = 2.0, 0.8


def _medium(eps):
    return IsotropicMedium(CallableModule(lambda wl: torch.full_like(wl, eps, dtype=CDT)))


def _ridge(x0=0.0, soft=None):
    def shape_fn(lattice, nx, ny):
        x = ((torch.arange(nx, dtype=DT) + 0.5) / nx - 0.5) * L
        if soft is None:
            p = ((x - x0).abs() < 0.25).to(DT)
        else:
            p = torch.sigmoid((0.25 - (x - x0).abs()) / soft)
        return p[None, :].expand(ny, nx)
    return CallableModule(shape_fn)


def _model(theta=0.0, phi=0.0, pml=True, thickness=0.3, shape=None, eps_t=2.25, film=True):
    layers = [Layer(medium_solid=_medium(4.0), medium_void=_medium(1.0),
                    shape_fn=shape or _ridge(), thickness=thickness)]
    if film:
        layers.append(Layer(medium_solid=_medium(2.0), thickness=0.1))
    stack = Stack(incidence=_medium(1.0), layers=layers, transmission=_medium(eps_t),
                  lattice=Lattice.rectangular(L, 0.5),
                  pml=PML(width=(0.5, 0.0)) if pml else None)
    # float64 angles: a Python float is stored in the default dtype (float32), and
    # cos(pi/2) would then leave kx0 ~ 1e-8 instead of round-off.
    as64 = lambda v: torch.as_tensor(v, dtype=DT)
    return Model(stack, Source(wavelength=as64(WVL), theta=as64(theta), phi=as64(phi)))


def _config(**kw):
    base = dict(dtype=DT, nx=96, ny=4, m=10, n=0, truncation="rectangular", factorization=None)
    base.update(kw)
    return Config(**base)


def _obs(model, **kw):
    return Observables(Solver(model, _config(**kw)).run())


# ---------------------------------------------------------------------------
# Background solver
# ---------------------------------------------------------------------------

class TestBackground:

    def test_fresnel_interface(self):
        theta, n2 = 0.5, 1.5
        stack = Stack(incidence=_medium(1.0), layers=[], transmission=_medium(n2 ** 2),
                      lattice=Lattice.rectangular(L, 0.5))
        spec = Model(stack, Source(wavelength=torch.tensor(WVL, dtype=DT), theta=theta)).spec(4, 4)
        bg = solve_background(spec)
        c1, c2 = math.cos(theta), math.sqrt(1 - (math.sin(theta) / n2) ** 2)
        R = sp_matrix(spec.kx0, spec.ky0, -bg.lam[0] / 1j, torch.ones(1, 1, 1, dtype=CDT),
                      torch.atan2(spec.ky0, spec.kx0))
        J = torch.linalg.solve(R, bg.c_minus[0]).reshape(2, 2)
        assert_close(J[0, 0].real, torch.tensor((c1 - n2 * c2) / (c1 + n2 * c2), dtype=DT))
        assert_close(J[1, 1].real.abs(), torch.tensor(abs((n2 * c1 - c2) / (n2 * c1 + c2)), dtype=DT))

    def test_matches_periodic_solver_for_uniform_stack(self):
        """Background Jones matrices = the main solver's for a homogeneous stack."""
        layers = [Layer(medium_solid=_medium(e), thickness=d) for e, d in ((4.0, 0.13), (2.0 + 0.3j, 0.27))]
        stack = Stack(incidence=_medium(1.44), layers=layers, transmission=_medium(2.25),
                      lattice=Lattice.rectangular(L, 0.5))
        model = Model(stack, Source(wavelength=torch.tensor([0.6, 0.8], dtype=DT),
                                    theta=torch.tensor([0.0, 0.4, 1.2]), phi=torch.tensor([0.0, 0.7])))
        cfg = _config(nx=16, m=1)
        obs = Observables(Solver(model, cfg).run())
        spec = model.spec(cfg.nx, cfg.ny)
        bg = solve_background(spec, cfg.grazing_eps_reg)
        phi0 = torch.atan2(spec.ky0, spec.kx0)
        n_i = torch.sqrt(spec.incidence.eps).reshape(-1, 1, 1)
        n_t = torch.sqrt(spec.transmission.eps).reshape(-1, 1, 1)
        Jr = torch.linalg.solve(sp_matrix(spec.kx0, spec.ky0, -bg.lam[0] / 1j, n_i, phi0), bg.c_minus[0])
        Jt = torch.linalg.solve(sp_matrix(spec.kx0, spec.ky0, bg.lam[-1] / 1j, n_t, phi0), bg.c_plus[-1])
        assert_close(Jr, obs.jones("reflection"), atol=1e-12, rtol=0)
        assert_close(Jt, obs.jones("transmission"), atol=1e-12, rtol=0)


# ---------------------------------------------------------------------------
# Affine S-matrix algebra, exponential integrals
# ---------------------------------------------------------------------------

def _random_affine(n=3):
    rb = lambda: Block(Block.DENSE, 0.3 * torch.randn(n, n, dtype=CDT))
    q = lambda: Block2x2(rb(), rb(), rb(), rb())
    return AffineS(Block2x2(q(), q(), q(), q()),
                   torch.randn(2 * n, 2, dtype=CDT), torch.randn(2 * n, 2, dtype=CDT))


class TestAffineStar:

    def test_associative(self):
        torch.manual_seed(0)
        A, B, C = _random_affine(), _random_affine(), _random_affine()
        X, Y = A.star(B).star(C), A.star(B.star(C))
        assert_close(X.S.to_dense(), Y.S.to_dense(), atol=1e-12, rtol=0)
        assert_close(X.e_L, Y.e_L, atol=1e-12, rtol=0)
        assert_close(X.e_R, Y.e_R, atol=1e-12, rtol=0)

    def test_matches_plain_star(self):
        """S part = Block2x2.star, with or without sources."""
        torch.manual_seed(1)
        A, B = _random_affine(), _random_affine()
        ref = A.S.star(B.S).to_dense()
        assert_close(A.star(B).S.to_dense(), ref, atol=1e-12, rtol=0)
        plain = AffineS(A.S).star(AffineS(B.S))
        assert plain.e_L is None and plain.e_R is None
        assert_close(plain.S.to_dense(), ref)

    def test_vector_helpers(self):
        torch.manual_seed(2)
        S = _random_affine().S
        v = torch.randn(12, 2, dtype=CDT)
        assert_close(apply_vec(S, v), S.to_dense() @ v)
        assert_close(solve_vec(S, apply_vec(S, v)), v, atol=1e-10, rtol=0)


class TestExpIntegrals:

    def test_phi1(self):
        x = torch.tensor([1e-9 + 1e-9j, 1e-5j, 3e-3 - 2e-3j, 0.099j, 0.101, 0.5 - 2j, -30 + 1j], dtype=CDT)
        # reference: series for small |x|, direct form otherwise, in extended precision
        ref = []
        for v in x.tolist():
            if abs(v) < 0.2:
                t, s = 1.0 + 0j, 0j
                for k in range(1, 30):
                    s += t
                    t *= v / (k + 1)
                ref.append(s)
            else:
                ref.append((torch.exp(torch.tensor(v, dtype=CDT)).item() - 1) / v)
        assert_close(phi1(x), torch.tensor(ref, dtype=CDT), rtol=1e-12, atol=0)

    def test_exp_integral(self):
        a, b, d = torch.tensor(-0.3 + 2j, dtype=CDT), torch.tensor(-1.0 + 0.5j, dtype=CDT), torch.tensor(1.7, dtype=DT)
        ref = (torch.exp(a * d) - torch.exp(b * d)) / (a - b)
        assert_close(exp_integral(a, b, d), ref)
        assert_close(exp_integral(b, a, d), ref)
        assert_close(exp_integral(a, a, d), d * torch.exp(a * d))     # removable singularity


# ---------------------------------------------------------------------------
# Scattered vs total field
# ---------------------------------------------------------------------------

class TestScatteredEqualsTotal:

    @pytest.mark.parametrize("modesolver", ["matexp", "eig"])
    @pytest.mark.parametrize("tvf", [False, True])
    def test_normal_incidence(self, modesolver, tvf):
        """Thick layer (several matexp slices), with and without TVF."""
        kw = dict(modesolver=modesolver, factorization=Factorization() if tvf else None)
        tot = _obs(_model(thickness=1.3), pml_formulation="total", **kw)
        sca = _obs(_model(thickness=1.3), **kw)
        for kind in ("reflection", "transmission"):
            assert_close(sca.jones(kind), tot.jones(kind), atol=1e-11, rtol=0)

    def test_tilt_along_periodic_axis(self):
        tot = _obs(_model(0.5, math.pi / 2), pml_formulation="total")
        sca = _obs(_model(0.5, math.pi / 2))
        for kind in ("reflection", "transmission"):
            assert_close(sca.jones(kind), tot.jones(kind), atol=1e-11, rtol=0)

    def test_uniform_stack(self):
        """No patterned layer: zero scattered field, specular = periodic result."""
        def uniform(pml):
            stack = Stack(incidence=_medium(1.0), layers=[Layer(medium_solid=_medium(4.0), thickness=0.13)],
                          transmission=_medium(2.25), lattice=Lattice.rectangular(L, 0.5),
                          pml=PML(width=(0.5, 0.0)) if pml else None)
            return _obs(Model(stack, Source(wavelength=torch.tensor(WVL, dtype=DT))))
        a, b = uniform(True), uniform(False)
        assert_close(a.jones("reflection"), b.jones("reflection"), atol=1e-12, rtol=0)
        assert float(a.scattered_power("transmission").abs().max()) == 0.0

    def test_oblique_total_formulation_raises(self):
        with pytest.raises(ValueError, match="Oblique incidence along the PML axis x"):
            Solver(_model(0.3), _config(pml_formulation="total"))


# ---------------------------------------------------------------------------
# Oblique incidence along the PML axis
# ---------------------------------------------------------------------------

class TestOblique:

    def test_eig_and_matexp_agree(self):
        a, b = _obs(_model(0.4), modesolver="matexp"), _obs(_model(0.4), modesolver="eig")
        assert_close(a.jones("transmission"), b.jones("transmission"), atol=1e-9, rtol=0)
        assert_close(a.scattered_power("reflection"), b.scattered_power("reflection"), atol=1e-9, rtol=0)

    def test_energy_balance(self):
        """Lossless structure: extinction ~ total scattered power (finite cell)."""
        obs = _obs(_model(0.4))
        scat = obs.scattered_power("reflection") + obs.scattered_power("transmission")
        assert_close(obs.extinction(), scat, rtol=0.1, atol=0)
        assert (scat > 0).all()

    def test_pattern_integrates_to_scattered_power(self):
        obs = _obs(_model(0.4))
        th = torch.linspace(-math.pi / 2, math.pi / 2, 1201, dtype=DT)
        # Tolerances reflect the tiny cell (physical region 1.0 wide): the
        # window truncates the near field, more so on the air side.
        for kind, rtol in (("reflection", 0.1), ("transmission", 0.03)):
            total = torch.trapezoid(obs.scattering_pattern(kind, th), th, dim=-2)
            assert_close(total, obs.scattered_power(kind) * L, rtol=rtol, atol=0)
        resolved = obs.scattering_pattern("transmission", th, polarization_resolved=True)
        assert_close(resolved.sum(-2), obs.scattering_pattern("transmission", th))

    def test_pattern_needs_lossless_half_space(self):
        obs = _obs(_model(0.3, eps_t=2.25 + 0.5j))
        with pytest.raises(ValueError, match="lossless"):
            obs.scattering_pattern("transmission", 0.1)

    def test_gradients_finite_and_consistent(self):
        d = torch.nn.Parameter(torch.tensor(0.3, dtype=DT))
        grads = []
        for modesolver in ("matexp", "eig"):
            obs = _obs(_model(0.3, thickness=d, shape=_ridge(soft=0.01)), modesolver=modesolver)
            g, = torch.autograd.grad(obs.scattered_power("transmission").sum(), d)
            grads.append(g)
        assert torch.isfinite(grads[0])
        assert_close(grads[0], grads[1], rtol=1e-6, atol=0)


# ---------------------------------------------------------------------------
# Regressions (FIX_PLAN.md)
# ---------------------------------------------------------------------------

class TestRegressions:

    def test_replaced_operator_thickness(self):
        """Background uses the operators' thicknesses, like the layer S-matrices."""
        solver = Solver(_model(thickness=0.3, film=False), _config())
        ops = list(solver._prepared.ops)
        ops[1] = dataclasses.replace(ops[1], thickness=torch.tensor(0.5, dtype=DT))
        solver._prepared = dataclasses.replace(solver._prepared, ops=tuple(ops))
        ref = _obs(_model(thickness=0.5, film=False))
        assert_close(Observables(solver.run()).jones("transmission"), ref.jones("transmission"),
                     atol=1e-11, rtol=0)

    def test_physical_mask_with_inexact_profile(self):
        """s(0) = 1 only to round-off must not empty the physical region."""
        profile = CallableModule(lambda d: 1 + (1e-12 + d) * (1 + 1j) - 1e-12)
        layer = Layer(medium_solid=_medium(4.0), medium_void=_medium(1.0), shape_fn=_ridge(), thickness=0.3)
        stack = Stack(incidence=_medium(1.0), layers=[layer], transmission=_medium(2.25),
                      lattice=Lattice.rectangular(L, 0.5), pml=PML(width=(0.5, 0.0), profile=profile))
        obs = _obs(Model(stack, Source(wavelength=torch.tensor(WVL, dtype=DT))))
        assert int(obs._mask("physical").sum()) == 4 * 48          # half of the 96 x-pixels
        assert (obs.scattered_power("transmission") > 0).all()

    def test_overlap_warning(self):
        """Warns when the structure reaches the PML, not for a soft pattern's tail."""
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            Solver(_model(shape=_ridge(soft=0.02)), _config())
        with pytest.warns(RuntimeWarning, match="inside the PML"):
            Solver(_model(shape=_ridge(x0=0.6)), _config())

    def test_shared_uniform_basis_matches_per_medium_eig(self, monkeypatch):
        shared = _obs(_model(0.3)).jones("transmission")
        monkeypatch.setattr(LayerSolver, "_prepare_uniform_basis", lambda self: None)
        assert_close(shared, _obs(_model(0.3)).jones("transmission"), atol=1e-11, rtol=0)
