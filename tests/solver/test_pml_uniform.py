# tests/solver/test_pml_uniform.py
# Tests for uniform media under a lateral PML (homogeneous layers and
# half-spaces, LayerSolver._stretched_uniform_modes), the background gap
# medium and full-stack solves. See docs/PML.md.

import pytest
import torch
from torch.testing import assert_close

from metarcwa import Lattice, Layer, Model, PML, Solver, Source, Stack, StretchAbsorbProfile
from metarcwa.model.nn_helpers import CallableModule
from metarcwa.solver.layersolver.operator import Background

from tests.solver.test_pml_stretch import _config, _disk, _medium, _unit_profile


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

N_FILM, D_FILM = (2.0, 2.0 ** 0.5), (0.13, 0.27)     # two films: eps 4.0 and 2.0
N_SUB, WVL = 1.5, 0.8


def _uniform_model(pml, theta=0.0, phi=0.0):
    layers = [Layer(medium_solid=_medium(n ** 2), thickness=d) for n, d in zip(N_FILM, D_FILM)]
    stack = Stack(incidence=_medium(1.0), layers=layers, transmission=_medium(N_SUB ** 2),
                  lattice=Lattice.rectangular(1.0, 1.0), pml=pml)
    return Model(stack, Source(wavelength=WVL, theta=theta, phi=phi))


def _patterned_model(pml):
    layers = [Layer(medium_solid=_medium(2.0), thickness=0.1),
              Layer(medium_solid=_medium(4.0), medium_void=_medium(1.0),
                    shape_fn=CallableModule(_disk), thickness=0.3)]
    stack = Stack(incidence=_medium(1.0), layers=layers, transmission=_medium(2.25),
                  lattice=Lattice.rectangular(1.0, 1.0), pml=pml)
    return Model(stack, Source(wavelength=WVL))


def _dense2d(X, nh):
    """Dense [2Nh, 2Nh] matrix of a Block2x2 (first batch element)."""
    x = X.to_dense(nh)
    while x.ndim > 2:
        x = x[0]
    return x


def _field_rt(solution):
    """Field-level reflection / transmission matrices W_i S11 W_i^-1 and
    W_t S21 W_i^-1 — independent of the half-space mode basis (order and
    phase of eigenvectors)."""
    nh = solution.prepared.layersolver.m_flat.shape[0]
    ops = solution.prepared.ops
    Wi = _dense2d(ops[0].W, nh).to(torch.complex128)
    Wt = _dense2d(ops[-1].W, nh).to(torch.complex128)
    Wi_inv = torch.linalg.inv(Wi)
    return Wi @ _dense2d(solution.S11, nh) @ Wi_inv, Wt @ _dense2d(solution.S21, nh) @ Wi_inv


def _zeroth_order(solution, pol=0):
    """Field-level zeroth-order (r, t) for an incident zeroth harmonic.

    S11/S21 act on half-space *mode* amplitudes; under the PML the mode
    basis W is a dense eigenvector matrix, so the incident field e is mapped
    to modes, scattered, and mapped back to fields: r = (W_i S11 W_i^-1 e)_e.
    ``pol`` selects E_x (0) or E_y (1).
    """
    ls = solution.prepared.layersolver
    nh = ls.m_flat.shape[0]
    ops = solution.prepared.ops
    Wi, Wt = _dense2d(ops[0].W, nh), _dense2d(ops[-1].W, nh)
    S11, S21 = _dense2d(solution.S11, nh), _dense2d(solution.S21, nh)
    i0 = int(((ls.m_flat == 0) & (ls.n_flat == 0)).nonzero()) + pol * nh
    e = torch.zeros(2 * nh, dtype=Wi.dtype)
    e[i0] = 1
    c = torch.linalg.solve(Wi, e)
    return (Wi @ (S11 @ c))[i0], (Wt @ (S21 @ c))[i0]


def _tmm_normal(wvl):
    """Normal-incidence characteristic-matrix TMM: power (R, T)."""
    k0 = 2 * torch.pi / wvl
    M = torch.eye(2, dtype=torch.complex128)
    for n, d in zip(N_FILM, D_FILM):
        dl = torch.tensor(k0 * n * d, dtype=torch.float64)
        M = M @ torch.stack([torch.stack([torch.cos(dl), -1j * torch.sin(dl) / n]),
                             torch.stack([-1j * n * torch.sin(dl), torch.cos(dl)])])
    B, C = M @ torch.tensor([1.0, N_SUB], dtype=torch.complex128)
    r, t = (B - C) / (B + C), 2 / (B + C)
    return abs(r) ** 2, N_SUB * abs(t) ** 2


# ---------------------------------------------------------------------------
# Uniform-medium modes under PML
# ---------------------------------------------------------------------------

class TestUniformModes:

    @pytest.mark.parametrize("truncation", ["rectangular", "circular"])
    def test_unit_profile_matches_closed_form(self, truncation):
        """s = 1: the eigensolver path reproduces the closed form. Compared in
        the field basis (the mode basis W differs: identity vs eigenvectors
        of a degenerate problem): the layer propagator, and the full
        field-level reflection / transmission matrices of the stack."""
        cfg = _config(truncation=truncation, m=4, n=4, nx=32, ny=32)
        out = {}
        for key, pml in (("ref", None), ("pml", PML(width=0.25, profile=_unit_profile()))):
            solution = Solver(_uniform_model(pml), cfg).run()
            ls, ops = solution.prepared.layersolver, solution.prepared.ops
            transfer = _dense2d(ops[1].transfer(ls.background, torch.tensor(0.1)), ls.m_flat.shape[0])
            out[key] = (transfer, *_field_rt(solution))
        for a, b in zip(out["ref"], out["pml"]):
            assert_close(b, a.to(torch.complex128), atol=1e-10, rtol=0)

    @pytest.mark.parametrize("truncation", ["rectangular", "circular"])
    def test_half_space_modes_physical(self, truncation):
        """All half-space modes lie in the physical quadrant (decaying,
        forward phase), up to the grazing regularization."""
        solution = Solver(_patterned_model(PML(width=0.25)), _config(truncation=truncation)).run()
        for op in (solution.prepared.ops[0], solution.prepared.ops[-1]):
            assert op.lam.real.max() < 1e-12
            assert op.lam.imag.min() > -1e-12


# ---------------------------------------------------------------------------
# Full stacks
# ---------------------------------------------------------------------------

class TestUniformStack:

    @pytest.mark.parametrize("truncation", ["rectangular", "circular"])
    @pytest.mark.parametrize("pol", [0, 1])
    def test_normal_incidence_matches_periodic_and_tmm(self, truncation, pol):
        """The zeroth harmonic is an exact mode under PML at normal incidence,
        so a uniform stack gives the periodic / TMM zeroth order exactly."""
        cfg = _config(truncation=truncation, m=4, n=4, nx=32, ny=32)
        r0, t0 = _zeroth_order(Solver(_uniform_model(None), cfg).run(), pol)
        r1, t1 = _zeroth_order(Solver(_uniform_model(PML(width=0.25)), cfg).run(), pol)
        assert_close(r1, r0, atol=1e-12, rtol=0)
        assert_close(t1, t0, atol=1e-12, rtol=0)
        R, T = _tmm_normal(WVL)
        assert abs(abs(r1) ** 2 - R) < 1e-6 and abs(N_SUB * abs(t1) ** 2 - T) < 1e-6

    @pytest.mark.parametrize("pol", [0, 1])
    def test_tilt_along_non_pml_axis_matches_periodic(self, pol):
        """x-only PML, plane of incidence along y: the zeroth harmonic is
        still an exact mode (kx = 0), so the oblique zeroth order matches."""
        cfg = _config(m=4, n=4, nx=32, ny=32)
        args = dict(theta=0.4, phi=torch.pi / 2)
        r0, t0 = _zeroth_order(Solver(_uniform_model(None, **args), cfg).run(), pol)
        r1, t1 = _zeroth_order(Solver(_uniform_model(PML(width=(0.25, 0.0)), **args), cfg).run(), pol)
        assert_close(r1, r0, atol=1e-12, rtol=0)
        assert_close(t1, t0, atol=1e-12, rtol=0)


class TestPatternedStack:

    @pytest.mark.parametrize("truncation", ["rectangular", "circular"])
    def test_gap_medium_choice_is_exact(self, truncation):
        """Referencing the stack to a stretched vacuum gap instead of the
        default unstretched one gives the same total S-matrix."""
        solution = Solver(_patterned_model(PML(width=0.25)), _config(truncation=truncation)).run()
        ls, ops = solution.prepared.layersolver, solution.prepared.ops
        nh = ls.m_flat.shape[0]
        vacuum = torch.ones(1, dtype=torch.complex128) * (1 + 1e-8j)
        _, Wg, Vg = ls._stretched_uniform_modes(vacuum)
        gap = Background(Wg, Vg, ls.wvl)
        S = ops[0].smatrix(gap, left=True)
        for op in ops[1:-1]:
            S = S.star(op.smatrix(gap))
        S = S.star(ops[-1].smatrix(gap, left=False))
        scale = _dense2d(solution.S11, nh).abs().max()
        assert (_dense2d(S.a, nh) - _dense2d(solution.S11, nh)).abs().max() < 1e-11 * scale
        assert (_dense2d(S.c, nh) - _dense2d(solution.S21, nh)).abs().max() < 1e-11 * scale

    @pytest.mark.parametrize("truncation", ["rectangular", "circular"])
    def test_eig_and_matexp_agree(self, truncation):
        S = {}
        for modesolver in ("eig", "matexp"):
            cfg = _config(truncation=truncation, modesolver=modesolver)
            S[modesolver] = _field_rt(Solver(_patterned_model(PML(width=0.25)), cfg).run())
        for a, b in zip(S["eig"], S["matexp"]):
            assert_close(b, a, atol=1e-10, rtol=0)

    def test_matexp_bound_covers_modal_exponents(self):
        """The slice-count bound from the stretched K̃ is a true upper bound."""
        lam_max = Solver(_patterned_model(PML(width=0.25)), _config(modesolver="eig"))._ops[2].lam.abs().max()
        op = Solver(_patterned_model(PML(width=0.25)), _config(modesolver="matexp"))._ops[2]
        assert op.lam_bound >= lam_max

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_runs_on_cuda(self):
        cpu = _field_rt(Solver(_patterned_model(PML(width=0.25)), _config(m=3, n=3, nx=32, ny=32)).run())
        gpu = _field_rt(Solver(_patterned_model(PML(width=0.25)),
                               _config(m=3, n=3, nx=32, ny=32, device="cuda")).run())
        for a, b in zip(cpu, gpu):
            assert_close(b.cpu(), a, atol=1e-9, rtol=0)

    def test_gradient_through_full_stack(self):
        sigma = torch.nn.Parameter(torch.tensor(1.0, dtype=torch.float64))
        pml = PML(width=0.25, profile=StretchAbsorbProfile(sigma_max=sigma))
        solution = Solver(_patterned_model(pml), _config(m=3, n=3, nx=32, ny=32)).run()
        r, _ = _zeroth_order(solution)
        abs(r).backward()
        assert sigma.grad is not None and torch.isfinite(sigma.grad) and sigma.grad != 0
