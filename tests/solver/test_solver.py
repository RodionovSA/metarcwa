# tests/solver/test_solver.py
# Integration tests for Solver: construction, TVF dispatch, truncation,
# and end-to-end run() correctness.

import pytest
import torch

import torch.nn as nn

from metarcwa.model.base import Model
from metarcwa.model.stack import Stack
from metarcwa.model.layer import Layer
from metarcwa.model.medium import IsotropicMedium
from metarcwa.model.lattice import Lattice
from metarcwa.model.source import Source
from metarcwa.model.utils import CallableModule
from metarcwa.solver.base import Solver, prepare, run, reprepare, PreparedStack, ModalSolution
from metarcwa.solver.layersolver.base import LayerSolver
from metarcwa.solver.config import Config
from metarcwa.solver.blockmatrix import Block


# ---------------------------------------------------------------------------
# Fixtures & helpers
# ---------------------------------------------------------------------------

@pytest.fixture(
    params=[
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="CUDA not available"
            ),
        ),
    ]
)
def device(request):
    return request.param


def _const_eps(val: complex):
    """Return a CallableModule that returns a constant complex eps."""
    return CallableModule(lambda wvl: torch.full_like(wvl, val, dtype=torch.complex128))


def _make_model(device: str) -> Model:
    """Minimal model: one ε=2.5 slab, vacuum incidence and transmission.

    Uses a 1×1 rectangular unit cell and normal-incidence plane-wave source
    at wavelength 1.0.
    """
    incidence    = IsotropicMedium(_const_eps(1.0 + 0j))
    transmission = IsotropicMedium(_const_eps(1.0 + 0j))
    layer        = Layer(IsotropicMedium(_const_eps(2.5 + 0j)), thickness=0.3)
    lattice      = Lattice.rectangular(1.0, 1.0)
    stack        = Stack(incidence, [layer], transmission, lattice)
    source       = Source(wavelength=1.0)
    return Model(stack, source).to(dtype=torch.float64, device=device)


def _make_vacuum_model(device: str) -> Model:
    """All-vacuum model: ε=1 everywhere, zero-thickness layer.

    Zero thickness ensures exp(i·kz·d) = I so the full S-matrix equals the
    star-product identity [[0, I], [I, 0]].
    """
    incidence    = IsotropicMedium(_const_eps(1.0 + 0j))
    transmission = IsotropicMedium(_const_eps(1.0 + 0j))
    layer        = Layer(IsotropicMedium(_const_eps(1.0 + 0j)), thickness=0.0)
    lattice      = Lattice.rectangular(1.0, 1.0)
    stack        = Stack(incidence, [layer], transmission, lattice)
    source       = Source(wavelength=1.0)
    return Model(stack, source).to(dtype=torch.float64, device=device)


def _get_leaf(entry):
    while hasattr(entry, "a"):
        entry = entry.a
    return entry


def _dense_block(entry, n: int) -> torch.Tensor:
    """Densify one S-matrix block (leaf Block or nested Block2x2) to a tensor.

    ``ModalSolution.S11``/``.S21`` are leaf ``Block``s coming out of ``run()``
    (no ``.to_dense``), but a manually star-composed block (e.g. from
    ``LayerSolver.run``) may be a nested ``Block2x2`` (has ``.to_dense``
    directly) — dispatch on which one it is.
    """
    return entry.to_dense(n) if hasattr(entry, "a") else entry.to(Block.DENSE, n).data


def _blocks_are_star_id(sol: ModalSolution, n: int, atol: float = 1e-5) -> bool:
    """Check that (S11, S21) equal the star-product identity's first column:
    reflection S11 ≈ 0, transmission S21 ≈ I."""
    S11 = _dense_block(sol.S11, n)
    S21 = _dense_block(sol.S21, n)
    I = torch.eye(S21.shape[-1], dtype=S21.dtype, device=S21.device)
    return (
        S11.abs().max().item() < atol
        and (S21 - I).abs().max().item() < atol
    )


def _nh(solver: Solver) -> int:
    return solver.layersolver.m_flat.shape[0]


def _circle_mask_fn(radius: torch.Tensor, softness: float = 0.02):
    """Smooth circular mask centered on the unit cell, differentiable in radius."""
    def fn(lattice, nx, ny):
        fx = torch.linspace(0, 1, nx, device=lattice.device, dtype=lattice.dtype)
        fy = torch.linspace(0, 1, ny, device=lattice.device, dtype=lattice.dtype)
        FY, FX = torch.meshgrid(fy, fx, indexing="ij")
        X, Y = lattice.to_cartesian(FX, FY)
        cx = 0.5 * (lattice.a1[0] + lattice.a2[0])
        cy = 0.5 * (lattice.a1[1] + lattice.a2[1])
        dist = torch.sqrt((X - cx) ** 2 + (Y - cy) ** 2 + 1e-12)
        return torch.sigmoid((radius - dist) / softness)
    return fn


def _make_two_layer_model(device: str, radius: nn.Parameter) -> Model:
    """Homogeneous layer + a patterned (circle) layer whose radius is an
    ``nn.Parameter``, so mutating it in place changes the pattern.

    Layer order: [homogeneous ε=2.5, patterned ε_solid=4/ε_void=1 circle].
    """
    incidence    = IsotropicMedium(_const_eps(1.0 + 0j))
    transmission = IsotropicMedium(_const_eps(1.0 + 0j))
    layer0       = Layer(IsotropicMedium(_const_eps(2.5 + 0j)), thickness=0.1)
    shape_fn     = CallableModule(_circle_mask_fn(radius), radius)
    layer1       = Layer(
        IsotropicMedium(_const_eps(4.0 + 0j)), thickness=0.15,
        medium_void=IsotropicMedium(_const_eps(1.0 + 0j)), shape_fn=shape_fn,
    )
    lattice = Lattice.rectangular(1.0, 1.0)
    stack   = Stack(incidence, [layer0, layer1], transmission, lattice)
    # Patterned-layer eps blending indexes wavelength as [N_wl, ...], so it
    # must be at least 1-D (unlike the scalar-float form used for
    # homogeneous-only fixtures elsewhere in this file).
    source  = Source(wavelength=torch.tensor([1.0]))
    return Model(stack, source).to(dtype=torch.float64, device=device)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

class TestSolverInit:

    def test_construction_default_config(self, device):
        Solver(_make_model(device), Config(m=1, n=1))

    def test_construction_no_tvf(self, device):
        solver = Solver(_make_model(device), Config(m=1, n=1, factorization=None))
        assert solver.layersolver.tvf is None

    def test_construction_with_tvf(self, device):
        solver = Solver(_make_model(device), Config(m=1, n=1))
        assert solver.layersolver.tvf is not None

    def test_rectangular_truncation_more_harmonics_than_circular(self, device):
        """Bug-regression: truncation='rectangular' must not use circular mode."""
        m, n = 2, 2
        cfg_rect = Config(m=m, n=n, truncation="rectangular")
        cfg_circ = Config(m=m, n=n, truncation="circular")
        solver_rect = Solver(_make_model(device), cfg_rect)
        solver_circ = Solver(_make_model(device), cfg_circ)
        Nh_rect = _nh(solver_rect)
        Nh_circ = _nh(solver_circ)
        assert Nh_rect > Nh_circ, (
            f"rectangular ({Nh_rect}) should have more harmonics than "
            f"circular ({Nh_circ}) for m=n={m}"
        )

    def test_model_moved_to_config_device(self, device):
        solver = Solver(_make_model("cpu"), Config(m=1, n=1))
        assert solver.model_spec.wavelength.device.type == "cpu"


# ---------------------------------------------------------------------------
# run()
# ---------------------------------------------------------------------------

class TestSolverRun:

    def test_returns_modal_solution(self, device):
        solver = Solver(_make_model(device), Config(m=1, n=1))
        sol = solver.run()
        assert isinstance(sol, ModalSolution)
        assert sol.prepared is solver._prepared
        Nh = _nh(solver)
        S11 = _dense_block(sol.S11, Nh)
        S21 = _dense_block(sol.S21, Nh)
        # S11/S21 are each nested one level (Sx/Sy field-vector sub-blocks),
        # so their dense size is a multiple of Nh, not Nh itself -- assert
        # the weaker, correct invariant: square and matching between blocks.
        assert S11.shape[-1] == S11.shape[-2]
        assert S21.shape == S11.shape

    def test_no_nan(self, device):
        solver = Solver(_make_model(device), Config(m=1, n=1))
        Nh = _nh(solver)
        sol = solver.run()
        assert not torch.isnan(_dense_block(sol.S11, Nh)).any()
        assert not torch.isnan(_dense_block(sol.S21, Nh)).any()

    def test_vacuum_stack_is_star_identity(self, device):
        """All-vacuum stack → (S11, S21) = star-product identity's first
        column: reflection 0, transmission I."""
        solver = Solver(_make_vacuum_model(device), Config(m=1, n=1))
        Nh = _nh(solver)
        sol = solver.run()
        assert _blocks_are_star_id(sol, Nh, atol=1e-6)

    def test_output_on_correct_device(self, device):
        solver = Solver(_make_model(device), Config(m=1, n=1, device=device))
        Nh = _nh(solver)
        sol = solver.run()
        assert _dense_block(sol.S11, Nh).device.type == device

    def test_slab_has_nonzero_transmission(self, device):
        """ε=2.5 slab must have non-zero transmission block (S21)."""
        solver = Solver(_make_model(device), Config(m=1, n=1))
        Nh = _nh(solver)
        sol = solver.run()
        assert _dense_block(sol.S21, Nh).abs().max().item() > 1e-6

    def test_slab_has_nonzero_reflection(self, device):
        """ε=2.5 slab must have non-zero reflection block (S11)."""
        solver = Solver(_make_model(device), Config(m=1, n=1))
        Nh = _nh(solver)
        sol = solver.run()
        assert _dense_block(sol.S11, Nh).abs().max().item() > 1e-6

    def test_run_fields_not_implemented(self, device):
        """Solver.run(fields=True) is not built yet — must raise."""
        solver = Solver(_make_model(device), Config(m=1, n=1))
        with pytest.raises(NotImplementedError):
            solver.run(fields=True)


# ---------------------------------------------------------------------------
# Operator precompute (LAYERSOLVER_PLAN Step 3, T6)
# ---------------------------------------------------------------------------

class TestSolverPrecompute:

    def test_ops_precomputed_at_init(self, device):
        """__init__ prepares one LayerOperator per stack element: incidence,
        each finite layer, transmission — boundaries have thickness=None,
        the finite layer does not."""
        solver = Solver(_make_model(device), Config(m=1, n=1))
        n_layers = len(solver.model_spec.layers)
        assert len(solver._ops) == n_layers + 2
        assert solver._ops[0].thickness is None          # incidence
        assert solver._ops[-1].thickness is None          # transmission
        for op in solver._ops[1:-1]:
            assert op.thickness is not None

    def test_run_matches_manual_layersolver_composition(self, device):
        """run() (star-composing precomputed operators) must exactly match
        the composition built directly from LayerSolver.run(), block for
        block (S11 vs. the manual composition's a=S11, S21 vs. its c=S21)."""
        solver = Solver(_make_model(device), Config(m=1, n=1))
        Nh = _nh(solver)

        ls = solver.layersolver
        S_manual = ls.run(solver.model_spec.incidence, left=True)
        for layer in solver.model_spec.layers:
            S_manual = S_manual.star(ls.run(layer))
        S_manual = S_manual.star(ls.run(solver.model_spec.transmission, left=False))

        S_new = solver.run()
        assert torch.allclose(_dense_block(S_new.S11, Nh), _dense_block(S_manual.a, Nh))
        assert torch.allclose(_dense_block(S_new.S21, Nh), _dense_block(S_manual.c, Nh))

    def test_run_is_deterministic_across_calls(self, device):
        """Repeated run() calls reuse the precomputed operators and must
        return bit-identical results (no re-solving of the eigenproblem)."""
        solver = Solver(_make_model(device), Config(m=1, n=1))
        Nh = _nh(solver)
        sol1 = solver.run()
        sol2 = solver.run()
        assert torch.equal(_dense_block(sol1.S11, Nh), _dense_block(sol2.S11, Nh))
        assert torch.equal(_dense_block(sol1.S21, Nh), _dense_block(sol2.S21, Nh))


# ---------------------------------------------------------------------------
# Functional core (prepare / run / reprepare)
# ---------------------------------------------------------------------------

class TestFunctionalCore:

    def test_prepare_run_matches_class(self, device):
        """The functional core must match the Solver class wrapper exactly
        (the class is a thin wrapper around prepare()/run())."""
        model = _make_model(device)
        config = Config(m=1, n=1)
        prepared = prepare(model, config)
        assert isinstance(prepared, PreparedStack)
        Nh = prepared.layersolver.m_flat.shape[0]

        solver = Solver(_make_model(device), config)
        sol_func = run(prepared)
        sol_class = solver.run()
        assert torch.allclose(_dense_block(sol_func.S11, Nh), _dense_block(sol_class.S11, Nh))
        assert torch.allclose(_dense_block(sol_func.S21, Nh), _dense_block(sol_class.S21, Nh))


class TestReprepare:

    def test_reprepare_matches_full_rebuild(self, device):
        """reprepare() on the changed layer must give the exact same S-matrix
        as building a fresh Solver from a model with the same new pattern."""
        radius = nn.Parameter(torch.tensor(0.2, dtype=torch.float64))
        model = _make_two_layer_model(device, radius)
        config = Config(m=2, n=2, factorization=None, truncation="rectangular")

        solver = Solver(model, config)
        Nh = _nh(solver)

        with torch.no_grad():
            radius.copy_(torch.tensor(0.35, dtype=radius.dtype, device=radius.device))

        sol_reprepared = solver.reprepare([1]).run()

        radius_fresh = nn.Parameter(torch.tensor(0.35, dtype=torch.float64))
        model_fresh = _make_two_layer_model(device, radius_fresh)
        sol_fresh = Solver(model_fresh, config).run()

        assert torch.allclose(
            _dense_block(sol_reprepared.S11, Nh), _dense_block(sol_fresh.S11, Nh), atol=1e-8
        )
        assert torch.allclose(
            _dense_block(sol_reprepared.S21, Nh), _dense_block(sol_fresh.S21, Nh), atol=1e-8
        )

    def test_reprepare_reuses_unchanged_ops_by_identity(self, device):
        """Only the targeted layer's operator is rebuilt; every other
        operator (and the layersolver context) is reused by reference."""
        radius = nn.Parameter(torch.tensor(0.2, dtype=torch.float64))
        model = _make_two_layer_model(device, radius)
        config = Config(m=1, n=1, factorization=None, truncation="rectangular")

        solver = Solver(model, config)
        old_ls = solver.layersolver
        old_ops = solver._ops

        with torch.no_grad():
            radius.copy_(torch.tensor(0.3, dtype=radius.dtype, device=radius.device))
        solver.reprepare([1])

        assert solver.layersolver is old_ls
        assert solver._ops[0] is old_ops[0]   # incidence, unchanged
        assert solver._ops[1] is old_ops[1]   # homogeneous layer 0, unchanged
        assert solver._ops[2] is not old_ops[2]  # patterned layer 1, rebuilt
        assert solver._ops[3] is old_ops[3]   # transmission, unchanged

    def test_reprepare_calls_prepare_once(self, device, monkeypatch):
        """reprepare([1]) on a 2-layer stack must call LayerSolver.prepare
        exactly once (only for the targeted layer)."""
        radius = nn.Parameter(torch.tensor(0.2, dtype=torch.float64))
        model = _make_two_layer_model(device, radius)
        config = Config(m=1, n=1, factorization=None, truncation="rectangular")
        solver = Solver(model, config)

        calls = []
        orig_prepare = LayerSolver.prepare

        def counting_prepare(self, element):
            calls.append(element)
            return orig_prepare(self, element)

        monkeypatch.setattr(LayerSolver, "prepare", counting_prepare)

        with torch.no_grad():
            radius.copy_(torch.tensor(0.3, dtype=radius.dtype, device=radius.device))
        solver.reprepare([1])

        assert len(calls) == 1

    def test_single_layer_optimization_loop(self, device):
        """End-to-end smoke test: reprepare() drives a single-layer geometry
        optimization loop, backprop reaches the changed layer's parameter,
        and the loss decreases."""
        radius = nn.Parameter(torch.tensor(0.15, dtype=torch.float64))
        model = _make_two_layer_model(device, radius)
        config = Config(m=1, n=1, factorization=None, truncation="rectangular")
        solver = Solver(model, config)
        Nh = _nh(solver)

        opt = torch.optim.Adam([radius], lr=1e-3)
        losses = []
        for _ in range(5):
            opt.zero_grad()
            solver.reprepare([1])
            sol = solver.run()
            loss = _dense_block(sol.S21, Nh).abs().pow(2).sum()
            loss.backward()
            losses.append(loss.item())
            opt.step()

        assert radius.grad is not None
        assert losses[-1] < losses[0]
