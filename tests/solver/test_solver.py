# tests/solver/test_solver.py
# Integration tests for Solver: construction, TVF dispatch, truncation,
# and end-to-end solve() correctness.

import pytest
import torch

from metarcwa.model.base import Model
from metarcwa.model.stack import Stack
from metarcwa.model.layer import Layer
from metarcwa.model.medium import IsotropicMedium
from metarcwa.model.lattice import Lattice
from metarcwa.model.source import PlaneWave
from metarcwa.model.utils import CallableModule
from metarcwa.solver.base import Solver
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
    source       = PlaneWave(wavelength=1.0, s_amp=1.0, p_amp=0.0)
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
    source       = PlaneWave(wavelength=1.0, s_amp=1.0, p_amp=0.0)
    return Model(stack, source).to(dtype=torch.float64, device=device)


def _is_block2x2_like(x) -> bool:
    return all(hasattr(x, attr) for attr in ("a", "b", "c", "d"))


def _get_leaf(entry):
    while hasattr(entry, "a"):
        entry = entry.a
    return entry


def _dense_is_star_id(M: torch.Tensor, atol: float = 1e-5) -> bool:
    """Check that M equals the star-product identity [[0, I], [I, 0]]."""
    N2 = M.shape[-1]
    N  = N2 // 2
    I  = torch.eye(N, dtype=M.dtype, device=M.device)
    return (
        M[..., :N, :N].abs().max().item() < atol
        and M[..., N:, N:].abs().max().item() < atol
        and (M[..., :N, N:] - I).abs().max().item() < atol
        and (M[..., N:, :N] - I).abs().max().item() < atol
    )


def _nh(solver: Solver) -> int:
    return solver.layersolver.m_flat.shape[0]


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
# solve()
# ---------------------------------------------------------------------------

class TestSolverSolve:

    def test_returns_block2x2_like(self, device):
        solver = Solver(_make_model(device), Config(m=1, n=1))
        assert _is_block2x2_like(solver.solve())

    def test_no_nan(self, device):
        solver = Solver(_make_model(device), Config(m=1, n=1))
        Nh = _nh(solver)
        M  = solver.solve().to_dense(Nh)
        assert not torch.isnan(M).any()

    def test_vacuum_stack_is_star_identity(self, device):
        """All-vacuum stack → S = star-product identity [[0, I], [I, 0]]."""
        solver = Solver(_make_vacuum_model(device), Config(m=1, n=1))
        Nh = _nh(solver)
        M  = solver.solve().to_dense(Nh)
        assert _dense_is_star_id(M, atol=1e-6)

    def test_output_on_correct_device(self, device):
        solver = Solver(_make_model(device), Config(m=1, n=1, device=device))
        Nh = _nh(solver)
        M  = solver.solve().to_dense(Nh)
        assert M.device.type == device

    def test_slab_has_nonzero_transmission(self, device):
        """ε=2.5 slab must have non-zero transmission block."""
        solver = Solver(_make_model(device), Config(m=1, n=1))
        Nh = _nh(solver)
        M  = solver.solve().to_dense(Nh)
        N  = M.shape[-1] // 2
        assert M[..., :N, N:].abs().max().item() > 1e-6

    def test_slab_has_nonzero_reflection(self, device):
        """ε=2.5 slab must have non-zero reflection block."""
        solver = Solver(_make_model(device), Config(m=1, n=1))
        Nh = _nh(solver)
        M  = solver.solve().to_dense(Nh)
        N  = M.shape[-1] // 2
        assert M[..., :N, :N].abs().max().item() > 1e-6
