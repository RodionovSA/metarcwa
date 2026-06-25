# tests/solver/test_layersolver.py
# Tests for LayerSolver: construction, element dispatch, homogeneous layers,
# patterned layers, and semi-infinite medium boundaries.

import pytest
import torch
from torch.testing import assert_close

from metarcwa.solver.config import Config
from metarcwa.solver.layersolver.base import LayerSolver
from metarcwa.solver.layersolver.homogeneous import homogeneous_modes
from metarcwa.solver.blockmatrix import Block, Block2x2
from metarcwa.solver.smatrix import S_prop
from metarcwa.solver.harmonics import harmonic_index_map, compute_kxy
from metarcwa.model.layer import HomogeneousLayer, PatternedLayer
from metarcwa.model.medium import MediumSpec, IsotropicMediumSpec


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

Nh_half = 1   # (2·1+1)² = 9 harmonics — small for speed


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


def _make_solver(device: str):
    """Build a LayerSolver for normal incidence on a square unit cell.

    Returns (solver, kx, ky, m_flat, wvl, Nh).
    """
    a1 = torch.tensor([1.0, 0.0], dtype=torch.float64, device=device)
    a2 = torch.tensor([0.0, 1.0], dtype=torch.float64, device=device)
    # Oblique incidence so kx*ky ≠ 0 for every harmonic; normal incidence
    # (kx0=ky0=0) creates harmonics with kx*ky=0, which makes V.d singular.
    kx0 = torch.tensor([0.0], dtype=torch.float64, device=device)
    ky0 = torch.tensor([0.0], dtype=torch.float64, device=device)
    m_flat, n_flat = harmonic_index_map(Nh_half, Nh_half, device=device)
    kx, ky = compute_kxy(kx0, ky0, a1, a2, m_flat, n_flat)   # [1, Nh]
    wvl    = torch.tensor([1.0], dtype=torch.float64, device=device)
    solver = LayerSolver(Config(), wvl, kx, ky, m_flat, n_flat, tvf=None)
    Nh = m_flat.shape[0]
    return solver, kx, ky, m_flat, wvl, Nh


def _eps(val: float, device: str) -> torch.Tensor:
    return torch.tensor([val + 0j], dtype=torch.complex128, device=device)


def _d(val: float, device: str) -> torch.Tensor:
    return torch.tensor([val], dtype=torch.float64, device=device)


def _medium(eps_val: float, device: str) -> IsotropicMediumSpec:
    return IsotropicMediumSpec(eps=_eps(eps_val, device))


def _hom(eps_val: float, d_val: float, device: str) -> HomogeneousLayer:
    return HomogeneousLayer(thickness=_d(d_val, device),
                            medium=_medium(eps_val, device))


def _pat(eps_solid: float, eps_void: float, d_val: float,
         pattern: torch.Tensor, device: str) -> PatternedLayer:
    return PatternedLayer(
        thickness=_d(d_val, device),
        medium_solid=_medium(eps_solid, device),
        medium_void=_medium(eps_void, device),
        pattern=pattern.to(device),
    )


def _checkerboard(device: str) -> torch.Tensor:
    """8×8 checkerboard (50% fill) — ensures non-degenerate Ω² eigenvectors."""
    pat = torch.zeros(8, 8, dtype=torch.float64)
    pat[::2, ::2]   = 1.0
    pat[1::2, 1::2] = 1.0
    return pat.to(device)


def _get_leaf(entry):
    """Walk nested Block2x2 down to the first leaf Block."""
    while hasattr(entry, 'a'):
        entry = entry.a
    return entry


def _is_block2x2_like(x) -> bool:
    """Duck-type check: has .a, .b, .c, .d attributes (Block2x2-like)."""
    return all(hasattr(x, attr) for attr in ('a', 'b', 'c', 'd'))


def _dense_is_star_id(M: torch.Tensor, atol: float = 1e-5) -> bool:
    """Check that a dense [..., 2N, 2N] tensor equals [[0, I], [I, 0]]."""
    N2 = M.shape[-1]
    N  = N2 // 2
    I  = torch.eye(N, dtype=M.dtype, device=M.device)
    return (M[..., :N, :N].abs().max().item() < atol and
            M[..., N:, N:].abs().max().item() < atol and
            (M[..., :N, N:] - I).abs().max().item() < atol and
            (M[..., N:, :N] - I).abs().max().item() < atol)


# ---------------------------------------------------------------------------
# Construction and vacuum pre-computation
# ---------------------------------------------------------------------------

class TestLayerSolverInit:

    def test_w0_is_identity_matrix(self, device):
        """W0 = I: to_dense should give the identity matrix [[I, 0], [0, I]]."""
        solver, *_, Nh = _make_solver(device)
        M  = solver.W0.to_dense(Nh)
        N2 = M.shape[-1]
        eye = torch.eye(N2, dtype=M.dtype, device=M.device)
        assert_close(M.squeeze(0), eye, atol=1e-10, rtol=0)

    def test_v0_is_block2x2_like(self, device):
        """V0 should be a Block2x2-like object with DIAG leaf entries."""
        solver, *_, Nh = _make_solver(device)
        v0 = solver.V0
        assert _is_block2x2_like(v0), "V0 is not Block2x2-like"
        for entry in (v0.a, v0.b, v0.c, v0.d):
            assert entry.kind == Block.DIAG

    def test_v0_no_nan(self, device):
        """Vacuum mode computation must not produce NaN (checks complex eps fix)."""
        solver, *_, Nh = _make_solver(device)
        M = solver.V0.to_dense(Nh)
        assert not torch.isnan(M).any(), "V0 contains NaN (eps was probably real)"

    def test_vacuum_modes_on_correct_device(self, device):
        solver, *_ = _make_solver(device)
        assert _get_leaf(solver.W0).data.device.type == device
        assert _get_leaf(solver.V0).data.device.type == device


# ---------------------------------------------------------------------------
# solve() dispatch
# ---------------------------------------------------------------------------

class TestLayerSolverDispatch:

    def test_dispatches_homogeneous(self, device):
        solver, *_, Nh = _make_solver(device)
        S = solver.solve(_hom(2.5, 0.3, device))
        assert _is_block2x2_like(S)

    def test_dispatches_patterned(self, device):
        solver, *_ = _make_solver(device)
        S = solver.solve(_pat(2.5, 1.0, 0.3, _checkerboard(device), device))
        assert _is_block2x2_like(S)

    def test_dispatches_medium(self, device):
        solver, *_ = _make_solver(device)
        assert _is_block2x2_like(solver.solve(_medium(2.5, device)))

    def test_unknown_element_type_raises(self, device):
        solver, *_ = _make_solver(device)
        with pytest.raises(TypeError):
            solver.solve("not_an_element")


# ---------------------------------------------------------------------------
# _homogeneous
# ---------------------------------------------------------------------------

class TestLayerSolverHomogeneous:

    def test_returns_block2x2_like(self, device):
        solver, *_ = _make_solver(device)
        assert _is_block2x2_like(solver.solve(_hom(2.5, 0.3, device)))

    def test_no_nan(self, device):
        solver, *_, Nh = _make_solver(device)
        M = solver.solve(_hom(2.5, 0.3, device)).to_dense(Nh)
        assert not torch.isnan(M).any()

    def test_vacuum_layer_equals_s_prop(self, device):
        """HomogeneousLayer(ε=1) in vacuum → S = S_in ⋆ S_p ⋆ S_out = S_p
        because S_in = star_identity (same medium on both sides).
        """
        solver, kx, ky, _, wvl, Nh = _make_solver(device)
        eps_vac = torch.tensor([[1.0 + 0j]], dtype=torch.complex128, device=device)
        lam_vac, _ = homogeneous_modes(eps_vac, kx, ky)
        d_val = 0.3

        S_hom = solver.solve(_hom(1.0, d_val, device))
        S_ref = S_prop(lam_vac, wvl, _d(d_val, device))

        assert_close(S_hom.to_dense(Nh), S_ref.to_dense(Nh), atol=1e-5, rtol=1e-5)

    def test_vacuum_zero_thickness_is_star_identity(self, device):
        """HomogeneousLayer(ε=1, d=0) → S_p = star_identity (exp(0)=I)."""
        solver, *_, Nh = _make_solver(device)
        S = solver.solve(_hom(1.0, 0.0, device))
        assert _dense_is_star_id(S.to_dense(Nh))

    def test_non_vacuum_has_transmission(self, device):
        """For ε≠1 and d>0, both off-diagonal S blocks must be non-zero."""
        solver, *_, Nh = _make_solver(device)
        M = solver.solve(_hom(2.5, 0.3, device)).to_dense(Nh)
        N = M.shape[-1] // 2
        assert M[..., :N, N:].abs().max().item() > 1e-6
        assert M[..., N:, :N].abs().max().item() > 1e-6

    def test_output_device(self, device):
        solver, *_ = _make_solver(device)
        S = solver.solve(_hom(2.5, 0.3, device))
        assert _get_leaf(S).data.device.type == device


# ---------------------------------------------------------------------------
# _patterned
# ---------------------------------------------------------------------------

class TestLayerSolverPatterned:

    def test_returns_block2x2_like(self, device):
        solver, *_ = _make_solver(device)
        S = solver.solve(_pat(2.5, 1.0, 0.3, _checkerboard(device), device))
        assert _is_block2x2_like(S)

    def test_no_nan(self, device):
        solver, *_, Nh = _make_solver(device)
        S = solver.solve(_pat(2.5, 1.0, 0.3, _checkerboard(device), device))
        assert not torch.isnan(S.to_dense(Nh)).any()

    def test_non_vacuum_has_transmission(self, device):
        """For ε≠1 and d>0, the transmission block must be non-zero."""
        solver, *_, Nh = _make_solver(device)
        M = solver.solve(_pat(4.0, 1.0, 0.3, _checkerboard(device), device)).to_dense(Nh)
        N = M.shape[-1] // 2
        assert M[..., :N, N:].abs().max().item() > 1e-6

    def test_non_vacuum_has_reflection(self, device):
        """For ε≠1, the reflection block must be non-zero."""
        solver, *_, Nh = _make_solver(device)
        M = solver.solve(_pat(4.0, 1.0, 0.3, _checkerboard(device), device)).to_dense(Nh)
        N = M.shape[-1] // 2
        assert M[..., :N, :N].abs().max().item() > 1e-6

    def test_output_device(self, device):
        solver, *_ = _make_solver(device)
        S = solver.solve(_pat(2.5, 1.0, 0.3, _checkerboard(device), device))
        assert _get_leaf(S).data.device.type == device


# ---------------------------------------------------------------------------
# _medium
# ---------------------------------------------------------------------------

class TestLayerSolverMedium:

    def test_returns_block2x2_like(self, device):
        solver, *_ = _make_solver(device)
        assert _is_block2x2_like(solver.solve(_medium(2.5, device)))

    def test_no_nan(self, device):
        solver, *_, Nh = _make_solver(device)
        M = solver.solve(_medium(2.5, device)).to_dense(Nh)
        assert not torch.isnan(M).any()

    def test_vacuum_medium_is_star_identity(self, device):
        """MediumSpec(ε=1) in vacuum background → S_boundary = star_identity."""
        solver, *_, Nh = _make_solver(device)
        S = solver.solve(_medium(1.0, device))
        assert _dense_is_star_id(S.to_dense(Nh))

    def test_non_vacuum_has_reflection(self, device):
        """For ε≠1 the reflection block must be non-zero."""
        solver, *_, Nh = _make_solver(device)
        M = solver.solve(_medium(4.0, device)).to_dense(Nh)
        N = M.shape[-1] // 2
        assert M[..., :N, :N].abs().max().item() > 1e-6

    def test_left_right_give_different_s_matrices(self, device):
        """left=True and left=False must yield different S-matrices for ε≠1."""
        solver, *_, Nh = _make_solver(device)
        med = _medium(4.0, device)
        S_L = solver._medium(med, left=True)
        S_R = solver._medium(med, left=False)
        diff = (S_L.to_dense(Nh) - S_R.to_dense(Nh)).abs().max().item()
        assert diff > 1e-6

    def test_output_device(self, device):
        solver, *_ = _make_solver(device)
        S = solver.solve(_medium(2.5, device))
        assert _get_leaf(S).data.device.type == device
