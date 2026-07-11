# tests/solver/test_smatrix.py
# Tests for metarcwa.solver.smatrix: S_boundary, S_prop, S_layer.

import pytest
import torch
from torch.testing import assert_close

from metarcwa.solver.smatrix import S_boundary, S_prop, S_layer
from metarcwa.solver.blockmatrix import Block, Block2x2
from metarcwa.solver.layersolver.homogeneous import homogeneous_modes
from metarcwa.solver.layersolver.isotropic import compute_isotropic
from metarcwa.solver.layersolver.eigsolver import eigsolver
from metarcwa.solver.harmonics import harmonic_index_map, compute_kxy


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

Nh = 5   # harmonics per polarisation block


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


def _scalar_modes(val: float, device: str = "cpu") -> Block2x2:
    """Block2x2 with all-SCALAR entries equal to val (represents val·I)."""
    s = Block(Block.SCALAR, torch.tensor(val, dtype=torch.float64, device=device))
    z = Block.zeros(dtype=torch.float64, device=device)
    return Block2x2(s, z, z, s)


def _diag_modes(Nh: int, val: float, device: str = "cpu") -> Block2x2:
    """Block2x2 with DIAG entries: [[val·I, 0], [0, val·I]] of shape [Nh]."""
    d = Block(Block.DIAG, torch.full((Nh,), val, dtype=torch.complex128, device=device))
    z = Block.zeros(dtype=torch.complex128, device=device)
    return Block2x2(d, z, z, d)


def _hom_lam_V(eps: float, Nh: int, device: str = "cpu"):
    """Return (lam, V) from homogeneous_modes with well-conditioned kx, ky."""
    g = torch.Generator(); g.manual_seed(5)
    kx = (0.2 * torch.randn(Nh, dtype=torch.float64, generator=g)).to(device)
    ky = (0.2 * torch.randn(Nh, dtype=torch.float64, generator=g)).to(device)
    eps_t = torch.tensor([[eps + 0j]], dtype=torch.complex128, device=device)
    return homogeneous_modes(eps_t, kx.unsqueeze(0), ky.unsqueeze(0))


def _is_leaf(entry) -> bool:
    """True when entry is a leaf Block (has .data but no .a/.b sub-entries)."""
    return hasattr(entry, 'data') and not hasattr(entry, 'a')


def _is_zero_like(entry, atol: float = 1e-10) -> bool:
    """True if all data in a Block or Block2x2 (at any depth) is ~0."""
    if _is_leaf(entry):
        return entry.data.abs().max().item() < atol
    return all(_is_zero_like(e, atol) for e in (entry.a, entry.b, entry.c, entry.d))


def _is_identity_like(entry, atol: float = 1e-10) -> bool:
    """True if entry acts as identity (SCALAR data = 1, or DIAG/DENSE = I)."""
    if _is_leaf(entry):
        if entry.kind == Block.SCALAR:
            return (entry.data - 1).abs().max().item() < atol
        n = entry.n
        return (entry.to(Block.DENSE, n).data - torch.eye(n, dtype=entry.data.dtype,
                device=entry.data.device)).abs().max().item() < atol
    # Block2x2: a=I, b=0, c=0, d=I
    return (_is_identity_like(entry.a, atol) and _is_zero_like(entry.b, atol) and
            _is_zero_like(entry.c, atol) and _is_identity_like(entry.d, atol))


# ---------------------------------------------------------------------------
# S_boundary
# ---------------------------------------------------------------------------

class TestSBoundary:

    def test_returns_block2x2(self, device):
        W = _scalar_modes(1.0, device)
        V = _scalar_modes(2.0, device)
        assert isinstance(S_boundary(W, V, W, V), Block2x2)

    def test_same_medium_a_is_zero(self, device):
        """WL=WR, VL=VR → S.a (reflection) is zero."""
        W = _scalar_modes(1.0, device)
        V = _scalar_modes(3.0, device)
        S = S_boundary(W, V, W, V)
        assert _is_zero_like(S.a)

    def test_same_medium_b_is_identity(self, device):
        """WL=WR, VL=VR → S.b (transmission from right) is identity."""
        W = _scalar_modes(1.0, device)
        V = _scalar_modes(3.0, device)
        S = S_boundary(W, V, W, V)
        assert _is_identity_like(S.b)

    def test_same_medium_gives_star_identity(self, device):
        """WL=WR, VL=VR → full S = [[0,I],[I,0]] (star-product identity)."""
        W = _diag_modes(Nh, 1.0, device)
        V = _diag_modes(Nh, 2.0, device)
        S = S_boundary(W, V, W, V)
        assert _is_zero_like(S.a)
        assert _is_identity_like(S.b)
        assert _is_identity_like(S.c)
        assert _is_zero_like(S.d)

    def test_output_device(self, device):
        W = _scalar_modes(1.0, device)
        V = _scalar_modes(2.0, device)
        S = S_boundary(W, V, W, V)
        # Check one entry's data recursively
        def _device_of(e):
            if isinstance(e, Block):
                return e.data.device.type
            return _device_of(e.a)
        assert _device_of(S.a) == device


# ---------------------------------------------------------------------------
# S_boundary DIAG fast path (A4/C4)
# ---------------------------------------------------------------------------

def _hom_WV(eps: float, kx: torch.Tensor, ky: torch.Tensor):
    """(W, V) Block2x2 for a homogeneous medium: W=I (SCALAR), V=DIAG."""
    eps_t = torch.tensor([[eps + 0j]], dtype=torch.complex128, device=kx.device)
    _, V = homogeneous_modes(eps_t, kx, ky)
    return V.eye_like(), V


def _pat_WV(eps_solid: float, eps_void: float, pattern: torch.Tensor,
           kx: torch.Tensor, ky: torch.Tensor, m: torch.Tensor, n: torch.Tensor):
    """(W, V) Block2x2 for a patterned layer: both DENSE (via eigsolver)."""
    eps_grid = (eps_solid * pattern[None] + eps_void * (1 - pattern[None])).to(torch.complex128)
    P, Q = compute_isotropic(eps_grid, m, n, kx, ky)
    _, W, V = eigsolver(P, Q)
    return W, V


def _dense_ref(WL, VL, WR, VR, Nh):
    """Dense ground-truth: densify left/right to [4Nh,4Nh] and solve once."""
    left  = Block2x2(WL, -WR, VL, VR)
    right = Block2x2(-WL, WR, VL, VR)
    return torch.linalg.solve(left.to_dense(Nh), right.to_dense(Nh))


class TestSBoundaryDispatch:
    """``S_boundary`` delegates all dispatch logic to ``Block2x2.solve``.

    These tests confirm end-to-end that the three dispatch branches
    (all-SCALAR → Schur, all-DIAG → per-harmonic O(n), any-DENSE → dense)
    produce correct and finite results, and that the efficient per-harmonic
    branch is actually taken for homogeneous boundaries (shape-regression
    guard)."""

    def _harmonics(self, kx0: float, ky0: float, Nh_half: int = 2, device: str = "cpu"):
        a1 = torch.tensor([1.0, 0.0], dtype=torch.float64, device=device)
        a2 = torch.tensor([0.0, 1.0], dtype=torch.float64, device=device)
        kx0_t = torch.tensor([kx0], dtype=torch.float64, device=device)
        ky0_t = torch.tensor([ky0], dtype=torch.float64, device=device)
        m, n = harmonic_index_map(Nh_half, Nh_half, device=device)
        kx, ky = compute_kxy(kx0_t, ky0_t, a1, a2, m, n)
        return kx, ky, m, n

    def test_matches_dense_at_normal_incidence(self, device):
        """Normal incidence: several harmonics have kx*ky=0 → V.d has
        exact-zero diagonal entries — exactly the case that breaks a naive
        Schur-complement path on DIAG blocks.  The per-harmonic solve must
        still match the dense ground truth and produce no nan/inf."""
        kx, ky, m, n = self._harmonics(0.0, 0.0, device=device)
        Nh = kx.shape[-1]
        assert (kx * ky == 0).any(), "test setup must include a kx*ky=0 harmonic"

        WL, VL = _hom_WV(1.0, kx, ky)
        WR, VR = _hom_WV(4.0, kx, ky)
        assert (VL.d.to(Block.DENSE, Nh).data == 0).any()

        S = S_boundary(WL, VL, WR, VR)
        S_d   = S.to_dense(Nh)
        S_ref = _dense_ref(WL, VL, WR, VR, Nh)

        assert not torch.isnan(S_d).any()
        assert not torch.isinf(S_d).any()
        assert_close(S_d, S_ref, atol=1e-8, rtol=1e-8)

    def test_matches_dense_at_oblique_incidence(self, device):
        kx, ky, m, n = self._harmonics(0.13, 0.07, device=device)
        Nh = kx.shape[-1]

        WL, VL = _hom_WV(1.0, kx, ky)
        WR, VR = _hom_WV(4.0, kx, ky)

        assert_close(
            S_boundary(WL, VL, WR, VR).to_dense(Nh),
            _dense_ref(WL, VL, WR, VR, Nh),
            atol=1e-8, rtol=1e-8,
        )

    def test_gradients_flow_through_diag_path(self, device):
        """Gradients through the all-DIAG per-harmonic path must match
        those from a direct dense reference solve."""
        kx, ky, m, n = self._harmonics(0.13, 0.07, device=device)
        Nh = kx.shape[-1]
        eps_L = torch.tensor([[1.0 + 0j]], dtype=torch.complex128, device=device)

        def _grad_via(use_dense: bool):
            eps_R = torch.tensor([[4.0 + 0j]], dtype=torch.complex128,
                                 device=device, requires_grad=True)
            _, VL = homogeneous_modes(eps_L, kx, ky)
            _, VR = homogeneous_modes(eps_R, kx, ky)
            WL, WR = VL.eye_like(), VR.eye_like()
            if use_dense:
                loss = _dense_ref(WL, VL, WR, VR, Nh).abs().sum()
            else:
                loss = S_boundary(WL, VL, WR, VR).to_dense(Nh).abs().sum()
            loss.backward()
            return eps_R.grad.clone()

        assert_close(_grad_via(False), _grad_via(True), atol=1e-8, rtol=1e-8)

    def test_mixed_diag_dense_matches_dense_ref(self, device):
        """One side homogeneous (DIAG), other patterned (DENSE): ``S_boundary``
        must fall through to the dense solve and match the reference."""
        kx, ky, m, n = self._harmonics(0.13, 0.07, device=device)
        Nh = kx.shape[-1]

        WL, VL = _hom_WV(1.0, kx, ky)
        pattern = torch.zeros(8, 8, dtype=torch.float64, device=device)
        pattern[::2, ::2]   = 1.0
        pattern[1::2, 1::2] = 1.0
        WR, VR = _pat_WV(4.0, 1.0, pattern, kx, ky, m, n)

        assert_close(
            S_boundary(WL, VL, WR, VR).to_dense(Nh),
            _dense_ref(WL, VL, WR, VR, Nh),
            atol=1e-8, rtol=1e-8,
        )

    def test_scalar_only_uses_schur_path(self, monkeypatch, device):
        """True all-SCALAR inputs (Nh unknowable) must stay on the Schur path —
        ``torch.linalg.solve`` must not be called at all for this branch."""
        calls = {"n": 0}
        real_solve = torch.linalg.solve

        def counting_solve(a, b):
            calls["n"] += 1
            return real_solve(a, b)

        monkeypatch.setattr(torch.linalg, "solve", counting_solve)
        W = _scalar_modes(1.0, device)
        V = _scalar_modes(2.0, device)
        S_boundary(W, V, W, V)
        assert calls["n"] == 0

    def test_diag_path_solve_shape_regression(self, monkeypatch, device):
        """Regression guard: for all-DIAG inputs the ``torch.linalg.solve``
        call must operate on (4, 4) matrices, not (4Nh, 4Nh) — guards that
        the per-harmonic O(n) path is taken, not a full densification."""
        kx, ky, m, n = self._harmonics(0.13, 0.07, device=device)
        WL, VL = _hom_WV(1.0, kx, ky)
        WR, VR = _hom_WV(4.0, kx, ky)

        shapes = []
        real_solve = torch.linalg.solve

        def capturing_solve(a, b):
            shapes.append(tuple(a.shape[-2:]))
            return real_solve(a, b)

        monkeypatch.setattr(torch.linalg, "solve", capturing_solve)
        S_boundary(WL, VL, WR, VR)
        assert len(shapes) == 1
        assert shapes[0] == (4, 4)


# ---------------------------------------------------------------------------
# S_prop
# ---------------------------------------------------------------------------

class TestSProp:

    def _make_lam(self, Nh: int = Nh, device: str = "cpu") -> tuple:
        """Return (lam, wvl, d) for S_prop tests."""
        lam = torch.linspace(0.1, 0.5, 2 * Nh, dtype=torch.complex128, device=device)
        wvl = torch.tensor(1.0, dtype=torch.float64, device=device)
        d   = torch.tensor(0.5, dtype=torch.float64, device=device)
        return lam, wvl, d

    def test_returns_block2x2(self, device):
        lam, wvl, d = self._make_lam(device=device)
        assert isinstance(S_prop(lam, wvl, d), Block2x2)

    def test_entries_are_block2x2(self, device):
        """After the S_prop fix, each top-level entry must be Block2x2 (not Block)."""
        lam, wvl, d = self._make_lam(device=device)
        S = S_prop(lam, wvl, d)
        for entry in (S.a, S.b, S.c, S.d):
            assert isinstance(entry, Block2x2), (
                f"Expected Block2x2 entry, got {type(entry).__name__}")

    def test_diagonal_entries_are_zero(self, device):
        """S_prop.a and S_prop.d (reflection blocks) must be zero."""
        lam, wvl, d = self._make_lam(device=device)
        S = S_prop(lam, wvl, d)
        assert _is_zero_like(S.a)
        assert _is_zero_like(S.d)

    def test_zero_d_gives_star_identity(self, device):
        """d=0 → exp(0)=1 → S_prop = [[0,I],[I,0]]."""
        lam = torch.ones(2 * Nh, dtype=torch.complex128, device=device)
        wvl = torch.tensor(1.0, dtype=torch.float64, device=device)
        d   = torch.tensor(0.0, dtype=torch.float64, device=device)
        S   = S_prop(lam, wvl, d)
        assert _is_zero_like(S.a)
        assert _is_identity_like(S.b)
        assert _is_identity_like(S.c)
        assert _is_zero_like(S.d)

    def test_off_diagonal_b_matches_exp_lam(self, device):
        """S_prop.b.a data should equal exp(lam[:Nh] * k0 * d)."""
        lam, wvl, d = self._make_lam(device=device)
        S   = S_prop(lam, wvl, d)
        k0  = 2 * torch.pi / wvl
        expected_a = torch.exp(lam[:Nh] * k0 * d)
        expected_d = torch.exp(lam[Nh:] * k0 * d)
        assert_close(S.b.a.data, expected_a, atol=1e-12, rtol=1e-12)
        assert_close(S.b.d.data, expected_d, atol=1e-12, rtol=1e-12)

    def test_output_device(self, device):
        lam, wvl, d = self._make_lam(device=device)
        S = S_prop(lam, wvl, d)
        assert S.b.a.data.device.type == device


# ---------------------------------------------------------------------------
# S_layer
# ---------------------------------------------------------------------------

class TestSLayer:

    def _inputs(self, device: str):
        """Return (W0, V0, W, V, lam, d, wvl) for a homogeneous layer in the same medium."""
        eps = 2.5
        lam, V = _hom_lam_V(eps, Nh, device)
        W   = V.eye_like()           # W=I for homogeneous layer
        V0  = V                      # same medium on both sides
        W0  = V0.eye_like()          # background E-mode matrix = I
        wvl = torch.tensor(1.0, dtype=torch.float64, device=device)
        d   = torch.tensor(0.3, dtype=torch.float64, device=device)
        return W0, V0, W, V, lam, d, wvl

    def test_returns_block2x2(self, device):
        W0, V0, W, V, lam, d, wvl = self._inputs(device)
        assert isinstance(S_layer(W0, V0, W, V, lam, d, wvl), Block2x2)

    def test_same_medium_equals_s_prop(self, device):
        """W=I, V=V0 → S_in=star_identity → S_layer = S_prop.

        Compare via to_dense because S_boundary's leaf kind depends on which
        internal path it dispatches to (DIAG fast path vs. DENSE fallback,
        see TestBoundaryDiagFastPath) while S_prop always returns DIAG
        entries -- to_dense is the representation-agnostic comparison.
        """
        W0, V0, W, V, lam, d, wvl = self._inputs(device)
        S_l = S_layer(W0, V0, W, V, lam, d, wvl)
        S_p = S_prop(lam, wvl, d)
        Nh  = lam.shape[-1] // 2
        assert_close(S_l.b.to_dense(Nh), S_p.b.to_dense(Nh), atol=1e-7, rtol=1e-7)

    def test_zero_d_same_medium_gives_star_identity(self, device):
        """d=0 and same medium → S_prop = star_identity → S_layer = star_identity."""
        W0, V0, W, V, lam, _, wvl = self._inputs(device)
        d_zero = torch.tensor(0.0, dtype=torch.float64, device=device)
        S = S_layer(W0, V0, W, V, lam, d_zero, wvl)
        assert _is_zero_like(S.a)
        assert _is_identity_like(S.b)
        assert _is_identity_like(S.c)
        assert _is_zero_like(S.d)

    def test_star_associativity(self, device):
        """(S_in ⋆ S_p) ⋆ S_out == S_in ⋆ (S_p ⋆ S_out) — verifies star order."""
        W0, V0, W, V, lam, d, wvl = self._inputs(device)
        S_in  = S_boundary(W0, V0, W, V)
        S_p   = S_prop(lam, wvl, d)
        S_out = Block2x2(S_in.d, S_in.c, S_in.b, S_in.a)
        lhs = S_in.star(S_p).star(S_out)
        rhs = S_in.star(S_p.star(S_out))
        assert_close(lhs.b.a.data, rhs.b.a.data, atol=1e-8, rtol=1e-8)
        assert_close(lhs.b.d.data, rhs.b.d.data, atol=1e-8, rtol=1e-8)
        assert_close(lhs.a.a.data, rhs.a.a.data, atol=1e-8, rtol=1e-8)

    def test_output_device(self, device):
        W0, V0, W, V, lam, d, wvl = self._inputs(device)
        S = S_layer(W0, V0, W, V, lam, d, wvl)
        assert S.b.a.data.device.type == device
