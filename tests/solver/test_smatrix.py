# tests/solver/test_smatrix.py
# Tests for metarcwa.solver.smatrix: S_boundary, S_prop, S_layer.

import pytest
import torch
from torch.testing import assert_close

from metarcwa.solver.smatrix import S_boundary, S_prop, S_layer
from metarcwa.solver.blockmatrix import Block, Block2x2
from metarcwa.solver.modesolver.homogeneous import homogeneous_modes


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
        """W=I, V=V0 → S_in=star_identity → S_layer = S_prop."""
        W0, V0, W, V, lam, d, wvl = self._inputs(device)
        S_l = S_layer(W0, V0, W, V, lam, d, wvl)
        S_p = S_prop(lam, wvl, d)
        assert_close(S_l.b.a.data, S_p.b.a.data, atol=1e-8, rtol=1e-8)
        assert_close(S_l.b.d.data, S_p.b.d.data, atol=1e-8, rtol=1e-8)

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
