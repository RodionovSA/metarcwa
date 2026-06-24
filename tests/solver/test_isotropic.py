# tests/solver/test_isotropic.py
# Tests for metarcwa.solver.modesolver.isotropic: P and Q operators for
# isotropic patterned layers — structure, known values, and homogeneous limit.

import pytest
import torch
from torch.testing import assert_close

from metarcwa.solver.modesolver.isotropic import (
    compute_Q0,
    compute_A,
    compute_Qfact,
    compute_Q,
    compute_P,
    compute_isotropic,
)
from metarcwa.solver.blockmatrix import Block, Block2x2
from metarcwa.solver.modesolver.homogeneous import homogeneous_Q


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

Nh = 5   # number of Fourier harmonics (odd, avoids accidental symmetry)


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
    """Parametrize each test over CPU and (when available) CUDA."""
    return request.param


def _kxy(Nh: int = Nh, device: str = "cpu") -> tuple[Block, Block]:
    """Return a well-conditioned (Kx, Ky) DIAG Block pair, shape [Nh]."""
    g = torch.Generator()
    g.manual_seed(7)
    kx = (0.3 * torch.randn(Nh, dtype=torch.float64, generator=g)).to(device)
    ky = (0.3 * torch.randn(Nh, dtype=torch.float64, generator=g)).to(device)
    return Block(Block.DIAG, kx), Block(Block.DIAG, ky)


def _eps_conv(eps_val: float = 2.0, Nh: int = Nh, device: str = "cpu") -> Block:
    """Return eps_val * eye(Nh) as a DENSE Block (mimics a uniform medium)."""
    data = eps_val * torch.eye(Nh, dtype=torch.float64, device=device)
    return Block(Block.DENSE, data)


def _uniform_grid(eps_val: float = 2.0, Ny: int = 8, Nx: int = 8,
                  device: str = "cpu") -> torch.Tensor:
    """Return a constant [1, Ny, Nx] permittivity grid."""
    return torch.full((1, Ny, Nx), eps_val, dtype=torch.float64, device=device)


def _harmonic_indices(Nh_half: int = 1, device: str = "cpu"):
    """Return (m_flat, n_flat) integer index arrays for a (2M+1)-harmonic grid."""
    idx = torch.arange(-Nh_half, Nh_half + 1, device=device)
    m_flat = idx.repeat(2 * Nh_half + 1)
    n_flat = idx.repeat_interleave(2 * Nh_half + 1)
    return m_flat, n_flat


class MockTVF:
    """Minimal TVF stand-in: returns Tx = ones, Ty = zeros for any field."""

    def compute(self, field: torch.Tensor):
        Tx = torch.ones_like(field, dtype=torch.complex128)
        Ty = torch.zeros_like(field, dtype=torch.complex128)
        return Tx, Ty


# ---------------------------------------------------------------------------
# compute_Q0
# ---------------------------------------------------------------------------

class TestComputeQ0:

    def test_returns_block2x2(self, device):
        Kx, Ky = _kxy(device=device)
        eps = _eps_conv(device=device)
        assert isinstance(compute_Q0(Kx, Ky, eps), Block2x2)

    def test_block_values_simple(self, device):
        """Single-harmonic case: kx=1, ky=0, eps=2.
           a = 0,  b = -1,  c = 2,  d = 0."""
        Kx  = Block(Block.DIAG, torch.tensor([1.0], dtype=torch.float64, device=device))
        Ky  = Block(Block.DIAG, torch.tensor([0.0], dtype=torch.float64, device=device))
        eps = Block(Block.DENSE, torch.tensor([[2.0]], dtype=torch.float64, device=device))
        Q0  = compute_Q0(Kx, Ky, eps)

        assert_close(Q0.a.to(Block.DENSE, 1).data, torch.tensor([[0.0]], dtype=torch.float64, device=device))
        assert_close(Q0.b.to(Block.DENSE, 1).data, torch.tensor([[-1.0]], dtype=torch.float64, device=device))
        assert_close(Q0.c.to(Block.DENSE, 1).data, torch.tensor([[2.0]], dtype=torch.float64, device=device))
        assert_close(Q0.d.to(Block.DENSE, 1).data, torch.tensor([[0.0]], dtype=torch.float64, device=device))

    def test_symmetry_a_eq_neg_d(self, device):
        """Q0 satisfies a = -d for any inputs (structural constraint)."""
        Kx, Ky = _kxy(device=device)
        eps = _eps_conv(device=device)
        Q0  = compute_Q0(Kx, Ky, eps)
        assert_close(Q0.a.to(Block.DENSE, Nh).data,
                     (-Q0.d).to(Block.DENSE, Nh).data, atol=1e-12, rtol=0)

    def test_output_device(self, device):
        Kx, Ky = _kxy(device=device)
        eps = _eps_conv(device=device)
        Q0  = compute_Q0(Kx, Ky, eps)
        for entry in (Q0.a, Q0.b, Q0.c, Q0.d):
            assert entry.data.device.type == device


# ---------------------------------------------------------------------------
# compute_P
# ---------------------------------------------------------------------------

class TestComputeP:

    def test_returns_block2x2(self, device):
        Kx, Ky = _kxy(device=device)
        eps = _eps_conv(device=device)
        assert isinstance(compute_P(Kx, Ky, eps), Block2x2)

    def test_block_values_simple(self, device):
        """Single-harmonic: kx=1, ky=0, eps=4.
           a = 0,  b = -0.75,  c = 1,  d = 0."""
        Kx  = Block(Block.DIAG, torch.tensor([1.0], dtype=torch.float64, device=device))
        Ky  = Block(Block.DIAG, torch.tensor([0.0], dtype=torch.float64, device=device))
        eps = Block(Block.DENSE, torch.tensor([[4.0]], dtype=torch.float64, device=device))
        P   = compute_P(Kx, Ky, eps)

        assert_close(P.a.to(Block.DENSE, 1).data,
                     torch.tensor([[0.0]], dtype=torch.float64, device=device), atol=1e-12, rtol=0)
        assert_close(P.b.to(Block.DENSE, 1).data,
                     torch.tensor([[-0.75]], dtype=torch.float64, device=device), atol=1e-12, rtol=0)
        assert_close(P.c.to(Block.DENSE, 1).data,
                     torch.tensor([[1.0]], dtype=torch.float64, device=device), atol=1e-12, rtol=0)
        assert_close(P.d.to(Block.DENSE, 1).data,
                     torch.tensor([[0.0]], dtype=torch.float64, device=device), atol=1e-12, rtol=0)

    def test_zero_wavevectors_gives_identity_blocks(self, device):
        """At kx=ky=0: a=0, b=-I, c=I, d=0."""
        n   = 3
        Kx  = Block(Block.DIAG, torch.zeros(n, dtype=torch.float64, device=device))
        Ky  = Block(Block.DIAG, torch.zeros(n, dtype=torch.float64, device=device))
        eps = _eps_conv(2.0, Nh=n, device=device)
        P   = compute_P(Kx, Ky, eps)

        eye = torch.eye(n, dtype=torch.float64, device=device)
        assert_close(P.a.to(Block.DENSE, n).data, torch.zeros(n, n, dtype=torch.float64, device=device), atol=1e-12, rtol=0)
        assert_close(P.b.to(Block.DENSE, n).data, -eye, atol=1e-12, rtol=0)
        assert_close(P.c.to(Block.DENSE, n).data,  eye, atol=1e-12, rtol=0)
        assert_close(P.d.to(Block.DENSE, n).data, torch.zeros(n, n, dtype=torch.float64, device=device), atol=1e-12, rtol=0)

    def test_output_device(self, device):
        Kx, Ky = _kxy(device=device)
        eps = _eps_conv(device=device)
        P   = compute_P(Kx, Ky, eps)
        for entry in (P.a, P.b, P.c, P.d):
            assert entry.data.device.type == device


# ---------------------------------------------------------------------------
# compute_Q
# ---------------------------------------------------------------------------

class TestComputeQ:

    def test_no_A_returns_Q0(self, device):
        """compute_Q without A blocks must equal compute_Q0 entry-by-entry."""
        Kx, Ky = _kxy(device=device)
        eps = _eps_conv(device=device)
        Q0 = compute_Q0(Kx, Ky, eps)
        Q  = compute_Q(Kx, Ky, eps)
        for ref, got in ((Q0.a, Q.a), (Q0.b, Q.b), (Q0.c, Q.c), (Q0.d, Q.d)):
            assert_close(ref.to(Block.DENSE, Nh).data,
                         got.to(Block.DENSE, Nh).data, atol=1e-12, rtol=0)

    def test_partial_none_returns_Q0(self, device):
        """If any A block is None, the result must still equal Q0."""
        Kx, Ky = _kxy(device=device)
        eps = _eps_conv(device=device)
        dummy = _eps_conv(device=device)
        Q0 = compute_Q0(Kx, Ky, eps)
        Q  = compute_Q(Kx, Ky, eps, Axx=dummy, Axy=None, Ayx=dummy, Ayy=dummy)
        assert_close(Q0.a.to(Block.DENSE, Nh).data,
                     Q.a.to(Block.DENSE, Nh).data, atol=1e-12, rtol=0)

    def test_with_A_blocks_differs_from_Q0(self, device):
        """Non-trivial A blocks must produce a result that differs from Q0."""
        Kx, Ky = _kxy(device=device)
        eps    = _eps_conv(3.0, device=device)
        A_blk  = Block(Block.DENSE, (0.5 * torch.eye(Nh, dtype=torch.float64)).to(device))
        Q0     = compute_Q0(Kx, Ky, eps)
        Q      = compute_Q(Kx, Ky, eps, Axx=A_blk, Axy=A_blk, Ayx=A_blk, Ayy=A_blk)
        # At least one entry should differ
        diff = (Q.a.to(Block.DENSE, Nh).data - Q0.a.to(Block.DENSE, Nh).data).abs().max()
        assert diff > 1e-10


# ---------------------------------------------------------------------------
# compute_A (MockTVF)
# ---------------------------------------------------------------------------

class TestComputeA:

    def test_returns_four_blocks(self, device):
        grid = _uniform_grid(device=device)
        m, n = _harmonic_indices(device=device)
        result = compute_A(grid, m, n, MockTVF())
        assert len(result) == 4
        assert all(isinstance(b, Block) for b in result)

    def test_blocks_are_dense(self, device):
        grid = _uniform_grid(device=device)
        m, n = _harmonic_indices(device=device)
        for b in compute_A(grid, m, n, MockTVF()):
            assert b.kind == Block.DENSE

    def test_shape(self, device):
        """Each block should have last two dims [Nh, Nh] where Nh = (2*1+1)^2 = 9."""
        grid = _uniform_grid(device=device)
        m, n = _harmonic_indices(Nh_half=1, device=device)
        Nh_actual = m.shape[0]
        for b in compute_A(grid, m, n, MockTVF()):
            assert b.data.shape[-2:] == (Nh_actual, Nh_actual)

    def test_mock_tvf_tx_ones_ty_zeros(self, device):
        """With MockTVF (Tx=1, Ty=0): axx component = |Ty|² = 0 → Axx is all-zero."""
        grid = _uniform_grid(device=device)
        m, n = _harmonic_indices(Nh_half=1, device=device)
        Axx, Axy, Ayx, Ayy = compute_A(grid, m, n, MockTVF())
        # Ty = zeros → Ty_fft = zeros → axx = |Ty_fft|² = 0 → Axx all-zero
        assert_close(Axx.data.abs().max(),
                     torch.tensor(0.0, dtype=torch.float64, device=device),
                     atol=1e-10, rtol=0)
        # cross-terms also zero since Ty=0
        assert_close(Axy.data.abs().max(),
                     torch.tensor(0.0, dtype=torch.float64, device=device),
                     atol=1e-10, rtol=0)


# ---------------------------------------------------------------------------
# compute_isotropic
# ---------------------------------------------------------------------------

class TestComputeIsotropic:

    def test_returns_tuple_of_block2x2(self, device):
        grid = _uniform_grid(device=device)
        m, n = _harmonic_indices(device=device)
        kx = torch.zeros(m.shape[0], dtype=torch.float64, device=device)
        ky = torch.zeros(m.shape[0], dtype=torch.float64, device=device)
        result = compute_isotropic(grid, m, n, kx, ky)
        assert isinstance(result, tuple) and len(result) == 2
        P, Q = result
        assert isinstance(P, Block2x2)
        assert isinstance(Q, Block2x2)

    def test_uniform_eps_Q_matches_homogeneous_Q(self, device):
        """Uniform epsilon_grid → epsilon_conv = eps*I → Q should match
        homogeneous_Q numerically (different kind: DENSE vs DIAG)."""
        eps_val = 2.5
        Nh_half = 1
        m, n = _harmonic_indices(Nh_half, device=device)
        Nh_actual = m.shape[0]
        g = torch.Generator(); g.manual_seed(3)
        kx = (0.2 * torch.randn(Nh_actual, dtype=torch.float64, generator=g)).to(device)
        ky = (0.2 * torch.randn(Nh_actual, dtype=torch.float64, generator=g)).to(device)
        grid = _uniform_grid(eps_val, device=device)

        _, Q_iso  = compute_isotropic(grid, m, n, kx, ky)
        # Use complex eps so homogeneous_Q produces the same dtype as Q_iso
        eps_tensor = torch.tensor([[eps_val + 0j]], dtype=torch.complex128, device=device)
        Q_hom = homogeneous_Q(eps_tensor, kx.to(torch.complex128).unsqueeze(0),
                              ky.to(torch.complex128).unsqueeze(0))

        for iso_entry, hom_entry in (
            (Q_iso.a, Q_hom.a), (Q_iso.b, Q_hom.b),
            (Q_iso.c, Q_hom.c), (Q_iso.d, Q_hom.d),
        ):
            iso_dense = iso_entry.to(Block.DENSE, Nh_actual).data.squeeze(0).to(torch.complex128)
            hom_dense = hom_entry.to(Block.DENSE, Nh_actual).data.squeeze(0).to(torch.complex128)
            assert_close(iso_dense, hom_dense, atol=1e-10, rtol=1e-10)

    def test_uniform_eps_P_zero_kx_ky(self, device):
        """At kx=ky=0, uniform eps: a=0, b=-I, c=I, d=0."""
        eps_val = 3.0
        Nh_half = 1
        m, n = _harmonic_indices(Nh_half, device=device)
        Nh_actual = m.shape[0]
        kx = torch.zeros(Nh_actual, dtype=torch.float64, device=device)
        ky = torch.zeros(Nh_actual, dtype=torch.float64, device=device)
        grid = _uniform_grid(eps_val, device=device)

        P, _ = compute_isotropic(grid, m, n, kx, ky)
        # P entries are complex128 (epsilon_conv is complex); compare against complex identity
        eye  = torch.eye(Nh_actual, dtype=torch.complex128, device=device)
        zero = torch.zeros(Nh_actual, Nh_actual, dtype=torch.complex128, device=device)

        assert_close(P.a.to(Block.DENSE, Nh_actual).data.squeeze(0), zero, atol=1e-12, rtol=0)
        assert_close(P.b.to(Block.DENSE, Nh_actual).data.squeeze(0), -eye, atol=1e-12, rtol=0)
        assert_close(P.c.to(Block.DENSE, Nh_actual).data.squeeze(0),  eye, atol=1e-12, rtol=0)
        assert_close(P.d.to(Block.DENSE, Nh_actual).data.squeeze(0), zero, atol=1e-12, rtol=0)

    def test_with_mock_tvf_returns_different_Q(self, device):
        """Q with MockTVF must differ from Q without TVF for non-trivial inputs."""
        Nh_half = 1
        m, n = _harmonic_indices(Nh_half, device=device)
        Nh_actual = m.shape[0]
        g = torch.Generator(); g.manual_seed(11)
        kx = (0.25 * torch.randn(Nh_actual, dtype=torch.float64, generator=g)).to(device)
        ky = (0.25 * torch.randn(Nh_actual, dtype=torch.float64, generator=g)).to(device)
        # Use a non-uniform grid so the A blocks are non-trivial
        grid = (torch.rand(1, 8, 8, dtype=torch.float64, generator=g) + 1.0).to(device)

        _, Q_plain = compute_isotropic(grid, m, n, kx, ky, tvf=None)
        _, Q_tvf   = compute_isotropic(grid, m, n, kx, ky, tvf=MockTVF())
        # MockTVF has Tx=1, Ty=0 → Ayy≠0, so Qfact.b = eps@Ayy - eps⁻¹@Ayy ≠ 0
        diff = (Q_tvf.b.to(Block.DENSE, Nh_actual).data
                - Q_plain.b.to(Block.DENSE, Nh_actual).data).abs().max()
        assert diff > 1e-10
