# tests/solver/test_homogeneous.py
# Tests for metarcwa.solver.homogeneous: kz, Q, modes — shapes, physics,
# device portability (CPU + CUDA), and autograd flow.

import warnings

import pytest
import torch

from metarcwa.solver.homogeneous import homogeneous_kz, homogeneous_Q, homogeneous_modes
from metarcwa.solver.blockmatrix import Block, Block2x2

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

N_WL = 2   # number of wavelengths in batched tests
Nh   = 5   # number of Fourier harmonics (odd to avoid accidental symmetry)


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


def _make_inputs(
    device: str,
    N_wl: int = N_WL,
    Nh: int = Nh,
    *,
    requires_grad: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return ``(epsilon, kx, ky)`` well away from any grazing singularity.

    ``epsilon`` is complex128, shape ``(N_wl, 1)``.
    ``kx``, ``ky``  are float64,    shape ``(N_wl, Nh)``.

    With eps ~ U(3, 5) and |kx|, |ky| <= 0.3 the maximum
    kx² + ky² ≈ 0.09 * Nh  <<  3 = min(eps), so all modes are propagating.
    """
    g = torch.Generator()
    g.manual_seed(42)
    eps = (3.0 + 2.0 * torch.rand(N_wl, 1, generator=g)).to(torch.complex128).to(device)
    kx  = (0.3  * torch.randn(N_wl, Nh, generator=g)).to(torch.float64).to(device)
    ky  = (0.3  * torch.randn(N_wl, Nh, generator=g)).to(torch.float64).to(device)
    if requires_grad:
        eps.requires_grad_(True)
        kx.requires_grad_(True)
        ky.requires_grad_(True)
    return eps, kx, ky


def _block2x2_entries(m: Block2x2):
    return (m.a, m.b, m.c, m.d)


# ---------------------------------------------------------------------------
# homogeneous_kz
# ---------------------------------------------------------------------------

class TestHomogeneousKz:

    def test_output_shape(self, device):
        eps, kx, ky = _make_inputs(device)
        kz = homogeneous_kz(eps, kx, ky)
        assert kz.shape == (N_WL, 2 * Nh)

    def test_kz_squared_matches_dispersion_relation(self, device):
        """kz² == eps - kx² - ky² regardless of which branch is chosen."""
        eps, kx, ky = _make_inputs(device)
        kz = homogeneous_kz(eps, kx, ky)
        # Each harmonic contributes one forward and one backward mode (duplicated block).
        block    = eps - kx**2 - ky**2                           # (N_wl, Nh)
        expected = torch.cat([block, block], dim=-1)             # (N_wl, 2Nh)
        assert torch.allclose(kz**2, expected, atol=1e-10)

    def test_propagating_mode_has_positive_real_kz(self, device):
        """kx = ky = 0, eps = 2  →  kz = sqrt(2): purely real and positive."""
        eps = torch.tensor([[2.0 + 0j]], dtype=torch.complex128, device=device)
        kx  = torch.zeros(1, 1, dtype=torch.float64, device=device)
        ky  = torch.zeros(1, 1, dtype=torch.float64, device=device)
        kz  = homogeneous_kz(eps, kx, ky, forward="positive")
        assert (kz.real > 0).all()
        assert torch.all(kz.imag.abs() < 1e-10)

    def test_evanescent_mode_has_positive_imag_kz(self, device):
        """kx = 2, ky = 0, eps = 1  →  kz² = -3, Im(kz) > 0."""
        eps = torch.tensor([[1.0 + 0j]], dtype=torch.complex128, device=device)
        kx  = torch.tensor([[2.0]],      dtype=torch.float64,    device=device)
        ky  = torch.zeros(1, 1,          dtype=torch.float64,    device=device)
        kz  = homogeneous_kz(eps, kx, ky, forward="positive")
        assert (kz.imag > 0).all()
        assert torch.all(kz.real.abs() < 1e-10)

    def test_negative_branch_negates_positive_branch(self, device):
        eps, kx, ky = _make_inputs(device)
        kz_pos = homogeneous_kz(eps, kx, ky, forward="positive")
        kz_neg = homogeneous_kz(eps, kx, ky, forward="negative")
        assert torch.allclose(kz_pos + kz_neg, torch.zeros_like(kz_pos), atol=1e-12)

    def test_invalid_forward_raises_value_error(self, device):
        eps, kx, ky = _make_inputs(device)
        with pytest.raises(ValueError, match="forward"):
            homogeneous_kz(eps, kx, ky, forward="sideways")

    def test_output_device(self, device):
        eps, kx, ky = _make_inputs(device)
        kz = homogeneous_kz(eps, kx, ky)
        assert kz.device.type == device


# ---------------------------------------------------------------------------
# homogeneous_Q
# ---------------------------------------------------------------------------

class TestHomogeneousQ:

    def test_returns_block2x2_of_diag(self, device):
        """Q is a Block2x2; every entry is Block(DIAG, …) of shape (N_wl, Nh)."""
        eps, kx, ky = _make_inputs(device)
        Q = homogeneous_Q(eps, kx, ky)
        assert isinstance(Q, Block2x2)
        for entry in _block2x2_entries(Q):
            assert entry.kind == Block.DIAG
            assert entry.data.shape == (N_WL, Nh)

    def test_block_diagonal_values(self, device):
        """Each of the four diagonal blocks holds the correct expression."""
        eps, kx, ky = _make_inputs(device)
        Q  = homogeneous_Q(eps, kx, ky)
        dt = Q.a.data.dtype
        assert torch.allclose(Q.a.data, -(kx * ky).to(dt), atol=1e-12)
        assert torch.allclose(Q.b.data,  (kx**2 - eps),    atol=1e-12)
        assert torch.allclose(Q.c.data,  (eps - ky**2),    atol=1e-12)
        assert torch.allclose(Q.d.data,  (ky * kx).to(dt), atol=1e-12)

    def test_output_device(self, device):
        eps, kx, ky = _make_inputs(device)
        Q = homogeneous_Q(eps, kx, ky)
        for entry in _block2x2_entries(Q):
            assert entry.data.device.type == device


# ---------------------------------------------------------------------------
# homogeneous_modes
# ---------------------------------------------------------------------------

class TestHomogeneousModes:

    def test_output_shapes(self, device):
        """lam is [..., 2Nh]; V is Block2x2 with DIAG entries of shape [..., Nh]."""
        eps, kx, ky = _make_inputs(device)
        lam, V = homogeneous_modes(eps, kx, ky)
        assert lam.shape == (N_WL, 2 * Nh)
        assert isinstance(V, Block2x2)
        for entry in _block2x2_entries(V):
            assert entry.kind == Block.DIAG
            assert entry.data.shape == (N_WL, Nh)

    def test_W_is_identity(self, device):
        """For a homogeneous layer W = I (not returned; harmonics are eigenmodes).
        Verified indirectly: I @ V == V must hold."""
        eps, kx, ky = _make_inputs(device)
        _, V = homogeneous_modes(eps, kx, ky)
        kw  = dict(device=device, dtype=V.a.data.dtype)
        eye = Block2x2(Block.eye(**kw), Block.zeros(**kw), Block.zeros(**kw), Block.eye(**kw))
        res = eye @ V
        for ref, got in zip(_block2x2_entries(V), _block2x2_entries(res)):
            assert torch.allclose(ref.data, got.data, atol=1e-12)

    def test_lam_equals_1j_times_kz(self, device):
        """lam = 1j * kz by definition; must round-trip through homogeneous_kz."""
        eps, kx, ky = _make_inputs(device)
        lam, _ = homogeneous_modes(eps, kx, ky)
        kz = homogeneous_kz(eps, kx, ky)
        assert torch.allclose(lam, 1j * kz, atol=1e-12)

    def test_V_is_Q_scaled_by_inverse_lam(self, device):
        """V = Q0 @ diag(1/lam): left Nh columns scaled by lam[:Nh], right by lam[Nh:]."""
        eps, kx, ky = _make_inputs(device)
        lam, V = homogeneous_modes(eps, kx, ky)
        Q0 = homogeneous_Q(eps, kx, ky)
        lam_l = lam[..., :Nh]
        lam_r = lam[..., Nh:]
        assert torch.allclose(V.a.data, Q0.a.data / lam_l, atol=1e-10)
        assert torch.allclose(V.b.data, Q0.b.data / lam_r, atol=1e-10)
        assert torch.allclose(V.c.data, Q0.c.data / lam_l, atol=1e-10)
        assert torch.allclose(V.d.data, Q0.d.data / lam_r, atol=1e-10)

    def test_grazing_mode_emits_runtime_warning(self, device):
        """lam = 0 at kx² + ky² == eps (exact grazing) must emit RuntimeWarning."""
        eps = torch.tensor([[1.0 + 0j]], dtype=torch.complex128, device=device)
        kx  = torch.tensor([[1.0]],      dtype=torch.float64,    device=device)
        ky  = torch.zeros(1, 1,          dtype=torch.float64,    device=device)
        with pytest.warns(RuntimeWarning, match="grazing"):
            homogeneous_modes(eps, kx, ky)

    def test_no_warning_for_non_grazing_inputs(self, device):
        """No RuntimeWarning must be emitted for well-conditioned inputs."""
        eps, kx, ky = _make_inputs(device)
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            homogeneous_modes(eps, kx, ky)  # must not raise

    def test_outputs_on_correct_device(self, device):
        eps, kx, ky = _make_inputs(device)
        lam, V = homogeneous_modes(eps, kx, ky)
        assert lam.device.type == device
        for entry in _block2x2_entries(V):
            assert entry.data.device.type == device


# ---------------------------------------------------------------------------
# Gradient flow
# ---------------------------------------------------------------------------

class TestGradientFlow:
    """
    Autograd through kx / ky (real) and epsilon (complex — Wirtinger gradient)
    for all three public functions.  Inputs are kept well away from grazing so
    that 1/lam never produces NaN in V.
    """

    @staticmethod
    def _assert_finite_grads(*tensors: torch.Tensor) -> None:
        for t in tensors:
            assert t.grad is not None,                 "gradient is None"
            assert t.grad.shape == t.shape,            "gradient shape mismatch"
            # .abs() handles both real and complex gradients uniformly.
            assert torch.isfinite(t.grad.abs()).all(),  "gradient contains non-finite values"

    def test_kz_gradients(self, device):
        eps, kx, ky = _make_inputs(device, requires_grad=True)
        homogeneous_kz(eps, kx, ky).abs().sum().backward()
        self._assert_finite_grads(eps, kx, ky)

    def test_Q_gradients(self, device):
        eps, kx, ky = _make_inputs(device, requires_grad=True)
        Q = homogeneous_Q(eps, kx, ky)
        loss = sum(e.data.abs().sum() for e in _block2x2_entries(Q))
        loss.backward()
        self._assert_finite_grads(eps, kx, ky)

    def test_modes_lam_gradients(self, device):
        eps, kx, ky = _make_inputs(device, requires_grad=True)
        lam, _ = homogeneous_modes(eps, kx, ky)
        (lam.abs() ** 2).sum().backward()
        self._assert_finite_grads(eps, kx, ky)

    def test_modes_V_gradients(self, device):
        """Gradient flows through both Q0 and the 1/lam column scaling."""
        eps, kx, ky = _make_inputs(device, requires_grad=True)
        _, V = homogeneous_modes(eps, kx, ky)
        loss = sum(e.data.abs().sum() for e in _block2x2_entries(V))
        loss.backward()
        self._assert_finite_grads(eps, kx, ky)
