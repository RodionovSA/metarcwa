# tests/test_homogeneous.py
# Tests for metarcwa.solver.homogeneous: kz, Q, modes — shapes, physics,
# device portability (CPU + CUDA), and autograd flow.

import warnings

import pytest
import torch

from metarcwa.solver.homogeneous import homogeneous_kz, homogeneous_Q, homogeneous_modes

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

    def test_output_shape(self, device):
        eps, kx, ky = _make_inputs(device)
        Q = homogeneous_Q(eps, kx, ky)
        assert Q.shape == (N_WL, 2 * Nh, 2 * Nh)

    def test_block_diagonal_values(self, device):
        """Each of the four N×N diagonal blocks holds the correct expression."""
        eps, kx, ky = _make_inputs(device)
        Q = homogeneous_Q(eps, kx, ky)
        N = Nh

        def diag(block):
            return torch.diagonal(block, dim1=-2, dim2=-1)

        # Q0 = [[ Kx*Ky ,      eps - Kx^2 ],
        #       [ Ky^2 - eps , -Ky*Kx     ]]
        # Cast real expected values to Q's complex dtype before comparing.
        dt = Q.dtype
        assert torch.allclose(diag(Q[..., :N, :N]), (kx * ky).to(dt),     atol=1e-12)
        assert torch.allclose(diag(Q[..., :N, N:]),  eps - kx**2,          atol=1e-12)
        assert torch.allclose(diag(Q[..., N:, :N]),  ky**2 - eps,          atol=1e-12)
        assert torch.allclose(diag(Q[..., N:, N:]), -(ky * kx).to(dt),    atol=1e-12)

    def test_each_block_is_a_diagonal_matrix(self, device):
        """Off-diagonal entries within every N×N block must be zero."""
        eps, kx, ky = _make_inputs(device)
        Q = homogeneous_Q(eps, kx, ky)
        N = Nh
        for block in (
            Q[..., :N, :N], Q[..., :N, N:],
            Q[..., N:, :N], Q[..., N:, N:],
        ):
            d   = torch.diagonal(block, dim1=-2, dim2=-1)
            off = block - torch.diag_embed(d)
            assert torch.allclose(off, torch.zeros_like(off), atol=1e-12)

    def test_output_device(self, device):
        eps, kx, ky = _make_inputs(device)
        Q = homogeneous_Q(eps, kx, ky)
        assert Q.device.type == device


# ---------------------------------------------------------------------------
# homogeneous_modes
# ---------------------------------------------------------------------------

class TestHomogeneousModes:

    def test_output_shapes(self, device):
        eps, kx, ky = _make_inputs(device)
        lam, W, V = homogeneous_modes(eps, kx, ky)
        assert lam.shape == (N_WL, 2 * Nh)
        assert   W.shape == (N_WL, 2 * Nh, 2 * Nh)
        assert   V.shape == (N_WL, 2 * Nh, 2 * Nh)

    def test_W_is_identity(self, device):
        """Fourier harmonics are eigenmodes of a homogeneous layer: W = I."""
        eps, kx, ky = _make_inputs(device)
        _, W, _ = homogeneous_modes(eps, kx, ky)
        eye = torch.eye(2 * Nh, dtype=W.dtype, device=device)
        assert torch.allclose(W, eye.expand_as(W))

    def test_lam_equals_1j_times_kz(self, device):
        """lam = 1j * kz by definition; must round-trip through homogeneous_kz."""
        eps, kx, ky = _make_inputs(device)
        lam, _, _ = homogeneous_modes(eps, kx, ky)
        kz = homogeneous_kz(eps, kx, ky)
        assert torch.allclose(lam, 1j * kz, atol=1e-12)

    def test_V_is_Q_scaled_by_inverse_lam(self, device):
        """V[:,j] = Q0[:,j] / lam_j  (column-wise scaling)."""
        eps, kx, ky = _make_inputs(device)
        lam, _, V = homogeneous_modes(eps, kx, ky)
        Q0    = homogeneous_Q(eps, kx, ky)
        V_ref = Q0 * (1.0 / lam).unsqueeze(-2)
        assert torch.allclose(V, V_ref, atol=1e-10)

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
        lam, W, V = homogeneous_modes(eps, kx, ky)
        for tensor, name in ((lam, "lam"), (W, "W"), (V, "V")):
            assert tensor.device.type == device, f"{name} ended up on wrong device"


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
        homogeneous_Q(eps, kx, ky).abs().sum().backward()
        self._assert_finite_grads(eps, kx, ky)

    def test_modes_lam_gradients(self, device):
        eps, kx, ky = _make_inputs(device, requires_grad=True)
        lam, _, _ = homogeneous_modes(eps, kx, ky)
        (lam.abs() ** 2).sum().backward()
        self._assert_finite_grads(eps, kx, ky)

    def test_modes_V_gradients(self, device):
        """Gradient flows through both Q0 and the 1/lam column scaling."""
        eps, kx, ky = _make_inputs(device, requires_grad=True)
        _, _, V = homogeneous_modes(eps, kx, ky)
        V.abs().sum().backward()
        self._assert_finite_grads(eps, kx, ky)
