# tests/test_tvf.py
# Tests for metarcwa.solver.tvf — utilities, TVF class, and gradient flow.
# Mirrors the conventions of tests/test_homogeneous.py:
#   - cpu/cuda device fixture with automatic CUDA skip
#   - seeded _make_inputs helpers with complex128 / float64 tensors
#   - per-function test classes
#   - shared finite-grad helper for autograd tests

import math

import pytest
import torch

from metarcwa.solver.tvf import TVF
from metarcwa.solver.tvf.tvf_utils import (
    _grad_periodic,
    low_pass_mask,
    low_pass_filter,
    _field_magnitude,
    normalize_max_global,
    normalize_elementwise,
    normalize_jones,
    alignment_loss,
    fourier_regularization_loss,
    smoothness_loss,
    total_loss,
)
from metarcwa.solver.harmonics import reciprocal_lattice_vectors

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

B   = 2    # batch size
D0  = 16   # axis -2, a2 direction
D1  = 12   # axis -1, a1 direction  (deliberately non-square)
M   = 3    # Fourier truncation order in a1 direction
N   = 2    # Fourier truncation order in a2 direction  (M != N to catch swaps)

# Rectangular lattice periods
LX = 1.0   # a1 period
LY = 0.8   # a2 period


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


def _rect_lattice(device: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Rectangular lattice: a1=[Lx,0], a2=[0,Ly]."""
    a1 = torch.tensor([LX, 0.0], dtype=torch.float64, device=device)
    a2 = torch.tensor([0.0, LY], dtype=torch.float64, device=device)
    return a1, a2


def _hex_lattice(device: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Hexagonal (pointy-top) lattice: a1=[1,0], a2=[1/2, sqrt(3)/2]."""
    a = 1.0
    a1 = torch.tensor([a, 0.0], dtype=torch.float64, device=device)
    a2 = torch.tensor([a / 2.0, a * math.sqrt(3.0) / 2.0], dtype=torch.float64, device=device)
    return a1, a2


@pytest.fixture(params=["rectangular", "hexagonal"])
def lattice(request, device):
    """Parametrize tests over rectangular and hexagonal lattices."""
    if request.param == "rectangular":
        return _rect_lattice(device)
    return _hex_lattice(device)


def _make_field(device: str, *,
                B: int = B, D0: int = D0, D1: int = D1,
                requires_grad: bool = False) -> torch.Tensor:
    """
    Structured complex128 permittivity field on ``device``.

    Shape ``[B, D0, D1]``.  The real part has a rectangular inclusion so
    that gradients are non-trivial almost everywhere.
    Convention: axis -2 (D0) along a2, axis -1 (D1) along a1.
    """
    field = 2.25 * torch.ones(B, D0, D1, dtype=torch.complex128, device=device)
    field[:, D0 // 4 : 3 * D0 // 4, D1 // 4 : 3 * D1 // 4] = 12.25 + 0j
    g = torch.Generator(); g.manual_seed(1)
    noise = torch.rand(B, D0, D1, dtype=torch.float64, generator=g).to(device)
    field = field + 0.1j * noise
    if requires_grad:
        field.requires_grad_(True)
    return field


def _make_vector_field(device: str, *,
                       B: int = B, D0: int = D0, D1: int = D1,
                       is_complex: bool = False) -> torch.Tensor:
    """Seeded vector field of shape [B, D0, D1, 2]."""
    g = torch.Generator(); g.manual_seed(7)
    dtype = torch.complex128 if is_complex else torch.float64
    return torch.randn(B, D0, D1, 2, dtype=dtype, generator=g).to(device)


# ---------------------------------------------------------------------------
# _grad_periodic
# ---------------------------------------------------------------------------

class TestGradPeriodic:

    def test_output_shapes(self, device, lattice):
        a1, a2 = lattice
        s = torch.real(_make_field(device))
        gx, gy = _grad_periodic(s, a1, a2)
        assert gx.shape == (B, D0, D1)
        assert gy.shape == (B, D0, D1)

    def test_output_device(self, device, lattice):
        a1, a2 = lattice
        s = torch.real(_make_field(device))
        gx, gy = _grad_periodic(s, a1, a2)
        assert gx.device.type == device
        assert gy.device.type == device

    def test_constant_field_has_zero_gradient(self, device, lattice):
        a1, a2 = lattice
        s = torch.ones(B, D0, D1, dtype=torch.float64, device=device)
        gx, gy = _grad_periodic(s, a1, a2)
        assert torch.allclose(gx, torch.zeros_like(gx), atol=1e-12)
        assert torch.allclose(gy, torch.zeros_like(gy), atol=1e-12)

    # ------------------------------------------------------------------
    # Exact axis-orthogonality tests (hold for any lattice, any grid size)
    # ------------------------------------------------------------------

    def test_field_constant_along_a1_axis_gives_gradient_orthogonal_to_a1(
        self, device, lattice
    ):
        """
        If s[b, i, j] = f(i) (constant along axis -1 = a1 direction),
        then ds/df1 = 0, so ∇s = ds/df2 * b2/(2π) which is parallel to b2.
        Since b2 · a1 = 0, the gradient is exactly perpendicular to a1.
        This holds for any lattice and any grid size.
        """
        a1, a2 = lattice
        # Field varies only along axis -2 (a2 direction), constant along axis -1
        g = torch.Generator(); g.manual_seed(2)
        profile = torch.randn(B, D0, dtype=torch.float64, generator=g).to(device)
        s = profile.unsqueeze(-1).expand(B, D0, D1)   # [B, D0, D1], axis -1 constant

        gx, gy = _grad_periodic(s, a1, a2)

        # Dot product ∇s · a1 must be exactly zero
        dot_a1 = gx * a1[0] + gy * a1[1]
        assert torch.allclose(dot_a1, torch.zeros_like(dot_a1), atol=1e-10)

    def test_field_constant_along_a2_axis_gives_gradient_orthogonal_to_a2(
        self, device, lattice
    ):
        """
        If s[b, i, j] = g(j) (constant along axis -2 = a2 direction),
        then ds/df2 = 0, so ∇s = ds/df1 * b1/(2π) which is parallel to b1.
        Since b1 · a2 = 0, the gradient is exactly perpendicular to a2.
        """
        a1, a2 = lattice
        # Field varies only along axis -1 (a1 direction), constant along axis -2
        g = torch.Generator(); g.manual_seed(3)
        profile = torch.randn(B, D1, dtype=torch.float64, generator=g).to(device)
        s = profile.unsqueeze(-2).expand(B, D0, D1)   # [B, D0, D1], axis -2 constant

        gx, gy = _grad_periodic(s, a1, a2)

        # Dot product ∇s · a2 must be exactly zero
        dot_a2 = gx * a2[0] + gy * a2[1]
        assert torch.allclose(dot_a2, torch.zeros_like(dot_a2), atol=1e-10)

    def test_rectangular_regression_against_central_differences(self, device):
        """
        For a rectangular lattice the generalised formula must give the same
        result as simple central differences divided by grid spacing.
        (Regression guard — ensures the chain-rule form didn't break the
        rectangular case.)
        """
        a1, a2 = _rect_lattice(device)
        s = torch.real(_make_field(device))

        gx_new, gy_new = _grad_periodic(s, a1, a2)

        # Reference: plain central differences / physical spacing
        dx = a1[0] / D1   # a1 along axis -1, size D1
        dy = a2[1] / D0   # a2 along axis -2, size D0
        gx_ref = 0.5 * (torch.roll(s, -1, -1) - torch.roll(s, 1, -1)) / dx
        gy_ref = 0.5 * (torch.roll(s, -1, -2) - torch.roll(s, 1, -2)) / dy

        assert torch.allclose(gx_new, gx_ref, atol=1e-10), "gradx mismatch for rectangular lattice"
        assert torch.allclose(gy_new, gy_ref, atol=1e-10), "grady mismatch for rectangular lattice"

    def test_linearly_varying_field_along_a1_has_correct_gradient(self, device):
        """
        s[b, i, j] = j/D1  (fractional coord f1 along a1 on axis -1).
        Interior points: ds/df1 = 1, ds/df2 = 0  =>  ∇s = b1/(2π).
        """
        a1, a2 = _rect_lattice(device)
        b1, _ = reciprocal_lattice_vectors(a1, a2)

        # Field varies along axis -1 (a1 direction)
        js = torch.arange(D1, dtype=torch.float64, device=device) / D1
        s = js.unsqueeze(0).unsqueeze(0).expand(B, D0, D1)

        gx, gy = _grad_periodic(s, a1, a2)

        expected_gx = (b1[0] / (2 * torch.pi)).item()
        expected_gy = (b1[1] / (2 * torch.pi)).item()

        # Check interior (boundary wraps are wrong for non-periodic linear field)
        assert torch.allclose(gx[:, :, 1:-1],
                               torch.full_like(gx[:, :, 1:-1], expected_gx), atol=1e-10)
        assert torch.allclose(gy[:, :, 1:-1],
                               torch.full_like(gy[:, :, 1:-1], expected_gy), atol=1e-10)


# ---------------------------------------------------------------------------
# low_pass_mask
# ---------------------------------------------------------------------------

class TestLowPassMask:

    def test_shape(self, device):
        mask = low_pass_mask(D0, D1, M, N, device=device)
        assert mask.shape == (D0, D1)

    def test_dtype_is_bool(self, device):
        assert low_pass_mask(D0, D1, M, N, device=device).dtype == torch.bool

    def test_device(self, device):
        assert low_pass_mask(D0, D1, M, N, device=device).device.type == device

    def test_dc_always_passes(self, device):
        mask = low_pass_mask(D0, D1, M, N, device=device)
        assert mask[D0 // 2, D1 // 2].item()

    def test_bandwidth_count(self, device):
        """Total number of passed harmonics = (2M+1) * (2N+1)."""
        mask = low_pass_mask(D0, D1, M, N, device=device)
        assert mask.sum().item() == (2 * M + 1) * (2 * N + 1)

    def test_m_applies_to_a1_axis(self, device):
        """
        M is the a1-direction (axis -1) half-bandwidth.
        The M-th harmonic in axis -1 should pass; M+1-th should not.
        (Tests that M and N are not swapped — important when M != N.)
        """
        mask = low_pass_mask(D0, D1, M, N, device=device)
        # a1 axis is -1 (columns); DC row = D0//2
        row = D0 // 2   # zero n (a2 order = 0)
        assert mask[row, D1 // 2 + M].item(),     "M-th harmonic in a1 dir should pass"
        assert not mask[row, D1 // 2 + M + 1].item(), "M+1-th harmonic in a1 dir should be blocked"

    def test_n_applies_to_a2_axis(self, device):
        """
        N is the a2-direction (axis -2) half-bandwidth.
        The N-th harmonic in axis -2 should pass; N+1-th should not.
        """
        mask = low_pass_mask(D0, D1, M, N, device=device)
        # a2 axis is -2 (rows); DC column = D1//2
        col = D1 // 2   # zero m (a1 order = 0)
        assert mask[D0 // 2 + N, col].item(),     "N-th harmonic in a2 dir should pass"
        assert not mask[D0 // 2 + N + 1, col].item(), "N+1-th harmonic in a2 dir should be blocked"


# ---------------------------------------------------------------------------
# low_pass_filter
# ---------------------------------------------------------------------------

class TestLowPassFilter:

    def _make_grad(self, device):
        a1, a2 = _rect_lattice(device)
        s = torch.real(_make_field(device))
        return _grad_periodic(s, a1, a2)

    def test_output_shapes(self, device):
        gx, gy = self._make_grad(device)
        fx, fy = low_pass_filter((gx, gy), M, N)
        assert fx.shape == (B, D0, D1)
        assert fy.shape == (B, D0, D1)

    def test_output_is_real(self, device):
        gx, gy = self._make_grad(device)
        fx, fy = low_pass_filter((gx, gy), M, N)
        assert not torch.is_complex(fx)
        assert not torch.is_complex(fy)

    def test_output_device(self, device):
        gx, gy = self._make_grad(device)
        fx, fy = low_pass_filter((gx, gy), M, N)
        assert fx.device.type == device
        assert fy.device.type == device

    def test_zero_gradients_pass_through(self, device):
        """Zero input → zero output."""
        gx = torch.zeros(B, D0, D1, dtype=torch.float64, device=device)
        gy = torch.zeros_like(gx)
        fx, fy = low_pass_filter((gx, gy), M, N)
        assert torch.allclose(fx, gx, atol=1e-12)
        assert torch.allclose(fy, gy, atol=1e-12)

    def test_invalid_input_raises(self):
        with pytest.raises(ValueError, match="tuple"):
            low_pass_filter(None, M, N)
        with pytest.raises(ValueError, match="shape"):
            t1 = torch.zeros(B, D0, D1)
            t2 = torch.zeros(B, D0 + 1, D1)
            low_pass_filter((t1, t2), M, N)


# ---------------------------------------------------------------------------
# _field_magnitude
# ---------------------------------------------------------------------------

class TestFieldMagnitude:

    def test_shape(self, device):
        v = _make_vector_field(device)
        assert _field_magnitude(v).shape == (B, D0, D1, 1)

    def test_device(self, device):
        v = _make_vector_field(device)
        assert _field_magnitude(v).device.type == device

    def test_zero_field_gives_zero_magnitude(self, device):
        v = torch.zeros(B, D0, D1, 2, dtype=torch.float64, device=device)
        assert torch.allclose(_field_magnitude(v), torch.zeros_like(_field_magnitude(v)))

    def test_unit_vector_gives_unit_magnitude(self, device):
        v = torch.zeros(B, D0, D1, 2, dtype=torch.float64, device=device)
        v[..., 0] = 1.0
        assert torch.allclose(_field_magnitude(v), torch.ones_like(_field_magnitude(v)), atol=1e-12)


# ---------------------------------------------------------------------------
# normalize_max_global
# ---------------------------------------------------------------------------

class TestNormalizeMaxGlobal:

    def test_shape_and_device(self, device):
        v = _make_vector_field(device)
        n = normalize_max_global(v)
        assert n.shape == v.shape
        assert n.device.type == device

    def test_max_magnitude_is_one(self, device):
        v = _make_vector_field(device)
        n = normalize_max_global(v)
        max_mag = torch.amax(_field_magnitude(n), dim=(-3, -2))   # [B, 1]
        assert torch.allclose(max_mag, torch.ones_like(max_mag), atol=1e-10)

    def test_zero_field_stays_zero(self, device):
        v = torch.zeros(B, D0, D1, 2, dtype=torch.float64, device=device)
        assert torch.allclose(normalize_max_global(v), v)


# ---------------------------------------------------------------------------
# normalize_elementwise
# ---------------------------------------------------------------------------

class TestNormalizeElementwise:

    def test_shape_and_device(self, device):
        v = _make_vector_field(device)
        n = normalize_elementwise(v)
        assert n.shape == v.shape
        assert n.device.type == device

    def test_each_pixel_has_unit_magnitude(self, device):
        v = _make_vector_field(device)
        n = normalize_elementwise(v)
        mag = _field_magnitude(n)
        non_zero = _field_magnitude(v).squeeze(-1) > 1e-12
        assert torch.allclose(
            mag.squeeze(-1)[non_zero],
            torch.ones(non_zero.sum(), dtype=mag.dtype, device=v.device),
            atol=1e-10,
        )


# ---------------------------------------------------------------------------
# normalize_jones
# ---------------------------------------------------------------------------

class TestNormalizeJones:

    def test_shape_and_device(self, device):
        v = _make_vector_field(device)
        j = normalize_jones(v)
        assert j.shape == (B, D0, D1, 2)
        assert j.device.type == device

    def test_output_is_complex(self, device):
        v = _make_vector_field(device)
        assert torch.is_complex(normalize_jones(v))

    def test_invalid_last_dim_raises(self, device):
        with pytest.raises(ValueError, match="2"):
            normalize_jones(torch.zeros(B, D0, D1, 3, device=device))

    def test_invalid_ndim_raises(self, device):
        with pytest.raises(ValueError, match="4-D"):
            normalize_jones(torch.zeros(D0, D1, 2, device=device))


# ---------------------------------------------------------------------------
# alignment_loss
# ---------------------------------------------------------------------------

class TestAlignmentLoss:

    def test_identical_fields_give_zero_loss(self, device):
        v = _make_vector_field(device)
        w = torch.ones(B, D0, D1, 1, dtype=v.dtype, device=device)
        loss = alignment_loss(v, v, w)
        assert loss.shape == (B,)
        assert torch.allclose(loss, torch.zeros_like(loss), atol=1e-12)

    def test_shape_and_device(self, device):
        v = _make_vector_field(device)
        t = _make_vector_field(device)
        w = torch.ones(B, D0, D1, 1, dtype=v.dtype, device=device)
        loss = alignment_loss(v, t, w)
        assert loss.shape == (B,)
        assert loss.device.type == device

    def test_shape_mismatch_raises(self, device):
        v = torch.zeros(B, D0, D1, 2, device=device)
        t = torch.zeros(B, D0, D1 + 1, 2, device=device)
        w = torch.ones(B, D0, D1, 1, device=device)
        with pytest.raises(ValueError, match="shape"):
            alignment_loss(v, t, w)


# ---------------------------------------------------------------------------
# fourier_regularization_loss
# ---------------------------------------------------------------------------

class TestFourierRegularizationLoss:

    def test_shape_and_device(self, device, lattice):
        a1, a2 = lattice
        fft = _make_vector_field(device, is_complex=True)
        loss = fourier_regularization_loss(fft, a1, a2)
        assert loss.shape == (B,)
        assert loss.device.type == device

    def test_dc_only_field_gives_zero_loss(self, device, lattice):
        """DC-only signal has G=0 so K_norm2=0 and loss=0."""
        a1, a2 = lattice
        fft = torch.zeros(B, D0, D1, 2, dtype=torch.complex128, device=device)
        fft[:, D0 // 2, D1 // 2, :] = 1.0
        loss = fourier_regularization_loss(fft, a1, a2)
        assert torch.allclose(loss, torch.zeros_like(loss), atol=1e-10)

    def test_non_negative(self, device, lattice):
        a1, a2 = lattice
        fft = _make_vector_field(device, is_complex=True)
        assert (fourier_regularization_loss(fft, a1, a2) >= 0).all()


# ---------------------------------------------------------------------------
# smoothness_loss
# ---------------------------------------------------------------------------

class TestSmoothnessLoss:

    def test_shape_and_device(self, device, lattice):
        a1, a2 = lattice
        v = _make_vector_field(device)
        loss = smoothness_loss(v, a1, a2)
        assert loss.shape == (B,)
        assert loss.device.type == device

    def test_constant_field_gives_zero_loss(self, device, lattice):
        a1, a2 = lattice
        v = torch.ones(B, D0, D1, 2, dtype=torch.float64, device=device)
        assert torch.allclose(smoothness_loss(v, a1, a2),
                               torch.zeros(B, dtype=torch.float64, device=device), atol=1e-10)

    def test_non_negative(self, device, lattice):
        a1, a2 = lattice
        v = _make_vector_field(device)
        assert (smoothness_loss(v, a1, a2) >= 0).all()


# ---------------------------------------------------------------------------
# total_loss
# ---------------------------------------------------------------------------

class TestTotalLoss:

    def _make_params(self, device):
        """Valid [B, D0, D1, 2, 2] float64 params tensor."""
        g = torch.Generator(); g.manual_seed(3)
        return torch.randn(B, D0, D1, 2, 2, dtype=torch.float64, generator=g).to(device)

    def test_shape_and_device(self, device, lattice):
        a1, a2 = lattice
        params = self._make_params(device)
        target = _make_vector_field(device)
        weights = torch.ones(B, D0, D1, 1, dtype=torch.float64, device=device)
        loss = total_loss(params, target, weights, a1, a2)
        assert loss.shape == (B,)
        assert loss.device.type == device

    def test_non_negative(self, device, lattice):
        a1, a2 = lattice
        params = self._make_params(device)
        target = _make_vector_field(device)
        weights = torch.ones(B, D0, D1, 1, dtype=torch.float64, device=device)
        assert (total_loss(params, target, weights, a1, a2) >= 0).all()


# ---------------------------------------------------------------------------
# TVF.compute — shapes, device, finite values
# ---------------------------------------------------------------------------

class TestTVFCompute:

    @pytest.mark.parametrize("method", TVF.METHODS)
    def test_output_shapes(self, device, method):
        a1, a2 = _rect_lattice(device)
        field = _make_field(device)
        Tx, Ty = TVF(a1, a2, M=M, N=N, method=method).compute(field, steps=1)
        assert Tx.shape == (B, D0, D1), f"Bad Tx shape for method={method}"
        assert Ty.shape == (B, D0, D1), f"Bad Ty shape for method={method}"

    @pytest.mark.parametrize("method", TVF.METHODS)
    def test_output_device(self, device, method):
        a1, a2 = _rect_lattice(device)
        field = _make_field(device)
        Tx, Ty = TVF(a1, a2, M=M, N=N, method=method).compute(field, steps=1)
        assert Tx.device.type == device, f"Tx device wrong for method={method}"
        assert Ty.device.type == device, f"Ty device wrong for method={method}"

    @pytest.mark.parametrize("method", TVF.METHODS)
    def test_output_is_finite(self, device, method):
        a1, a2 = _rect_lattice(device)
        field = _make_field(device)
        Tx, Ty = TVF(a1, a2, M=M, N=N, method=method).compute(field, steps=1)
        assert torch.isfinite(Tx.abs()).all(), f"Tx non-finite for method={method}"
        assert torch.isfinite(Ty.abs()).all(), f"Ty non-finite for method={method}"

    # --- Hexagonal lattice ---

    @pytest.mark.parametrize("method", TVF.METHODS)
    def test_hexagonal_shapes_and_finite(self, device, method):
        """TVF.compute on a hexagonal lattice: shapes, device, finite values."""
        a1, a2 = _hex_lattice(device)
        field = _make_field(device)
        Tx, Ty = TVF(a1, a2, M=M, N=N, method=method).compute(field, steps=1)
        assert Tx.shape == (B, D0, D1), f"Hexagonal Tx shape for method={method}"
        assert Ty.shape == (B, D0, D1), f"Hexagonal Ty shape for method={method}"
        assert Tx.device.type == device
        assert torch.isfinite(Tx.abs()).all(), f"Hexagonal Tx non-finite for method={method}"
        assert torch.isfinite(Ty.abs()).all(), f"Hexagonal Ty non-finite for method={method}"

    def test_invalid_method_raises(self, device):
        a1, a2 = _rect_lattice(device)
        with pytest.raises(ValueError, match="Unknown TVF method"):
            TVF(a1, a2, M=M, N=N, method="BadMethod")

    def test_non_3d_field_raises(self, device):
        a1, a2 = _rect_lattice(device)
        tvf = TVF(a1, a2, M=M, N=N, method="Pol")
        with pytest.raises(ValueError, match="3-D"):
            tvf._optimize(torch.zeros(D0, D1, device=device))


# ---------------------------------------------------------------------------
# Gradient flow — through differentiable utilities (not TVF.compute which
# intentionally detaches from the computation graph)
# ---------------------------------------------------------------------------

class TestGradientFlow:
    """
    Autograd tests for the utility functions that compose the TVF loss.
    TVF.compute is detached by design; tested for shape/device/finite only.
    """

    @staticmethod
    def _assert_finite_grads(*tensors: torch.Tensor) -> None:
        for t in tensors:
            assert t.grad is not None, "gradient is None"
            assert t.grad.shape == t.shape, "gradient shape mismatch"
            assert torch.isfinite(t.grad.abs()).all(), "gradient is non-finite"

    def test_grad_periodic_gradient(self, device):
        a1, a2 = _rect_lattice(device)
        s = torch.real(_make_field(device)).requires_grad_(True)
        gx, gy = _grad_periodic(s, a1, a2)
        (gx.abs() + gy.abs()).sum().backward()
        self._assert_finite_grads(s)

    def test_normalize_max_global_gradient(self, device):
        v = _make_vector_field(device).requires_grad_(True)
        normalize_max_global(v).abs().sum().backward()
        self._assert_finite_grads(v)

    def test_normalize_elementwise_gradient(self, device):
        v = _make_vector_field(device).requires_grad_(True)
        normalize_elementwise(v).abs().sum().backward()
        self._assert_finite_grads(v)

    def test_normalize_jones_gradient(self, device):
        v = _make_vector_field(device).requires_grad_(True)
        normalize_jones(v).abs().sum().backward()
        self._assert_finite_grads(v)

    def test_alignment_loss_gradient(self, device):
        v = _make_vector_field(device).requires_grad_(True)
        t = _make_vector_field(device)
        w = torch.ones(B, D0, D1, 1, dtype=torch.float64, device=device)
        alignment_loss(v, t, w).sum().backward()
        self._assert_finite_grads(v)

    def test_smoothness_loss_gradient(self, device):
        a1, a2 = _rect_lattice(device)
        v = _make_vector_field(device).requires_grad_(True)
        smoothness_loss(v, a1, a2).sum().backward()
        self._assert_finite_grads(v)

    def test_total_loss_gradient(self, device):
        """Gradient flows through the full total_loss w.r.t. Fourier params."""
        a1, a2 = _rect_lattice(device)
        g = torch.Generator(); g.manual_seed(5)
        params = torch.randn(B, D0, D1, 2, 2, dtype=torch.float64,
                             generator=g).to(device).requires_grad_(True)
        target = _make_vector_field(device)
        weights = torch.ones(B, D0, D1, 1, dtype=torch.float64, device=device)
        total_loss(params, target, weights, a1, a2).sum().backward()
        self._assert_finite_grads(params)

    def test_total_loss_gradient_hexagonal(self, device):
        """total_loss gradient also works for a hexagonal lattice."""
        a1, a2 = _hex_lattice(device)
        g = torch.Generator(); g.manual_seed(6)
        params = torch.randn(B, D0, D1, 2, 2, dtype=torch.float64,
                             generator=g).to(device).requires_grad_(True)
        target = _make_vector_field(device)
        weights = torch.ones(B, D0, D1, 1, dtype=torch.float64, device=device)
        total_loss(params, target, weights, a1, a2).sum().backward()
        self._assert_finite_grads(params)
