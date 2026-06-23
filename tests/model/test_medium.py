import pytest
import torch
import torch.nn as nn

from metarcwa.model.medium import IsotropicMedium, IsotropicMediumSpec, Medium
from metarcwa.model.utils import CallableModule

N_WL = 5


@pytest.fixture
def wl():
    return torch.linspace(400, 800, N_WL)


def _const_eps(value, *, real=False):
    dtype = torch.float32 if real else torch.complex64
    val = torch.as_tensor(value, dtype=dtype)
    return CallableModule(lambda wl: val.expand(wl.shape))


# ---------------------------------------------------------------------------
# IsotropicMedium
# ---------------------------------------------------------------------------

class TestIsotropicMedium:
    def test_is_nn_module(self):
        assert isinstance(IsotropicMedium(_const_eps(2.25)), nn.Module)

    def test_is_medium_subclass(self):
        assert isinstance(IsotropicMedium(_const_eps(2.25)), Medium)

    def test_requires_callable(self):
        with pytest.raises(TypeError, match="callable"):
            IsotropicMedium(42)

    def test_spec_returns_isotropic_medium_spec(self, wl):
        m = IsotropicMedium(_const_eps(2.25))
        spec = m.spec(wl)
        assert isinstance(spec, IsotropicMediumSpec)

    def test_spec_eps_shape_matches_wl(self, wl):
        m = IsotropicMedium(_const_eps(2.25))
        spec = m.spec(wl)
        assert spec.eps.shape == wl.shape

    def test_spec_eps_is_complex_for_complex_input(self, wl):
        m = IsotropicMedium(_const_eps(2.25 + 0.1j))
        assert m.spec(wl).eps.is_complex()

    def test_spec_eps_promoted_to_complex_from_real(self, wl):
        m = IsotropicMedium(_const_eps(2.25, real=True))
        assert m.spec(wl).eps.is_complex()

    def test_spec_raises_on_shape_mismatch(self, wl):
        bad_fn = CallableModule(
            lambda wl: torch.ones(wl.shape[0] + 1, dtype=torch.complex64)
        )
        m = IsotropicMedium(bad_fn)
        with pytest.raises(ValueError):
            m.spec(wl)


# ---------------------------------------------------------------------------
# IsotropicMediumSpec
# ---------------------------------------------------------------------------

class TestIsotropicMediumSpec:
    def test_refractive_index_formula(self, wl):
        eps_val = torch.tensor(2.25 + 0j, dtype=torch.complex64)
        m = IsotropicMedium(_const_eps(2.25))
        spec = m.spec(wl)
        n = spec.refractive_index()
        expected = torch.sqrt(eps_val).expand(wl.shape)
        assert torch.allclose(n, expected)

    def test_refractive_index_lossy_medium(self, wl):
        m = IsotropicMedium(_const_eps(-16.0 + 1.0j))
        spec = m.spec(wl)
        n = spec.refractive_index()
        assert n.is_complex()
        assert n.shape == wl.shape

    def test_refractive_index_air(self, wl):
        m = IsotropicMedium(_const_eps(1.0 + 0j))
        n = m.spec(wl).refractive_index()
        assert torch.allclose(n.real, torch.ones(wl.shape))
        assert torch.allclose(n.imag.abs(), torch.zeros(wl.shape), atol=1e-6)
