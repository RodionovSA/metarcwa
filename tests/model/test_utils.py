import warnings

import pytest
import torch
import torch.nn as nn

from metarcwa.model.utils import CallableModule, from_metashapes, from_dispertorch, to_complex, to_real

NX, NY = 32, 32


# ---------------------------------------------------------------------------
# CallableModule
# ---------------------------------------------------------------------------

class TestCallableModule:
    def test_is_nn_module(self):
        cm = CallableModule(lambda x: x)
        assert isinstance(cm, nn.Module)

    def test_is_callable(self):
        cm = CallableModule(lambda x: x * 2)
        assert callable(cm)

    def test_forward_delegates_to_fn(self):
        t = torch.tensor([1.0, 2.0, 3.0])
        cm = CallableModule(lambda x: x * 3)
        assert torch.allclose(cm(t), t * 3)

    def test_module_dep_visible_to_traversal(self):
        dep = nn.Linear(2, 2)
        cm = CallableModule(lambda x: x, dep)
        assert any(dep is m for name, m in cm.named_modules())

    def test_requires_callable(self):
        with pytest.raises(TypeError):
            CallableModule(42)


# ---------------------------------------------------------------------------
# from_metashapes
# ---------------------------------------------------------------------------

class TestFromMetashapes:
    def test_returns_callable_module(self, rect_shape_fn):
        assert isinstance(rect_shape_fn, CallableModule)

    def test_mask_shape(self, rect_shape_fn, lattice):
        mask = rect_shape_fn(lattice, NX, NY)
        assert mask.shape == (NY, NX)

    def test_mask_is_real(self, rect_shape_fn, lattice):
        mask = rect_shape_fn(lattice, NX, NY)
        assert not mask.is_complex()

    def test_mask_range(self, rect_shape_fn, lattice):
        mask = rect_shape_fn(lattice, NX, NY)
        assert float(mask.min()) >= 0.0
        assert float(mask.max()) <= 1.0

    def test_shape_registered_as_submodule(self, rect_shape_fn):
        submodules = list(rect_shape_fn.named_modules())
        assert len(submodules) > 1


# ---------------------------------------------------------------------------
# from_dispertorch
# ---------------------------------------------------------------------------

class TestFromDispertorch:
    def test_returns_callable_module(self, au_eps_fn):
        assert isinstance(au_eps_fn, CallableModule)

    def test_eps_shape_matches_wavelength(self, au_eps_fn, wl):
        eps = au_eps_fn(wl)
        assert eps.shape == wl.shape

    def test_eps_is_complex(self, au_eps_fn, wl):
        eps = au_eps_fn(wl)
        assert eps.is_complex()


# ---------------------------------------------------------------------------
# to_complex
# ---------------------------------------------------------------------------

class TestToComplex:
    def test_real_float32_promoted_to_complex64(self):
        t = torch.tensor([1.0, 2.0], dtype=torch.float32)
        c = to_complex(t)
        assert c.is_complex()
        assert c.dtype == torch.complex64

    def test_real_float64_promoted_to_complex128(self):
        t = torch.tensor([1.0, 2.0], dtype=torch.float64)
        c = to_complex(t)
        assert c.is_complex()
        assert c.dtype == torch.complex128

    def test_complex_passthrough_no_copy(self):
        t = torch.tensor([1.0 + 2j, 3.0 + 4j])
        c = to_complex(t)
        assert c is t


# ---------------------------------------------------------------------------
# to_real
# ---------------------------------------------------------------------------

class TestToReal:
    def test_real_passthrough_no_copy(self):
        t = torch.tensor([1.0, 2.0])
        r = to_real(t)
        assert r is t

    def test_pure_imag_zero_no_warning(self):
        t = torch.tensor([1.0 + 0j, 2.0 + 0j])
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            r = to_real(t)
        assert not r.is_complex()

    def test_lossy_emits_warning(self):
        t = torch.tensor([1.0 + 0.5j])
        with pytest.warns(UserWarning, match="non-negligible"):
            to_real(t, name="test_tensor")

    def test_result_is_real(self):
        t = torch.tensor([1.0 + 0.5j, 2.0 + 0.1j])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            r = to_real(t)
        assert not r.is_complex()
