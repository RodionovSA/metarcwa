# tests/model/test_adapters.py
# Tests for metarcwa.model.adapters: importable standalone from its new
# location (A2 -- split out of the model/utils.py grab-bag). Full behavioral
# coverage already exists in tests/model/test_utils.py via the
# metarcwa.model.utils re-export (and shares this directory's conftest.py
# fixtures); this file locks in the new import path with a couple of smoke
# checks.

from metarcwa.model.adapters import from_metashapes, from_dispertorch
from metarcwa.model.nn_helpers import CallableModule


class TestFromMetashapes:
    def test_returns_callable_module(self, rect_shape_fn):
        assert isinstance(rect_shape_fn, CallableModule)

    def test_mask_shape(self, rect_shape_fn, lattice):
        mask = rect_shape_fn(lattice, 32, 32)
        assert mask.shape == (32, 32)


class TestFromDispertorch:
    def test_returns_callable_module(self, au_eps_fn):
        assert isinstance(au_eps_fn, CallableModule)

    def test_eps_is_complex(self, au_eps_fn, wl):
        assert au_eps_fn(wl).is_complex()
