# tests/test_dtypes.py
# Tests for metarcwa._dtypes: importable standalone from the top of the
# package (A1 — shared by `model` and `solver` without either depending on
# the other). Behavior itself is already covered in depth by
# tests/model/test_utils.py via the metarcwa.model.utils re-export; this
# file just locks in the new import path and the dtype table.

import pytest
import torch

from metarcwa._dtypes import _REAL_TO_COMPLEX, to_complex, to_real


class TestRealToComplexTable:
    def test_maps_standard_float_dtypes(self):
        assert _REAL_TO_COMPLEX[torch.float32] == torch.complex64
        assert _REAL_TO_COMPLEX[torch.float64] == torch.complex128


class TestToComplex:
    def test_real_float32_promoted_to_complex64(self):
        t = torch.tensor([1.0, 2.0], dtype=torch.float32)
        c = to_complex(t)
        assert c.is_complex()
        assert c.dtype == torch.complex64

    def test_complex_passthrough_no_copy(self):
        t = torch.tensor([1.0 + 2j, 3.0 + 4j])
        assert to_complex(t) is t


class TestToReal:
    def test_real_passthrough_no_copy(self):
        t = torch.tensor([1.0, 2.0])
        assert to_real(t) is t

    def test_lossy_emits_warning(self):
        t = torch.tensor([1.0 + 0.5j])
        with pytest.warns(UserWarning, match="non-negligible"):
            to_real(t, name="test_tensor")


class TestNoModelSolverCoupling:
    """A1: solver-side modules must get _REAL_TO_COMPLEX from metarcwa._dtypes,
    not from metarcwa.model (one-way model -> solver dependency direction)."""

    def test_isotropic_and_homogeneous_import_from_dtypes_not_model(self):
        import inspect
        from metarcwa.solver.layersolver import isotropic, homogeneous

        for mod in (isotropic, homogeneous):
            src = inspect.getsource(mod)
            assert "metarcwa.model" not in src, (
                f"{mod.__name__} should not import from metarcwa.model"
            )
