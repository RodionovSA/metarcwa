# tests/model/test_nn_helpers.py
# Tests for metarcwa.model.nn_helpers: importable standalone from its new
# location (A2 -- split out of the model/utils.py grab-bag). Behavior is
# already covered in depth by tests/model/test_utils.py via the
# metarcwa.model.utils re-export; this file locks in the new import path.

import torch
import torch.nn as nn

from metarcwa.model.nn_helpers import register, CallableModule


class TestRegister:
    def test_parameter_becomes_attribute(self):
        m = nn.Module()
        p = nn.Parameter(torch.tensor(1.0))
        register(m, "p", p)
        assert m.p is p
        assert "p" in dict(m.named_parameters())

    def test_plain_value_becomes_buffer(self):
        m = nn.Module()
        register(m, "b", 2.0)
        assert "b" in dict(m.named_buffers())
        assert torch.equal(m.b, torch.tensor(2.0))


class TestCallableModule:
    def test_is_nn_module(self):
        cm = CallableModule(lambda x: x)
        assert isinstance(cm, nn.Module)

    def test_forward_delegates(self):
        cm = CallableModule(lambda x, y: x + y)
        assert cm(2, 3) == 5

    def test_non_callable_raises(self):
        import pytest
        with pytest.raises(TypeError):
            CallableModule(42)
