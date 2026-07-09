# tests/model/test_model.py
# Tests for Model.spec()'s ModelSpec field merge (ANALYSIS.md A6).
#
# Model.spec() used to copy each ModelSpec field by hand from stack_spec /
# source_spec; it now merges them automatically by field name via
# dataclasses.fields(ModelSpec). Two things need locking in: (1) the merge
# logic itself, tested against small decoupled dummy dataclasses so it
# doesn't depend on ModelSpec's actual field set, and (2) that Model.spec()
# still produces the same ModelSpec end-to-end for a real model.

import dataclasses

import pytest
import torch

from metarcwa.model.base import Model, ModelSpec
from metarcwa.model.stack import Stack
from metarcwa.model.layer import Layer
from metarcwa.model.medium import IsotropicMedium
from metarcwa.model.lattice import Lattice
from metarcwa.model.source import Source
from metarcwa.model.nn_helpers import CallableModule


# ---------------------------------------------------------------------------
# Decoupled unit test of the merge logic (dummy dataclasses, not ModelSpec)
# ---------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class _DummyTarget:
    x: int
    y: int


@dataclasses.dataclass(frozen=True)
class _DummySubA:
    x: int


@dataclasses.dataclass(frozen=True)
class _DummySubB:
    y: int


def _merge(target_cls, *subs):
    """Same merge logic as Model.spec(): pull each target field from
    whichever sub has it, by name."""
    kwargs = {}
    for f in dataclasses.fields(target_cls):
        for sub in subs:
            if hasattr(sub, f.name):
                kwargs[f.name] = getattr(sub, f.name)
                break
        else:
            raise AttributeError(
                f"{target_cls.__name__}.{f.name!r} not found on any sub-spec"
            )
    return target_cls(**kwargs)


class TestSpecMergeLogic:
    def test_merges_fields_from_multiple_subs(self):
        merged = _merge(_DummyTarget, _DummySubA(x=1), _DummySubB(y=2))
        assert merged == _DummyTarget(x=1, y=2)

    def test_first_matching_sub_wins(self):
        """If two subs both declare the same field name, the first one in
        the iteration order is used (mirrors Model.spec()'s
        (stack_spec, source_spec) order)."""
        @dataclasses.dataclass(frozen=True)
        class _SubWithXY:
            x: int
            y: int
        first  = _SubWithXY(x=1, y=2)
        second = _SubWithXY(x=99, y=99)   # would give wrong values if picked
        merged = _merge(_DummyTarget, first, second)
        assert merged == _DummyTarget(x=1, y=2)

    def test_missing_field_raises(self):
        with pytest.raises(AttributeError, match="y"):
            _merge(_DummyTarget, _DummySubA(x=1))


# ---------------------------------------------------------------------------
# End-to-end: Model.spec() still produces the correct ModelSpec
# ---------------------------------------------------------------------------

def _const_eps(val: complex):
    return CallableModule(lambda wvl: torch.full_like(wvl, val, dtype=torch.complex128))


def _make_model() -> Model:
    incidence    = IsotropicMedium(_const_eps(1.0 + 0j))
    transmission = IsotropicMedium(_const_eps(1.0 + 0j))
    layer        = Layer(IsotropicMedium(_const_eps(2.5 + 0j)), thickness=0.3)
    lattice      = Lattice.rectangular(1.0, 1.0)
    stack        = Stack(incidence, [layer], transmission, lattice)
    source       = Source(wavelength=1.0)
    return Model(stack, source).to(dtype=torch.float64)


class TestModelSpecEndToEnd:
    def test_returns_model_spec(self):
        model = _make_model()
        spec = model.spec(nx=8, ny=8)
        assert isinstance(spec, ModelSpec)

    def test_fields_sourced_correctly(self):
        model = _make_model()
        stack_spec = model.stack.spec(model.source.wavelength, 8, 8)
        source_spec = model.source.spec(stack_spec.incidence.refractive_index().real)
        spec = model.spec(nx=8, ny=8)

        # structure fields come from stack_spec
        assert spec.layers == stack_spec.layers
        assert spec.incidence == stack_spec.incidence
        assert spec.transmission == stack_spec.transmission
        assert torch.equal(spec.a1, stack_spec.a1)
        assert torch.equal(spec.a2, stack_spec.a2)

        # illumination fields come from source_spec
        assert torch.equal(spec.wavelength, source_spec.wavelength)
        assert torch.equal(spec.kx0, source_spec.kx0)
        assert torch.equal(spec.ky0, source_spec.ky0)

    def test_all_model_spec_fields_populated(self):
        """Every declared ModelSpec field must have been resolved (guards
        against the merge silently leaving a field unset)."""
        model = _make_model()
        spec = model.spec(nx=8, ny=8)
        for f in dataclasses.fields(ModelSpec):
            assert getattr(spec, f.name) is not None
