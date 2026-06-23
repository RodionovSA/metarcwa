import pytest
import torch
import torch.nn as nn

from metarcwa.model.lattice import Lattice
from metarcwa.model.layer import Layer, HomogeneousLayer, PatternedLayer
from metarcwa.model.stack import Stack
from metarcwa.model.medium import IsotropicMedium
from metarcwa.model.utils import CallableModule


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _const_eps_fn(value):
    val = torch.as_tensor(value, dtype=torch.complex64)
    return CallableModule(lambda wl: val.expand(wl.shape))


def _shape_fn(ny, nx):
    mask = torch.ones(ny, nx)
    def fn(lattice, nx_, ny_):
        assert nx_ == nx and ny_ == ny, (
            f"shape_fn called with wrong grid: got ({nx_}, {ny_}), expected ({nx}, {ny})"
        )
        return mask
    return CallableModule(fn)


def _make_stack(nx, ny, patterned=True):
    lat = Lattice.rectangular(1.0, 1.0)
    wl  = torch.tensor([1.0])

    uniform = Layer(
        medium_solid=IsotropicMedium(_const_eps_fn(2.25 + 0j)),
        thickness=torch.tensor(0.1),
    )

    if patterned:
        patt = Layer(
            medium_solid=IsotropicMedium(_const_eps_fn(12.0 + 0.1j)),
            medium_void=IsotropicMedium(_const_eps_fn(1.0 + 0j)),
            shape_fn=_shape_fn(ny, nx),
            thickness=torch.tensor(0.2),
        )
        layers = [uniform, patt]
    else:
        layers = [uniform]

    stack = Stack(
        incidence=IsotropicMedium(_const_eps_fn(1.0 + 0j)),
        layers=layers,
        transmission=IsotropicMedium(_const_eps_fn(2.25 + 0j)),
        lattice=lat,
    )
    return stack, wl, nx, ny


# ---------------------------------------------------------------------------
# Axis ordering
# ---------------------------------------------------------------------------

class TestAxisOrder:
    def test_uniform_layer_yields_homogeneous_spec(self):
        nx, ny = 7, 5
        stack, wl, nx, ny = _make_stack(nx, ny, patterned=False)
        spec = stack.spec(wl, nx, ny)
        assert isinstance(spec.layers[0], HomogeneousLayer)

    def test_patterned_layer_trailing_dims(self):
        nx, ny = 7, 5
        stack, wl, nx, ny = _make_stack(nx, ny, patterned=True)
        spec = stack.spec(wl, nx, ny)
        patt = spec.layers[1]
        assert isinstance(patt, PatternedLayer)
        assert patt.pattern.shape[-2:] == (ny, nx), (
            f"Expected trailing (ny={ny}, nx={nx}), got {patt.pattern.shape[-2:]}"
        )

    def test_uniform_and_patterned_coexist(self):
        nx, ny = 7, 5
        stack, wl, nx, ny = _make_stack(nx, ny, patterned=True)
        spec = stack.spec(wl, nx, ny)
        assert isinstance(spec.layers[0], HomogeneousLayer)
        assert isinstance(spec.layers[1], PatternedLayer)
        assert spec.layers[1].pattern.shape[-2:] == (ny, nx)


# ---------------------------------------------------------------------------
# Dtype enforcement
# ---------------------------------------------------------------------------

class TestDtype:
    def test_homogeneous_layer_eps_is_complex(self):
        nx, ny = 4, 3
        stack, wl, nx, ny = _make_stack(nx, ny, patterned=False)
        spec = stack.spec(wl, nx, ny)
        assert spec.layers[0].medium.eps.is_complex()

    def test_patterned_layer_solid_eps_is_complex(self):
        nx, ny = 4, 3
        stack, wl, nx, ny = _make_stack(nx, ny, patterned=True)
        spec = stack.spec(wl, nx, ny)
        assert spec.layers[1].medium_solid.eps.is_complex()

    def test_patterned_layer_void_eps_is_complex(self):
        nx, ny = 4, 3
        stack, wl, nx, ny = _make_stack(nx, ny, patterned=True)
        spec = stack.spec(wl, nx, ny)
        assert spec.layers[1].medium_void.eps.is_complex()

    def test_eps_transmission_is_complex(self):
        nx, ny = 4, 3
        stack, wl, nx, ny = _make_stack(nx, ny, patterned=False)
        spec = stack.spec(wl, nx, ny)
        assert spec.transmission.eps.is_complex()

    def test_eps_incidence_is_complex(self):
        nx, ny = 4, 3
        stack, wl, nx, ny = _make_stack(nx, ny, patterned=False)
        spec = stack.spec(wl, nx, ny)
        assert spec.incidence.eps.is_complex()


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

class TestValidation:
    def _minimal_layer(self):
        return Layer(
            medium_solid=IsotropicMedium(_const_eps_fn(1.0)),
            thickness=torch.tensor(0.1),
        )

    def test_stack_requires_medium_for_incidence(self):
        lat = Lattice.rectangular(1.0, 1.0)
        with pytest.raises(TypeError, match="incidence"):
            Stack(
                incidence=_const_eps_fn(1.0),
                layers=[self._minimal_layer()],
                transmission=IsotropicMedium(_const_eps_fn(1.0)),
                lattice=lat,
            )

    def test_stack_requires_medium_for_transmission(self):
        lat = Lattice.rectangular(1.0, 1.0)
        with pytest.raises(TypeError, match="transmission"):
            Stack(
                incidence=IsotropicMedium(_const_eps_fn(1.0)),
                layers=[self._minimal_layer()],
                transmission=_const_eps_fn(1.0),
                lattice=lat,
            )
