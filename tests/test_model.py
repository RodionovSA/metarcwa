# tests/test_model.py
# Tests for src/metarcwa/model — axis order and dtype enforcement.

import warnings
import pytest
import torch
import torch.nn as nn

from metarcwa.model.lattice import Lattice
from metarcwa.model.layer import Layer
from metarcwa.model.stack import Stack
from metarcwa.model.utils import CallableModule, to_complex, to_real


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _const_eps_fn(value):
    """Return a CallableModule that returns a constant scalar complex eps."""
    val = torch.as_tensor(value)
    def fn(wl):
        return val.expand(wl.shape)
    return CallableModule(fn)


def _shape_fn(ny, nx):
    """Return a shape_fn that outputs a known [Ny, Nx] mask of ones."""
    mask = torch.ones(ny, nx)
    def fn(lattice, nx_, ny_):
        assert nx_ == nx and ny_ == ny, (
            f"shape_fn called with wrong resolution: got ({nx_}, {ny_}), expected ({nx}, {ny})"
        )
        return mask
    return CallableModule(fn)


def _make_stack(nx, ny, patterned=True):
    """Build a minimal 2-layer Stack with an asymmetric grid."""
    lat = Lattice.rectangular(1.0, 1.0)
    wl  = torch.tensor([1.0])

    # Uniform layer (no shape_fn)
    uniform = Layer(
        eps_solid_fn=_const_eps_fn(2.25 + 0j),  # lossless → real input
        thickness=torch.tensor(0.1),
    )

    if patterned:
        # Patterned layer (with shape_fn)
        patt = Layer(
            eps_solid_fn=_const_eps_fn(12.0 + 0.1j),
            eps_void_fn=_const_eps_fn(1.0 + 0j),
            shape_fn=_shape_fn(ny, nx),
            thickness=torch.tensor(0.2),
        )
        layers = [uniform, patt]
    else:
        layers = [uniform]

    stack = Stack(
        incidence=_const_eps_fn(1.0 + 0j),      # real incidence (lossless air)
        layers=layers,
        transmission=_const_eps_fn(2.25 + 0j),
        lattice=lat,
        grid_shape=(nx, ny),
    )
    return stack, wl


# ---------------------------------------------------------------------------
# Part A: axis order
# ---------------------------------------------------------------------------

class TestAxisOrder:
    def test_uniform_layer_trailing_dims(self):
        """Uniform layer eps must have trailing shape (Ny, Nx)."""
        nx, ny = 7, 5          # asymmetric so we'd catch a swap
        stack, wl = _make_stack(nx, ny, patterned=False)
        spec = stack.spec(wl)
        assert spec.layer_eps.shape[-2:] == (ny, nx), (
            f"Expected trailing (ny={ny}, nx={nx}), got {spec.layer_eps.shape[-2:]}"
        )

    def test_patterned_layer_trailing_dims(self):
        """Patterned layer eps must have trailing shape (Ny, Nx)."""
        nx, ny = 7, 5
        stack, wl = _make_stack(nx, ny, patterned=True)
        spec = stack.spec(wl)
        # layer_eps has shape [N_layers, N_wl, Ny, Nx]
        assert spec.layer_eps.shape[-2:] == (ny, nx), (
            f"Expected trailing (ny={ny}, nx={nx}), got {spec.layer_eps.shape[-2:]}"
        )

    def test_uniform_and_patterned_agree(self):
        """Both uniform and patterned layers in the same stack must have the same trailing shape."""
        nx, ny = 7, 5
        stack, wl = _make_stack(nx, ny, patterned=True)
        spec = stack.spec(wl)
        # layer 0 is uniform, layer 1 is patterned
        assert spec.layer_eps[0].shape[-2:] == spec.layer_eps[1].shape[-2:]


# ---------------------------------------------------------------------------
# Part B: dtype
# ---------------------------------------------------------------------------

class TestDtype:
    def test_layer_eps_is_complex_even_when_real_input(self):
        """layer_eps must be complex even if both eps callables return real."""
        nx, ny = 4, 3
        stack, wl = _make_stack(nx, ny, patterned=True)
        spec = stack.spec(wl)
        assert spec.layer_eps.is_complex(), (
            "layer_eps should be complex but got dtype "
            f"{spec.layer_eps.dtype}"
        )

    def test_eps_transmission_is_complex(self):
        """eps_transmission must be complex."""
        nx, ny = 4, 3
        stack, wl = _make_stack(nx, ny, patterned=False)
        spec = stack.spec(wl)
        assert spec.eps_transmission.is_complex()

    def test_eps_incidence_is_real(self):
        """eps_incidence must be real (lossless incidence medium)."""
        nx, ny = 4, 3
        stack, wl = _make_stack(nx, ny, patterned=False)
        spec = stack.spec(wl)
        assert not spec.eps_incidence.is_complex(), (
            "eps_incidence should be real but got dtype "
            f"{spec.eps_incidence.dtype}"
        )

    def test_lossy_incidence_warns_and_returns_real(self):
        """A complex (lossy) incidence eps triggers a UserWarning and eps_incidence is still real."""
        nx, ny = 4, 3
        lat = Lattice.rectangular(1.0, 1.0)
        wl  = torch.tensor([1.0])
        stack = Stack(
            incidence=_const_eps_fn(2.25 + 0.5j),  # lossy incidence
            layers=[Layer(eps_solid_fn=_const_eps_fn(1.0+0j), thickness=torch.tensor(0.1))],
            transmission=_const_eps_fn(1.0+0j),
            lattice=lat,
            grid_shape=(nx, ny),
        )
        with pytest.warns(UserWarning, match="eps_incidence"):
            spec = stack.spec(wl)
        assert not spec.eps_incidence.is_complex()

    def test_lossless_incidence_no_warning(self):
        """A genuinely real (or zero-imag complex) incidence eps must NOT warn."""
        nx, ny = 4, 3
        stack, wl = _make_stack(nx, ny, patterned=False)
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            spec = stack.spec(wl)    # should not raise
        assert not spec.eps_incidence.is_complex()


# ---------------------------------------------------------------------------
# utils: to_complex / to_real unit tests
# ---------------------------------------------------------------------------

class TestToComplex:
    def test_real_promoted(self):
        t = torch.tensor([1.0, 2.0])
        c = to_complex(t)
        assert c.is_complex()
        assert c.dtype == torch.complex64

    def test_complex_passthrough(self):
        t = torch.tensor([1.0+2j, 3.0+4j])
        c = to_complex(t)
        assert c is t  # no copy


class TestToReal:
    def test_real_passthrough(self):
        t = torch.tensor([1.0, 2.0])
        r = to_real(t)
        assert r is t

    def test_pure_complex_no_warning(self):
        t = torch.tensor([1.0 + 0j, 2.0 + 0j])
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            r = to_real(t)
        assert not r.is_complex()

    def test_lossy_warns(self):
        t = torch.tensor([1.0 + 0.5j])
        with pytest.warns(UserWarning, match="non-negligible"):
            r = to_real(t, name="test_tensor")
        assert not r.is_complex()
