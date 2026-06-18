# tests/test_layer.py

import pytest
import torch

pytest.importorskip("metashapes")
pytest.importorskip("dispertorch")

from metashapes.shape import Rectangle, Cross, Ellipse
from dispertorch import material

from metarcwa.model.lattice import Lattice
from metarcwa.model.layer import Layer
from metarcwa.model.stack import Stack
from metarcwa.model.utils import from_metashapes, from_dispertorch, CallableModule

NX, NY = 32, 32
N_WL = 5


def _air_fn(wl: torch.Tensor) -> torch.Tensor:
    return torch.ones(wl.shape, dtype=torch.complex64)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def lattice():
    return Lattice.rectangular(400, 400)


@pytest.fixture(scope="module")
def wl():
    return torch.linspace(400, 800, N_WL)


@pytest.fixture(scope="module")
def rect_shape_fn():
    rect = Rectangle(center=(200, 200), size=(200, 200), angle=0)
    return from_metashapes(rect, soft=True, softness=0.01)


@pytest.fixture(scope="module")
def au_eps_fn():
    return from_dispertorch(material("Au"))


# ---------------------------------------------------------------------------
# Tests: from_metashapes / from_dispertorch
# ---------------------------------------------------------------------------

def test_from_metashapes_returns_callable_module(rect_shape_fn):
    assert isinstance(rect_shape_fn, CallableModule)
    assert callable(rect_shape_fn)


def test_mask_shape_and_range(rect_shape_fn, lattice):
    mask = rect_shape_fn(lattice, NX, NY)
    assert mask.shape == (NX, NY)
    assert not mask.is_complex()
    assert float(mask.min()) >= 0.0
    assert float(mask.max()) <= 1.0


def test_eps_fn_shape_and_dtype(au_eps_fn, wl):
    eps = au_eps_fn(wl)
    assert eps.shape == (N_WL,)
    assert eps.is_complex()


# ---------------------------------------------------------------------------
# Tests: Layer construction and validation
# ---------------------------------------------------------------------------

def test_layer_construction(au_eps_fn, rect_shape_fn):
    layer = Layer(
        eps_solid_fn=au_eps_fn,
        thickness=100.0,
        eps_void_fn=_air_fn,
        shape_fn=rect_shape_fn,
    )
    assert layer.thickness.item() == pytest.approx(100.0)
    assert layer.shape_fn is rect_shape_fn
    assert layer.eps_void_fn is _air_fn


def test_layer_patterned_requires_eps_void(rect_shape_fn, au_eps_fn):
    with pytest.raises(ValueError, match="eps_void_fn"):
        Layer(
            eps_solid_fn=au_eps_fn,
            thickness=100.0,
            shape_fn=rect_shape_fn,
        )


def test_layer_uniform_forbids_eps_void(au_eps_fn):
    with pytest.raises(ValueError, match="shape_fn"):
        Layer(
            eps_solid_fn=au_eps_fn,
            thickness=100.0,
            eps_void_fn=_air_fn,
        )


# ---------------------------------------------------------------------------
# Tests: Stack.spec
# ---------------------------------------------------------------------------

def test_single_layer_stack_spec(au_eps_fn, rect_shape_fn, lattice, wl):
    layer = Layer(
        eps_solid_fn=au_eps_fn,
        thickness=100.0,
        eps_void_fn=_air_fn,
        shape_fn=rect_shape_fn,
    )
    stack = Stack(
        incidence=_air_fn,
        layers=[layer],
        transmission=_air_fn,
        lattice=lattice,
        grid_shape=(NX, NY),
    )
    spec = stack.spec(wl)

    assert spec.layer_eps.shape == (1, N_WL, NX, NY)
    assert spec.layer_eps.is_complex()
    assert spec.layer_thickness.shape == (1,)
    assert spec.a1.shape == (2,)
    assert spec.a2.shape == (2,)


def test_multilayer_stack_spec(au_eps_fn, lattice, wl):
    # Layer 0: Au rectangle
    rect = Rectangle(center=(200, 200), size=(200, 200), angle=0)
    layer_0 = Layer(
        eps_solid_fn=au_eps_fn,
        thickness=100.0,
        eps_void_fn=_air_fn,
        shape_fn=from_metashapes(rect, soft=True, softness=0.01),
    )

    # Layer 1: SiO2 cross
    cross = Cross(center=(200, 200), length=200, width=60, angle=45.0,
                  outer_corner_radius=0.0, inner_corner_radius=0.0)
    layer_1 = Layer(
        eps_solid_fn=from_dispertorch(material("SiO2")),
        thickness=80.0,
        eps_void_fn=_air_fn,
        shape_fn=from_metashapes(cross, soft=True, softness=0.1),
    )

    # Layer 2: Au ellipse
    ellipse = Ellipse(center=(200, 200), axes=(100, 50))
    layer_2 = Layer(
        eps_solid_fn=au_eps_fn,
        thickness=50.0,
        eps_void_fn=_air_fn,
        shape_fn=from_metashapes(ellipse, soft=True, softness=0.01),
    )

    stack = Stack(
        incidence=_air_fn,
        layers=[layer_0, layer_1, layer_2],
        transmission=_air_fn,
        lattice=lattice,
        grid_shape=(NX, NY),
    )
    spec = stack.spec(wl)

    assert spec.layer_eps.shape == (3, N_WL, NX, NY)
    assert spec.layer_eps.is_complex()
    assert spec.layer_thickness.shape == (3,)
