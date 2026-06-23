import pytest
import torch

pytest.importorskip("metashapes")
pytest.importorskip("dispertorch")

from metashapes.shape import Rectangle, Cross, Ellipse
from dispertorch import material

from metarcwa.model.lattice import Lattice
from metarcwa.model.layer import Layer, HomogeneousLayer, PatternedLayer
from metarcwa.model.stack import Stack
from metarcwa.model.medium import IsotropicMedium, Medium
from metarcwa.model.utils import from_metashapes, from_dispertorch

NX, NY = 32, 32


# ---------------------------------------------------------------------------
# Layer construction
# ---------------------------------------------------------------------------

def test_patterned_layer_construction(au_medium, air_medium, rect_shape_fn):
    layer = Layer(
        medium_solid=au_medium,
        thickness=100.0,
        medium_void=air_medium,
        shape_fn=rect_shape_fn,
    )
    assert layer.thickness.item() == pytest.approx(100.0)
    assert layer.shape_fn is rect_shape_fn
    assert layer.medium_void is air_medium
    assert layer.medium_solid is au_medium


def test_uniform_layer_construction(au_medium):
    layer = Layer(medium_solid=au_medium, thickness=50.0)
    assert layer.thickness.item() == pytest.approx(50.0)
    assert layer.shape_fn is None
    assert layer.medium_void is None


# ---------------------------------------------------------------------------
# Layer validation
# ---------------------------------------------------------------------------

def test_patterned_layer_requires_medium_void(au_medium, rect_shape_fn):
    with pytest.raises(ValueError, match="medium_void"):
        Layer(medium_solid=au_medium, thickness=100.0, shape_fn=rect_shape_fn)


def test_uniform_layer_forbids_medium_void(au_medium, air_medium):
    with pytest.raises(ValueError, match="medium_void"):
        Layer(medium_solid=au_medium, thickness=100.0, medium_void=air_medium)


def test_medium_solid_must_be_medium_instance(air_medium, rect_shape_fn):
    with pytest.raises(TypeError, match="medium_solid"):
        Layer(
            medium_solid=lambda wl: torch.ones(wl.shape),
            thickness=100.0,
            medium_void=air_medium,
            shape_fn=rect_shape_fn,
        )


def test_medium_void_must_be_medium_instance(au_medium, rect_shape_fn):
    with pytest.raises(TypeError, match="medium_void"):
        Layer(
            medium_solid=au_medium,
            thickness=100.0,
            medium_void=lambda wl: torch.ones(wl.shape),
            shape_fn=rect_shape_fn,
        )


def test_mixed_medium_types_raises(au_medium, rect_shape_fn):
    class OtherMedium(Medium):
        def spec(self, wvl):
            pass

    with pytest.raises(TypeError, match="same Medium variant"):
        Layer(
            medium_solid=au_medium,
            thickness=100.0,
            medium_void=OtherMedium(),
            shape_fn=rect_shape_fn,
        )


# ---------------------------------------------------------------------------
# Stack.spec — patterned layers
# ---------------------------------------------------------------------------

def test_single_patterned_layer_spec(au_medium, air_medium, rect_shape_fn, lattice, wl):
    layer = Layer(
        medium_solid=au_medium, thickness=100.0,
        medium_void=air_medium, shape_fn=rect_shape_fn,
    )
    stack = Stack(incidence=air_medium, layers=[layer],
                  transmission=air_medium, lattice=lattice)
    spec = stack.spec(wl, NX, NY)

    assert len(spec.layers) == 1
    s = spec.layers[0]
    assert isinstance(s, PatternedLayer)
    assert s.pattern.shape == (NY, NX)
    assert s.medium_solid.eps.is_complex()
    assert s.medium_solid.eps.shape == wl.shape
    assert s.medium_void.eps.is_complex()
    assert s.thickness.item() == pytest.approx(100.0)
    assert spec.a1.shape == (2,)
    assert spec.a2.shape == (2,)


def test_single_uniform_layer_spec(au_medium, air_medium, lattice, wl):
    layer = Layer(medium_solid=au_medium, thickness=50.0)
    stack = Stack(incidence=air_medium, layers=[layer],
                  transmission=air_medium, lattice=lattice)
    spec = stack.spec(wl, NX, NY)

    assert len(spec.layers) == 1
    s = spec.layers[0]
    assert isinstance(s, HomogeneousLayer)
    assert s.medium.eps.is_complex()
    assert s.medium.eps.shape == wl.shape
    assert s.thickness.item() == pytest.approx(50.0)


def test_multilayer_stack_spec(au_medium, air_medium, lattice, wl):
    rect = Rectangle(center=(200, 200), size=(200, 200), angle=0)
    layer_0 = Layer(
        medium_solid=au_medium, thickness=100.0, medium_void=air_medium,
        shape_fn=from_metashapes(rect, soft=True, softness=0.01),
    )

    sio2_medium = IsotropicMedium(from_dispertorch(material("SiO2")))
    cross = Cross(center=(200, 200), length=200, width=60, angle=45.0,
                  outer_corner_radius=0.0, inner_corner_radius=0.0)
    layer_1 = Layer(
        medium_solid=sio2_medium, thickness=80.0, medium_void=air_medium,
        shape_fn=from_metashapes(cross, soft=True, softness=0.1),
    )

    ellipse = Ellipse(center=(200, 200), axes=(100, 50))
    layer_2 = Layer(
        medium_solid=au_medium, thickness=50.0, medium_void=air_medium,
        shape_fn=from_metashapes(ellipse, soft=True, softness=0.01),
    )

    stack = Stack(incidence=air_medium, layers=[layer_0, layer_1, layer_2],
                  transmission=air_medium, lattice=lattice)
    spec = stack.spec(wl, NX, NY)

    assert len(spec.layers) == 3
    for s in spec.layers:
        assert isinstance(s, PatternedLayer)
        assert s.medium_solid.eps.is_complex()
        assert s.medium_solid.eps.shape == wl.shape
        assert s.pattern.shape == (NY, NX)

    thicknesses = [100.0, 80.0, 50.0]
    for s, t in zip(spec.layers, thicknesses):
        assert s.thickness.item() == pytest.approx(t)


def test_stack_incidence_transmission_in_spec(au_medium, air_medium, lattice, wl):
    layer = Layer(medium_solid=au_medium, thickness=100.0)
    stack = Stack(incidence=air_medium, layers=[layer],
                  transmission=au_medium, lattice=lattice)
    spec = stack.spec(wl, NX, NY)

    assert spec.incidence.eps.is_complex()
    assert spec.transmission.eps.is_complex()
    assert spec.incidence.eps.shape == wl.shape
    assert spec.transmission.eps.shape == wl.shape
