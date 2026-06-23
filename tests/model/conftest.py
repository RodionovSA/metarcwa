import pytest
import torch

from metarcwa.model.lattice import Lattice
from metarcwa.model.medium import IsotropicMedium
from metarcwa.model.utils import CallableModule, from_metashapes, from_dispertorch

N_WL = 5


@pytest.fixture(scope="module")
def lattice():
    return Lattice.rectangular(400, 400)


@pytest.fixture(scope="module")
def wl():
    return torch.linspace(400, 800, N_WL)


@pytest.fixture(scope="module")
def air_medium():
    return IsotropicMedium(
        CallableModule(lambda wl: torch.ones(wl.shape, dtype=torch.complex64))
    )


@pytest.fixture(scope="module")
def rect_shape_fn():
    pytest.importorskip("metashapes")
    from metashapes.shape import Rectangle
    return from_metashapes(
        Rectangle(center=(200, 200), size=(200, 200), angle=0),
        soft=True, softness=0.01,
    )


@pytest.fixture(scope="module")
def au_eps_fn():
    dt = pytest.importorskip("dispertorch")
    return from_dispertorch(dt.material("Au"))


@pytest.fixture(scope="module")
def au_medium(au_eps_fn):
    return IsotropicMedium(au_eps_fn)
