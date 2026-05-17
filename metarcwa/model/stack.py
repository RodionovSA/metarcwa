# metarcwa/model/stack.py
# DESCRIPTION

from typing import Sequence
import torch
import torch.nn as nn

from .layer import Layer
from .lattice import Lattice
from .material import Material
from .spec import StackSpec


class Stack(nn.Module):
    """An ordered stack of finite layers between two semi-infinite media.

    Owns the system-wide invariants — lattice and grid shape — and builds
    the canvas passed to each layer's shape_fn. The incidence and
    transmission media are semi-infinite half-spaces: they have a material
    but no thickness and no pattern, so they are stored separately from the
    finite layers.

    Parameters
    ----------
    incidence : Material
        Semi-infinite medium on the incidence side.
    layers : Sequence[Layer]
        Ordered finite layers, incidence side first.
    transmission : Material
        Semi-infinite medium on the transmission side.
    lattice : Lattice
        In-plane periodicity, shared by all layers.
    grid_shape : tuple[int, int]
        Real-space sampling (Nx, Ny) of the unit cell.
    """

    def __init__(
        self,
        incidence: Material,
        layers: Sequence[Layer],
        transmission: Material,
        lattice: Lattice,
        grid_shape: tuple[int, int],
    ):
        super().__init__()
        nx, ny = grid_shape
        if not (isinstance(nx, int) and isinstance(ny, int) and nx > 0 and ny > 0):
            raise ValueError(f"grid_shape must be two positive ints, got {grid_shape}.")

        self.incidence = incidence
        self.layers = nn.ModuleList(layers)
        self.transmission = transmission
        self.lattice = lattice
        self.grid_shape = grid_shape

    def _build_canvas(self):
        """Construct a fresh canvas from lattice + grid_shape.

        Called once per spec() call — never cached — so geometry is
        re-rasterized each time and inverse design does not reuse stale
        coordinates.
        """
        ...

    def spec(self, wavelength: torch.Tensor) -> StackSpec:
        canvas = self._build_canvas()

        eps_layers = []
        thicknesses = []
        for layer in self.layers:
            # Problem with uniform and patterned layers stacking. Dim mismatch
            eps_layers.append(self._layer_epsilon(layer, canvas, wavelength))
            thicknesses.append(layer.thickness)

        return StackSpec(
            layer_eps=torch.stack(eps_layers, dim=0),
            layer_thickness=torch.stack(thicknesses, dim=0),
            eps_incidence=self.incidence.epsilon(wavelength),
            eps_transmission=self.transmission.epsilon(wavelength),
            lattice=self.lattice,
        )

    def _layer_epsilon(self, layer: Layer, canvas, wavelength: torch.Tensor) -> torch.tensor:
        """Combine a layer's shape mask and materials into a grid permittivity."""
        eps_solid = layer.material_solid.epsilon(wavelength)
        if layer.shape_fn is None:
            nx, ny = self.grid_shape
            return eps_solid * torch.ones(nx, ny, dtype=eps_solid.dtype)
        mask = layer.shape_fn(canvas)
        eps_void = layer.material_void.epsilon(wavelength)
        return mask * eps_solid + (1.0 - mask) * eps_void