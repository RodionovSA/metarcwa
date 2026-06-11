# metarcwa/model/stack.py
# DESCRIPTION

from typing import Callable, Sequence
import torch
import torch.nn as nn

from .layer import Layer
from .lattice import Lattice
from .spec import StackSpec


class Stack(nn.Module):
    """An ordered stack of finite layers between two semi-infinite media.

    Owns the system-wide invariants — lattice and grid shape — shared by all
    layers. The incidence and transmission media are semi-infinite half-spaces:
    they have permittivity but no thickness and no pattern, so they are stored
    separately from the finite layers.

    Parameters
    ----------
    incidence : Callable
        Permittivity of the semi-infinite incidence medium, called as
        ``eps_fn(wavelength) -> eps``. Same contract as
        ``Layer.eps_solid_fn`` but returns a scalar-like value (no Nx, Ny).
    layers : Sequence[Layer]
        Ordered finite layers, incidence side first.
    transmission : Callable
        Permittivity of the semi-infinite transmission medium, same contract
        as ``incidence``.
    lattice : Lattice
        In-plane periodicity, shared by all layers.
    grid_shape : tuple[int, int]
        Real-space sampling (Nx, Ny) of the unit cell.
    """

    def __init__(
        self,
        incidence: Callable,
        layers: Sequence[Layer],
        transmission: Callable,
        lattice: Lattice,
        grid_shape: tuple[int, int],
    ):
        super().__init__()
        nx, ny = grid_shape
        if not (isinstance(nx, int) and isinstance(ny, int) and nx > 0 and ny > 0):
            raise ValueError(f"grid_shape must be two positive ints, got {grid_shape}.")
        if not callable(incidence):
            raise TypeError("incidence must be callable.")
        if not callable(transmission):
            raise TypeError("transmission must be callable.")

        self.incidence = incidence
        self.layers = nn.ModuleList(layers)
        self.transmission = transmission
        self.lattice = lattice
        self.grid_shape = grid_shape

    def spec(self, wavelength: torch.Tensor) -> StackSpec:
        eps_layers = [self._layer_epsilon(layer, wavelength) for layer in self.layers]
        thicknesses = [layer.thickness for layer in self.layers]

        eps_layers = torch.broadcast_tensors(*eps_layers)
        return StackSpec(
            layer_eps=torch.stack(eps_layers, dim=0),
            layer_thickness=torch.stack(thicknesses, dim=0),
            eps_incidence=self.incidence(wavelength),
            eps_transmission=self.transmission(wavelength),
            lattice=self.lattice,
        )

    def _layer_epsilon(self, layer: Layer, wavelength: torch.Tensor) -> torch.Tensor:
        nx, ny = self.grid_shape
        eps_solid = layer.eps_solid_fn(wavelength)
        if layer.shape_fn is None:
            return eps_solid.unsqueeze(-1).unsqueeze(-1).expand(*eps_solid.shape, nx, ny)
        mask = layer.shape_fn(self.lattice, nx, ny)
        eps_void = layer.eps_void_fn(wavelength)
        return mask * eps_solid + (1.0 - mask) * eps_void


