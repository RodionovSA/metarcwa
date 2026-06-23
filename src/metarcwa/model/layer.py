# metarcwa/model/layer.py
# DESCRIPTION

import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Optional

from .utils import register, CallableModule
from .medium import Medium, MediumSpec
from .lattice import Lattice

@dataclass(frozen=True)
class HomogeneousLayer:
    thickness: torch.Tensor
    medium: MediumSpec

@dataclass(frozen=True)
class PatternedLayer:
    thickness: torch.Tensor
    medium_solid: MediumSpec
    medium_void: MediumSpec
    pattern: torch.Tensor


class Layer(nn.Module):
    """A single layer of the stack: a patterned or uniform slab.

    The in-plane permittivity (and permeability, if magnetic) is defined by
    a shape mask combined with two Medium instances. With no shape_fn the
    layer is uniform: only medium_solid is used, and medium_void must be
    omitted. Thickness sets the extent along the propagation axis.

    Parameters
    ----------
    medium_solid : Medium
        Fill material of the masked (shape_fn == 1) region, or the sole
        material for a uniform layer. One of ``Medium`` subclasses
        — see that class's docstring for its eps_fn/mu_fn contract 
        and parameter-visibility requirements.
        
    thickness : float | Tensor | nn.Parameter
        Layer thickness. May be an nn.Parameter for inverse design.
        
    medium_void : Medium, optional
        Material of the un-masked region. Required iff shape_fn is given;
        for a uniform layer (no shape_fn) it must be omitted. Must be the
        same Medium subclass as medium_solid — mixing variants (e.g. an
        isotropic void in an anisotropic solid) is not supported; promote
        the simpler material to the richer variant explicitly instead.
        Fills the ``mask == 0`` region in the blend
        ``eps = mask * eps_solid + (1 - mask) * eps_void`` (and likewise
        for mu, when both media are magnetic).
        
    shape_fn : CallableModule, optional
        Geometry mask, called as ``shape_fn(lattice, nx, ny) -> mask``.
        Receives the lattice and grid resolution (integers); returns a real
        mask in [0, 1] of shape ``[Ny, Nx]`` (rows = y, cols = x — the
        standard image convention, matching metashapes output) or
        ``[N_geom, Ny, Nx]`` for batched geometry. Same ``nn.Module``
        requirement as ``eps_solid_fn`` — use ``from_metashapes(shape, …)``
        or ``CallableModule(fn, dep, …)`` to make geometry parameters visible
        to the model. Required iff ``eps_void_fn`` is given; omit for a
        uniform layer.
    """

    def __init__(
        self,
        medium_solid: Medium,
        thickness,
        medium_void: Optional[Medium] = None,
        shape_fn: Optional[CallableModule] = None,
    ):
        super().__init__()

        if not isinstance(medium_solid, Medium):
            raise TypeError("medium_solid must be a Medium.")
        if medium_void is not None and not isinstance(medium_void, Medium):
            raise TypeError("medium_void must be a Medium.")
        if shape_fn is not None and not callable(shape_fn):
            raise TypeError("shape_fn must be callable.")

        if shape_fn is not None and medium_void is None:
            raise ValueError("Patterned layer (shape_fn given) requires medium_void.")
        if shape_fn is None and medium_void is not None:
            raise ValueError("Uniform layer (no shape_fn) must omit medium_void.")
        if medium_void is not None and type(medium_void) is not type(medium_solid):
            raise TypeError("medium_solid and medium_void must be the same Medium variant.")

        register(self, "thickness", thickness)
        self.medium_solid = medium_solid
        self.medium_void = medium_void
        self.shape_fn = shape_fn
        
    def spec(
        self,
        wvl: torch.Tensor,
        lattice: Optional["Lattice"] = None,
        nx: Optional[int] = None,
        ny: Optional[int] = None,
    ) -> HomogeneousLayer | PatternedLayer:
        """Resolve this Layer at a given wavelength into a Solver-facing spec.
        lattice/nx/ny are required iff this is a patterned layer (shape_fn set);
        """
        medium_solid_spec = self.medium_solid.spec(wvl)

        if self.shape_fn is None:
            return HomogeneousLayer(
                thickness=self.thickness,
                medium=medium_solid_spec,
            )

        if lattice is None or nx is None or ny is None:
            raise ValueError("Patterned layer requires lattice, nx, ny from Stack.")

        pattern = self.shape_fn(lattice, nx, ny)              
        medium_void_spec = self.medium_void.spec(wvl)

        return PatternedLayer(
            thickness=self.thickness,
            medium_solid=medium_solid_spec,
            medium_void=medium_void_spec,
            pattern=pattern,
        )
    

    