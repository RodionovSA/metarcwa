# metarcwa/model/layer.py
# DESCRIPTION

import torch
import torch.nn as nn

from typing import Callable, Optional

from .utils import register

class Layer(nn.Module):
    """A single layer of the stack: a patterned or uniform slab.

    The in-plane permittivity is defined by a shape mask combined with two
    permittivity callables. With no shape_fn the layer is uniform: only
    eps_solid_fn is used, and eps_void_fn must be omitted. Thickness sets the
    extent along the propagation axis.

    Parameters
    ----------
    eps_solid_fn : Callable
        Permittivity, called as ``eps_fn(wavelength) -> eps``. Receives
        wavelength; returns complex permittivity (isotropic scalar, one ε per
        point), broadcastable to ``[..., Nx, Ny]`` — a non-dispersive material
        returns a complex scalar, a dispersive one may return ``[N_wl]``. If the callable 
        is an ``nn.Module``, it registers as a submodule and its parameters are optimizable.

        Reserved (unsupported): anisotropy via 3x3 trailing dims; magnetic
        response via a separate mu callable. Current contract is isotropic,
        non-magnetic.
        
    thickness : float | Tensor | nn.Parameter
        Layer thickness. May be an nn.Parameter for inverse design.
        
    eps_void_fn : Callable, optional
        Permittivity of the void (un-masked) region, same contract as
        ``eps_solid_fn``. Fills the ``mask == 0`` region in the blend
        ``eps = mask * eps_solid + (1 - mask) * eps_void``. Required iff
        ``shape_fn`` is given; for a uniform layer (no ``shape_fn``) it must
        be omitted and only ``eps_solid_fn`` is used.
        
    shape_fn : Callable, optional
        Geometry mask, called as ``shape_fn(lattice, Nx, Ny) -> mask``.
        Receives the lattice and grid resolution; returns a real mask in
        [0, 1] of shape ``[Nx, Ny]`` (or ``[N_geom, Nx, Ny]`` for batched
        geometry). Any geometry parameters are closed over by the callable; an
        ``nn.Parameter`` so closed is optimizable. Required iff ``eps_void_fn``
        is given; omit for a uniform layer.
    """

    def __init__(
        self,
        eps_solid_fn: Callable,
        thickness,
        eps_void_fn: Optional[Callable] = None,
        shape_fn: Optional[Callable] = None,
    ):
        super().__init__()

        if not callable(eps_solid_fn):
            raise TypeError("eps_solid_fn must be callable.")
        if eps_void_fn is not None and not callable(eps_void_fn):
            raise TypeError("eps_void_fn must be callable.")
        if shape_fn is not None and not callable(shape_fn):
            raise TypeError("shape_fn must be callable.")

        if shape_fn is not None and eps_void_fn is None:
            raise ValueError("Patterned layer (shape_fn given) requires eps_void_fn.")
        if shape_fn is None and eps_void_fn is not None:
            raise ValueError("Uniform layer (no shape_fn) must omit eps_void_fn.")

        register(self, "thickness", thickness)
        self.eps_solid_fn = eps_solid_fn
        self.eps_void_fn = eps_void_fn
        self.shape_fn = shape_fn

    