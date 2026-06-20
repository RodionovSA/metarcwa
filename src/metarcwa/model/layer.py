# metarcwa/model/layer.py
# DESCRIPTION

import torch
import torch.nn as nn

from typing import Optional

from .utils import register, CallableModule

class Layer(nn.Module):
    """A single layer of the stack: a patterned or uniform slab.

    The in-plane permittivity is defined by a shape mask combined with two
    permittivity callables. With no shape_fn the layer is uniform: only
    eps_solid_fn is used, and eps_void_fn must be omitted. Thickness sets the
    extent along the propagation axis.

    Parameters
    ----------
    eps_solid_fn : CallableModule
        Permittivity, called as ``eps_fn(wavelength) -> eps``. Receives
        wavelength; returns complex permittivity (isotropic scalar, one ε per
        point), broadcastable to ``[..., Ny, Nx]`` — a non-dispersive material
        returns a complex scalar, a dispersive one may return ``[N_wl]``.

        **Parameter visibility:** for the callable's tensors to appear in
        ``model.parameters()`` / ``model.buffers()`` and move with
        ``model.to()``, the callable must be an ``nn.Module`` (PyTorch only
        traverses modules, not plain functions or closures). Use the provided
        helpers: ``from_dispertorch(disp)`` wraps a DisperTorch model;
        ``CallableModule(fn, dep1, dep2, …)`` wraps any plain callable and
        registers its module/parameter dependencies. A plain ``lambda`` or
        closure will work numerically but its closed-over tensors are invisible
        to the model.

        Reserved (unsupported): anisotropy via 3x3 trailing dims; magnetic
        response via a separate mu callable. Current contract is isotropic,
        non-magnetic.
        
    thickness : float | Tensor | nn.Parameter
        Layer thickness. May be an nn.Parameter for inverse design.
        
    eps_void_fn : CallableModule, optional
        Permittivity of the void (un-masked) region, same contract as
        ``eps_solid_fn``. Fills the ``mask == 0`` region in the blend
        ``eps = mask * eps_solid + (1 - mask) * eps_void``. Required iff
        ``shape_fn`` is given; for a uniform layer (no ``shape_fn``) it must
        be omitted and only ``eps_solid_fn`` is used.
        
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
        eps_solid_fn: CallableModule,
        thickness,
        eps_void_fn: Optional[CallableModule] = None,
        shape_fn: Optional[CallableModule] = None,
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

    