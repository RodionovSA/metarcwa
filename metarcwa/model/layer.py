# metarcwa/model/layer.py
# DESCRIPTION

import torch
import torch.nn as nn

from typing import Callable, Optional

from .material import Material
from .utils import register

class Layer(nn.Module):
    """A single layer of the stack: a patterned or uniform slab.

    The in-plane permittivity is defined by a shape mask combined with two
    materials. With no shape_fn the layer is uniform (material_solid only;
    material_void is ignored). Thickness sets the extent along the
    propagation axis.

    Parameters
    ----------
    material_solid : Material
        Permittivity of the mask foreground (mask == 1). Also the material
        of a uniform layer.
    thickness : float | Tensor | nn.Parameter
        Layer thickness. May be an nn.Parameter for inverse design.
    material_void : Material, optional
        Permittivity of the mask background (mask == 0). Required for a
        patterned layer; ignored when shape_fn is None.
    shape_fn : Callable, optional
        Maps a canvas to a mask tensor: (canvas) -> tensor. None means a
        uniform layer.
    """

    def __init__(
        self,
        material_solid: Material,
        thickness,
        material_void: Optional[Material] = None,
        shape_fn: Optional[Callable] = None,
    ):
        super().__init__()
        if shape_fn is not None and material_void is None:
            raise ValueError("Patterned layer (shape_fn given) requires material_void.")
        register(self, "thickness", thickness)
        self.material_solid = material_solid
        self.material_void = material_void
        self.shape_fn = shape_fn

    