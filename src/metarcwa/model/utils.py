# metarcwa/model/utils.py
# DESCRIPTION

import torch
import torch.nn as nn
from typing import Callable

def register(module, name, value, dtype=torch.float32):
    """Register `value` on `module` under `name`.

    If `value` is an nn.Parameter it becomes an optimizable parameter;
    otherwise it is stored as a (non-gradient) buffer that still moves
    with .to() and is saved in state_dict().
    """
    if isinstance(value, nn.Parameter):
        setattr(module, name, value)
    else:
        module.register_buffer(name, torch.as_tensor(value, dtype=dtype))
        
def from_metashapes(shape, **kwargs) -> Callable:
    """ Takes Shape object from metshapes + rasterisation params (smoothness, ...) as kwargs 
    and outputs shape_fn(lattice, nx, ny)
    """
    shape_fn = ...
    return shape_fn

def from_dispertorch(dispersion, **kwargs) -> Callable:
    """ Takes ModelDispersion object from dispertorch + some kwargs 
    and outputs eps_fn(wavelength). wavelength in **nm**
    """
    eps_fn = ...
    return eps_fn