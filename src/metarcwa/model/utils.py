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
        
def from_metashapes(shape, soft, softness) -> Callable:
    """Convert a MetaShapes Shape object into a shape_fn(lattice, nx, ny):

    Parameters:
    ---------------
    shape:
        MetaShapes shape object

    soft: Boolean
        Inside our mask, every pixel will either be a 1 or 0 depending
        the material occupying that specific pixel. 

        0  to 1 gives a sudden jump at those boundaries.

        This gives discontinuity where the derivative is undefined at these
        jumps. PyTorch does a lot of gradient-based optimisations for which
        this wouldn't be ideal. 

    softness: Float
        The degree to which you smooth the boundary between the 0 and 1
        can be controlled using softness

    Returns
    -----------------
    Callable
        shape_fn(lattice, nx,ny)
    """

    try:
        from metashapes import UnitCell
    except:
        raise ImportError("You should have installed metashapes")
    
    def shape_fn(lattice, nx, ny):
        cell = UnitCell(lattice = lattice, scene=shape)
        return cell.mask(nx = nx, ny=ny, soft=soft, softness=softness)
    
    return shape_fn


def from_dispertorch(dispersion) -> Callable:
    """ 
    Convers a DisperTorch dispersion model into eps_fn(wavelength).
    """
    def eps_fn(wl):
        return dispersion.permittivity(wl)

    return eps_fn