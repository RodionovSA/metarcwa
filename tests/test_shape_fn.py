# Test for shape_fn 

## Create a MetaShapes rectangle, lattice, call `shape_fn(lattice, nx, ny)` and check
## it returns a mask.

import matplotlib.pyplot as plt
from typing import Callable

import torch

from metashapes import UnitCell, Lattice
from metashapes.shape import Rectangle

# Lattice
Lx = 400
Ly = 400
lattice = Lattice.rectangular(Lx,Ly)

centre = (200,200)
size = (200,200)
angle = 0

rect = Rectangle(center=centre, size = size, angle=angle)

def from_metashapes(shape, soft, softness) -> Callable:
    """Convert a MetaShapes Shape object into a shape_fn(lattice, nx, ny):

    Parameters:
    ---------------
    shape:
        MetaShapes shape object

    soft: Boolean
        Inside our mask, every pixel will either be a 1 or 0 depending
        the material occupying that specific pixel. 

        0 to 1 gives a sudden jump at those boundaries.

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

        Arguments of shape:

            lattice:
                The shape input only gives the position, size and angle of the shape in the 
                unit cell. It doesn't give any information about the periodicity of this 
                unit cell. The lattice provides the lattice vectors, unit cell dimensions and the
                coordiante system (cartesian or fractional).
            grid resolution (nx,ny):
                This gives the number of pixels in the x and y direction of the unit cell. The higher
                the nx and ny, the higher the resolution.
    """

    try:
        from metashapes import UnitCell
    except:
        raise ImportError("You should have installed metashapes")
    
    def shape_fn(lattice, nx, ny):
        cell = UnitCell(lattice = lattice, scene=shape)
        return cell.mask(nx = nx, ny=ny, soft=soft, softness=softness)
    
    return shape_fn

shape_fn = from_metashapes(rect, soft = True, softness = 0.01)

mask = shape_fn(lattice, 256, 256)

print(type(mask))
print(mask.shape)
print(mask.dtype)
print(mask.min())
print(mask.max())

plt.imshow(mask)
plt.colorbar()
plt.show()


