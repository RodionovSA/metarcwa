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

shape_fn_0 = from_metashapes(rect, soft = True, softness = 0.01)

mask_0 = shape_fn_0(lattice, 256, 256)

# test for eps_fn

## Test epsilon with a material like gold:
## Create gold from DisperTorch, wrap it with `from_dispertorch`
## call `eps_fn(wl)` and inspect output


from dispertorch import material, list_materials

print(list_materials())

def from_dispertorch(dispersion) -> Callable:
    """ 
    Convers a DisperTorch dispersion model into eps_fn(wavelength).
    """
    def eps_fn(wl):
        return dispersion.permittivity(wl)

    return eps_fn

au = material("Au")
wl = torch.linspace(400,800,50)

eps_fn_0 = from_dispertorch(au)
eps_0 = eps_fn_0(wl)

print(eps_0.shape)
print(eps_0.dtype)

plt.plot(wl, eps_0.real, label = "Real component of permittivity")
plt.plot(wl, eps_0.imag, label = "Imaginary component of permittivity")
plt.legend()
plt.show()

# Test for layer of a rectangle shape in a lattice with material being 
# gold and air

# permittivity of material 1 which is gold
eps_solid_0 = eps_0

# permittivity of material 2 which is air
eps_void_0 = torch.ones_like(eps_0)

# combine mask and permittivity to create a 2D layer 
# eps = mask * eps_solid + (1 - mask) * eps_void 

# Need to make then a shape which allows multiplication
# mask has shape [Nx, Ny]
# eps_solid has shape [wl]
# eps_void has shape [wl]

eps_solid_0 = eps_solid_0[:, None, None]        # [N_wl, 1, 1]
eps_void_0 = eps_void_0[:, None, None]          # [N_wl, 1, 1]
mask_0 = mask_0[None, :, :]                     # [1, Nx, Ny]

eps_layer_0 = mask_0 * eps_solid_0 + torch.subtract(torch.ones_like(mask_0), mask_0) * eps_void_0

print(eps_solid_0.shape)          
# torch.Size([50, 1, 1])
print(eps_void_0.shape)
# torch.Size([50, 1, 1])
print(mask_0.shape)
# torch.Size([1, 256, 256])
print(eps_layer_0.shape)
# torch.Size([50, 256, 256])
print(eps_layer_0.dtype)
# torch.complex64

# Check what the layer looks like for the first wavelength, eps_layer[0]
# Plot real permittivity

plt.imshow(eps_layer_0[0].real)
plt.colorbar()
plt.title(
    f"Real Permittivity Distribution\n"
    f"Au rectangle in air, wavelength = {wl[0]} nm"
    )
plt.show()

# Plot imaginary permittivity

plt.imshow(eps_layer_0[0].imag)
plt.colorbar()
plt.title(
    f"Imaginary Permittivity Distribution\n"
    f"Au rectangle in air, wavelength = {wl[0]} nm"
          )
plt.show()

# Create a stack

## Aim is to create multiple layers and then stack them:
## [N_layer, N_wl, Nx, Ny]

# The layer created before will be referred to as layer 0


from metashapes.shape import ConvexQuad
from metashapes.shape import Cross
from metashapes.shape import Ellipse

# Lattice
Lx = 400
Ly = 400
lattice = Lattice.rectangular(Lx,Ly)

centre_1 = (200,200)
size_1 = (200,200)
angle_1 = 60

convexquad = ConvexQuad(center=centre_1, size = size_1, angle=angle_1)

# Define a new mask for a new layer_1

shape_fn_1 = from_metashapes(convexquad, soft = True, softness = 0.01)

mask_1 = shape_fn_1(lattice, 256, 256)

# Define a new mask for a another new layer_2

centre_2 = (200,200)
length_2 = 200
width_2 = 60
angle_2 = 45.0
outer_corner_radius = 0.0
inner_corner_radius = 0.0

cross = Cross(center = centre_2, length = length_2, width = width_2, angle = angle_2,
              outer_corner_radius = outer_corner_radius, inner_corner_radius = inner_corner_radius)

shape_fn_2 = from_metashapes(cross, soft = True, softness = 0.1)

mask_2 = shape_fn_2(lattice, 256, 256)

# Define a new mask for another new layer_3

centre_3 = (200,200)
axes_3 = (100,50)

ellipse = Ellipse(center = centre_3, axes = axes_3 )

shape_fn_3 = from_metashapes(ellipse, soft = True, softness = 0.01)

mask_3 = shape_fn_3(lattice, 256, 256)

# Add different materials to each layer

# Layer 1, eps

# we have gold (au) defined from earlier
# Will define some more materials 

# Silicon Dioxide will use for layer 1
si_O2 = material("SiO2")

eps_fn_1 = from_dispertorch(si_O2)
eps_1 = eps_fn_1(wl)

# permittivity of material 1 in this layer 
eps_solid_1 = eps_1

# permittivity of material 2 in this layer which is air
eps_void_1 = torch.ones_like(eps_0)

# combine mask and permittivity to create a 2D layer 
# eps = mask * eps_solid + (1 - mask) * eps_void 

# Need to make then a shape which allows multiplication
# mask has shape [Nx, Ny]
# eps_solid has shape [wl]
# eps_void has shape [wl]

eps_solid_1 = eps_solid_1[:, None, None]        # [N_wl, 1, 1]
eps_void_1 = eps_void_1[:, None, None]          # [N_wl, 1, 1]
mask_1 = mask_1[None, :, :]                     # [1, Nx, Ny]

# [N_wl, Nx, Ny]
eps_layer_1 = mask_1 * eps_solid_1 + torch.subtract(torch.ones_like(mask_1), mask_1) * eps_void_1

# Silicon Nitride will use for layer 2
si3_n4 = material("Si3N4")

eps_fn_2 = from_dispertorch(si3_n4)
eps_2 = eps_fn_2(wl)

# permittivity of material 1 in this layer 
eps_solid_2 = eps_2

# permittivity of material 2 in this layer which is air
eps_void_2 = torch.ones_like(eps_0)

# combine mask and permittivity to create a 2D layer 
# eps = mask * eps_solid + (1 - mask) * eps_void 

# Need to make then a shape which allows multiplication
# mask has shape [Nx, Ny]
# eps_solid has shape [wl]
# eps_void has shape [wl]

eps_solid_2 = eps_solid_2[:, None, None]        # [N_wl, 1, 1]
eps_void_2 = eps_void_2[:, None, None]          # [N_wl, 1, 1]
mask_2 = mask_2[None, :, :]                     # [1, Nx, Ny]

# [N_wl, Nx, Ny]
eps_layer_2 = mask_2 * eps_solid_2 + torch.subtract(torch.ones_like(mask_2), mask_2) * eps_void_2

# Will use au / gold for layer 3

# reshape mask_3
mask_3 = mask_3[None, :, :]                     # [1, Nx, Ny]

eps_layer_3 = mask_3 * eps_solid_0 + (1 - mask_3) * eps_void_0


# Now stack all the layers together
# Currently our eps_layer has shape [N_wl, Nx, Ny]
# By stacking, our shape should look like [N_layer, N_wl, Nx, Ny]

eps_stack = torch.stack([eps_layer_0, eps_layer_1, eps_layer_2, eps_layer_3], dim=0)

print(eps_layer_0.shape)
print(eps_layer_1.shape)
print(eps_layer_2.shape)
print(eps_layer_3.shape)

print(eps_stack.shape)









