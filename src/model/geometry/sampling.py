# src/model/geometry/sampling.py
# Functions for sampling geometric shapes onto a grid

from typing import Any, Tuple

from src.model.geometry.lattice import Lattice
from src.model.material import BaseMaterial
from src.backend import Backend

def matmap(backend: "Backend", 
           bitmap: Any, 
           matval: complex, 
           matbg: complex):
    '''
    Compute the real-space material distribution for the object.

    Parameters
    ----------
    backend : Backend
        Computational backend.
    bitmap: Any,
        Bitmap representation of the object.
    matval : complex
        Material value tensor inside the shape (B, 3, 3).
    matbg : complex
        Background material value tensor (B, 3, 3).
    Returns
    -------
    matdist_xy : backend tensor
        Material distribution in real space, shape (B, 3, 3, Nx, Ny), complex dtype.
    '''
    Nx, Ny = bitmap.shape

    # Get bitmap mask
    mask_c = bitmap          # (Nx, Ny), bool
    mask_c = backend.reshape(mask_c, (1, 1, 1, Nx, Ny))  # (1, 1, 1, Nx, Ny)
    
    # Material values
    mat_tensor = matval  # (wvl, 3, 3)
    mat_bg_tensor = matbg  # (wvl, 3, 3)
    
    if mat_tensor.shape[0] != mat_bg_tensor.shape[0]:
        if mat_bg_tensor.shape[0] == 1:
            # replicate along wavelength dimension
            target_shape = (mat_tensor.shape[0],) + mat_bg_tensor.shape[1:]
            mat_bg_tensor = backend.expand(mat_bg_tensor, target_shape)
        else:
            raise ValueError("Material and background material must have the same number of wavelengths")
    
    # Broadcast epsilon and epsilon_bg to (wvl, 3, 3, 1, 1)
    init_shape = mat_tensor.shape      # (wvl, 3, 3)
    mat = backend.reshape(mat_tensor, init_shape + (1, 1))# (wvl, 3, 3, 1, 1)
    mat_bg = backend.reshape(mat_bg_tensor, init_shape + (1, 1))# (wvl, 3, 3, 1, 1)
    
    # Expand eps_bg to (wvl, 3, 3, Nx, Ny)
    matdist_xy = backend.expand(mat_bg,init_shape + (Nx, Ny))# (wvl, 3, 3, Nx, Ny)
    # Δmat per batch
    delta_mat_b = mat - mat_bg                          # (B, 3, 3, 1, 1)

    # mat = bg + Δmat * mask
    matdist_xy = matdist_xy + delta_mat_b * mask_c      # (B, 3, 3, Nx, Ny)

    return matdist_xy

def get_rotated_grid(backend: "Backend",
                     lattice: "Lattice",
                     center: Tuple[float, float],
                     angle: float):
    '''
    Get the rotated coordinate grid (xr, yr) for the object's local frame.
    
    Parameters
    ----------
    backend : Backend
        Computational backend.
    lattice : Lattice
        Lattice object defining the simulation domain.
    center : tuple of float
        (x,y) coordinates of the rectangle's center (in the range [-Lx/2, Lx/2] x [-Ly/2, Ly/2]).
    angle : float
        Rotation angle in radians.
        
    Returns
    -------
    xr, yr : backend tensors
        Rotated coordinate grids in the rectangle's local frame, shape (Nx, Ny).
    '''
    
    Nx, Ny = lattice.grid
    Lx, Ly = lattice.period
    
    # Coordinate grid 
    ix = backend.arange(0, Nx)
    iy = backend.arange(0, Ny)
    x = ix * (Lx / Nx)
    y = iy * (Ly / Ny)
    X, Y = backend.meshgrid(x, y, indexing='ij')
    
    cx, cy = center
    
    # Shift center to [0, Lx] x [0, Ly] coordinates
    cx = cx + Lx / 2.0
    cy = cy + Ly / 2.0
    
    # angle in radians (scalar → backend tensor)
    theta = backend.asarray(angle, complex=False)
    cos_t = backend.cos(theta)
    sin_t = backend.sin(theta)

    # coordinates relative to center
    dx = X - cx    # (Nx, Ny)
    dy = Y - cy    # (Nx, Ny)
    
    Lx_t = backend.asarray(Lx, complex=False)
    Ly_t = backend.asarray(Ly, complex=False)  
    
    dx_p = backend.mod(dx + Lx_t/2, Lx_t) - Lx_t/2
    dy_p = backend.mod(dy + Ly_t/2, Ly_t) - Ly_t/2
    
    # rotate grid into rectangle's local frame
    # (so rectangle is axis-aligned in (xr, yr))
    xr = dx_p * cos_t + dy_p * sin_t
    yr = -dx_p * sin_t + dy_p * cos_t
    
    return xr, yr

def wrap_center(backend: "Backend", cx: float, cy: float, Lx: float, Ly: float):
    """
    Map center (cx, cy) to the principal cell [-Lx/2, Lx/2] x [-Ly/2, Ly/2]
    using periodicity.
    """

    cx_t = backend.asarray(cx, complex=False)
    cy_t = backend.asarray(cy, complex=False)

    Lx_t = backend.asarray(Lx, complex=False)
    Ly_t = backend.asarray(Ly, complex=False)

    cx_wrapped = backend.mod(cx_t + Lx_t / 2.0, Lx_t) - Lx_t / 2.0
    cy_wrapped = backend.mod(cy_t + Ly_t / 2.0, Ly_t) - Ly_t / 2.0
    
    return cx_wrapped, cy_wrapped 

""" Bitmaps for shapes """
def bitmap_rect(backend: "Backend",
                lattice: "Lattice",
                center: Tuple[float, float],
                size: Tuple[float, float],
                angle: float,
                soft_mask: bool = False,
                smoothness: float = 0.05):
    '''
    Convert the rectangle to a bitmap representation on the specified grid.
    
    Parameters
    ----------
    backend : Backend
        Computational backend.
    lattice : Lattice
        Lattice object defining the simulation domain.
    center : tuple of float
        (x,y) coordinates of the object's center in length units. (0, 0) is the center.
    size : tuple of float
        (width, height) of the rectangle. Length units.
    angle : float
        Rotation angle in radians.
    soft_mask : bool
        Whether the object should use a soft mask for differentiable operations. Default is False.
        If True the bitmap representation will use smooth sigmoid approximation.
        *Important*: Fourier coefficients will still be computed analytically for sharp boundaries,
        so soft_mask only affects real-space distributions.
    smoothness : float
        Smoothness parameter for sigmoid. Default is 0.05.
    
    Returns
    -------
    bitmap : backend tensor
        Bitmap representation of the rectangle, shape (Nx, Ny).
    '''
    
    w, h = size
    half_w, half_h = w / 2.0, h / 2.0
    
    #Wrap center to the unit cell
    center = wrap_center(backend, 
                         center[0], center[1], 
                         lattice.grid[0], lattice.grid[1])
    
    xr, yr = get_rotated_grid(backend,
                              lattice,
                              center,
                              angle)

    # mask in local coordinates
    ax = backend.abs(xr)
    ay = backend.abs(yr)
    
    if soft_mask:
        # Smooth mask
        eps = smoothness * min(w, h)  # tuning parameter

        dist_x = ax - half_w
        dist_y = ay - half_h

        ex = -dist_x / eps
        ey = -dist_y / eps

        sx = backend.sigmoid(ex)
        sy = backend.sigmoid(ey)

        mask = sx * sy        # (Nx, Ny), float, differentiable
    else:
        # Sharp mask
        mask = (ax <= half_w) & (ay <= half_h)            # (Nx, Ny), bool

    return backend.asarray(mask, complex=False)

def bitmap_ellipse(backend: Backend,
                   lattice: Lattice,
                   center: Tuple[float, float],
                   size: Tuple[float, float],
                   angle: float,
                   soft_mask: bool = False,
                   smoothness: float = 0.05):
    '''
    Convert the ellipse to a bitmap representation on the specified grid.
    
    Parameters
    ----------
    backend : Backend
        Computational backend.
    lattice : Lattice
        Lattice object defining the simulation domain.
    center : tuple of float
        (x,y) coordinates of the object's center in length units. (0, 0) is the center.
    size : tuple of float
        (width, height) of the ellipse. Length units.
    angle : float
        Rotation angle in radians.
    soft_mask : bool
        Whether the object should use a soft mask for differentiable operations. Default is False.
        If True the bitmap representation will use smooth sigmoid approximation.
        *Important*: Fourier coefficients will still be computed analytically for sharp boundaries,
        so soft_mask only affects real-space distributions.
    smoothness : float
        Smoothness parameter for sigmoid. Default is 0.05.
    
    Returns
    -------
    bitmap : backend tensor
        Bitmap representation of the ellipse, shape (Nx, Ny).
    '''
    
    w, h = size
    half_w, half_h = w / 2.0, h / 2.0
    
    #Wrap center to the unit cell
    center = wrap_center(backend, 
                         center[0], center[1], 
                         lattice.grid[0], lattice.grid[1])
    
    xr, yr = get_rotated_grid(backend,
                              lattice,
                              center,
                              angle)
    
    # normalized squared radius
    r2 = (xr / half_w)**2 + (yr / half_h)**2
    
    if soft_mask:
        eps = smoothness
        d = (backend.sqrt(r2) - 1.0) / (eps + 1e-8)  # avoid div by zero
        mask = backend.sigmoid(-d)    # float mask
    else:
        # Sharp mask
        mask = r2 <= 1.0         # (Nx, Ny), bool

    return backend.asarray(mask, complex=False)