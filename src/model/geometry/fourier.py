# src/model/geometry/fourier.py
# Fourier utilities for RCWA

from typing import Any, Tuple

from src.backend import Backend
from src.model.geometry.sampling import wrap_center

def custom_fft(backend: "Backend",
               field: Any,
               M: int,
               N: int) -> Any:
    """
    Custom function for FFT calculation
    
    Parameters
    ----------
    backend : Backend
        Computational backend.
    field : Any
        Field to be transformed. Shape: (..., Nx, Ny)
    M, N : int
        Truncation orders along x and y.
        
    Returns
    -------
    field_mn : Any
        Fourier coefficients of the field. Shape: (..., 2M+1, 2N+1)
    """
    field = backend.asarray(field, complex=True)

    shape = field.shape
    # Sanity: need enough points to support requested harmonics
    Nx, Ny = shape[-2], shape[-1]
    if (2 * M + 1) > Nx or (2 * N + 1) > Ny:
        raise ValueError(
            f"Grid too small for requested harmonics: "
            f"(2M+1, 2N+1)=({2*M+1}, {2*N+1}) vs (Nx,Ny)=({Nx}, {Ny})"
        )

    # FFT over x,y
    matmap_fft = backend.fft2(field, dim=(-2, -1))         # (..., Nx, Ny), complex
    matmap_fft_shifted = backend.fftshift(matmap_fft, dim=(-2, -1))  # center zero frequency   
    cx = Nx // 2
    cy = Ny // 2
    m_lo = cx - M
    m_hi = cx + M + 1
    n_lo = cy - N
    n_hi = cy + N + 1

    matmap_crop = matmap_fft_shifted[..., m_lo:m_hi, n_lo:n_hi]  # (..., 2M+1, 2N+1)

    norm = Nx * Ny
    matmap_mn = matmap_crop / norm

    return matmap_mn

def fft_matmap(
    backend: "Backend",
    matmap_xy: Any,
    M: int,
    N: int,
):
    """
    Compute FFT for matmap_{m,n} from real-space matmap(x,y).
    matmap_xy can be epsilon, mu, or any other material function.

    Parameters
    ----------
    backend : Backend
        Computational backend.
    matmap_xy : array-like or backend tensor
        Material function map in real space. Can be real or complex.
        Shape: (wvl, 3, 3, Nx, Ny)
    M, N : int
        Truncation orders along x and y. Total numbers are (2M+1) and (2N+1).

    Returns
    -------
    matmap_mn : backend tensor
        Fourier coefficients matmap_{m,n}, shape (wvl, 3, 3, 2M+1, 2N+1),
        complex-valued.
    """
    matmap_xy = backend.asarray(matmap_xy, complex=True)

    shape = matmap_xy.shape
    if len(shape) != 5:
        raise ValueError("matmap_xy must have shape (wvl, 3, 3, Nx, Ny)")
    if shape[1] != 3 or shape[2] != 3:
        raise ValueError("matmap_xy must have shape (wvl, 3, 3, Nx, Ny)")
    
    matmap_mn = custom_fft(backend, matmap_xy, M, N)  # (wvl, 3, 3, 2M+1, 2N+1)

    return matmap_mn

def sinc(backend: "Backend", z: Any):
    """
    Unnormalized sinc: sin(z) / z, with sinc(0) = 1.
    """
    z_abs = backend.abs(z)
    one = backend.ones_like(z)
    # Avoid division by zero
    z_safe = backend.where(z_abs < 1e-14, one, z)
    s = backend.sin(z) / z_safe
    # Enforce limit sinc(0) = 1
    s = backend.where(z_abs < 1e-14, one, s)
    return s

def get_fourier_rotated_grid(backend: "Backend",
                             center: Tuple[float, float],
                             period: Tuple[float, float],
                             angle: float,
                             M: int,
                             N: int):
    
    '''
    Get the rotated Fourier grid (ku, kv) for the object's local frame.
    
    Parameters
    ----------
    backend : Backend
        Computational backend.
    center: Tuple[float, float]
        Object's center
    period: Tuple[float, float]
        Lattice's period
    angle : float
        Rotation angle in radians.
    M, N : int
        Truncation orders along x and y.
    
    Returns
    -------
    ku, kv : backend tensors
        Rotated Fourier grids in the object's local frame, shape (2M+1, 2N+1).
    '''
    Lx, Ly = period
    cx, cy = center
    
    # Harmonic indices m, n
    m = backend.arange(-M, M + 1)   # (2M+1,)
    n = backend.arange(-N, N + 1)   # (2N+1,)
    # Reciprocal lattice vectors
    # G_m = 2π m / Lx, G_n = 2π n / Ly
    two_pi = 2.0 * backend.pi

    Gm = (two_pi / Lx) * m   # (2M+1,)
    Gn = (two_pi / Ly) * n   # (2N+1,)

    # 2D grids for Gm, Gn
    Gm_grid, Gn_grid = backend.meshgrid(Gm, Gn, indexing='ij')  # (2M+1, 2N+1)

    # --- Rotation: project (Gx, Gy) into local rectangle axes (u,v) ---
    # angle in radians
    theta = backend.asarray(angle, complex=False)
    cos_t = backend.cos(theta)
    sin_t = backend.sin(theta)

    # k_u, k_v in local frame
    ku = Gm_grid * cos_t + Gn_grid * sin_t          # (2M+1, 2N+1)
    kv = -Gm_grid * sin_t + Gn_grid * cos_t         # (2M+1, 2N+1)
    
    # Phase factor exp(-j(Gm x_c + Gn y_c))
    phase_arg = Gm_grid * cx + Gn_grid * cy
    phase = backend.exp(-1j * phase_arg) # (2M+1, 2N+1)
    
    return ku, kv, phase

""" Closed-form fourier for shapes """
def matmap_fourier_rect(backend: "Backend", 
                        center: Tuple[float, float],
                        size: Tuple[float, float],
                        angle: float,
                        period: Tuple[float, float],
                        M: int, 
                        N: int, 
                        matval: complex, 
                        matbg: complex):
    """
    Closed-form Fourier coefficients matmap_{m,n} for a single rectangle
    in a periodic cell. 
    Uses the standard formula:
        Δmat_{mn} = Δmat * (w*h / (Lx*Ly))
                     * sinc(G_m w/2) * sinc(G_n h/2)
                     * exp(-j(G_m x_c + G_n y_c))
    and adds the background term matbg * δ_{m0} δ_{n0}.
    
    Parameters
    ----------
    backend : Backend
        Computational backend.
    center : tuple of float
        (x,y) coordinates of the rectangle's center (in the range [-Lx/2, Lx/2] x [-Ly/2, Ly/2]).
    size : tuple of float
        (width, height) of the rectangle.
    angle : float
        Rotation angle in radians.
    period : tuple of float
        (Lx, Ly) period of the unit cell.
    M, N : int
        Number of harmonics along x and y.
    matval : complex
        Material value tensor inside the rectangle (B, 3, 3).
    matbg : complex
        Background material value tensor (B, 3, 3).
    
    Returns
    -------
    mat_mn : backend tensor
        Fourier coefficients mat_{m,n}, shape (B, 3, 3, 2M+1, 2N+1), complex.
        Indices correspond to m ∈ [-M..M], n ∈ [-N..N].
    """
    w, h = size
    Lx, Ly = period
    
    #Wrap center to the unit cell
    center = wrap_center(backend, 
                         center[0], center[1], 
                         Lx, Ly)
    
    cx, cy = center
    
    # Shift center to [0, Lx] x [0, Ly] coordinates
    cx = cx + Lx / 2.0
    cy = cy + Ly / 2.0
    
    # Adjust shapes
    if matval.shape[0] != matbg.shape[0]:
        if matbg.shape[0] == 1:
            # replicate along wavelength dimension
            target_shape = (matval.shape[0],) + matbg.shape[1:]
            matbg = backend.expand(matbg, target_shape)
        else:
            raise ValueError("Material and background material must have the same number of wavelengths")
        
    val_b = backend.reshape(matval, (matval.shape[0], 3, 3, 1, 1)) # (B, 3, 3, 1, 1)
    bg_b  = backend.reshape(matbg,  (matbg.shape[0], 3, 3, 1, 1)) # (B, 3, 3, 1, 1)

    # Material contrast
    delta_mat = val_b - bg_b   # (B, 3, 3, 1, 1)

    ku, kv, phase = get_fourier_rotated_grid(backend, (cx, cy), (Lx, Ly), angle, M, N)  # (2M+1, 2N+1)

    # Sinc factors in local coordinates
    zx = ku * (w / 2.0)
    zy = kv * (h / 2.0)

    Sx = sinc(backend, zx) # (2M+1, 2N+1)
    Sy = sinc(backend, zy) # (2M+1, 2N+1)
    
    #Broadcast to (B, 3, 3, 2M+1, 2N+1)
    Sx = backend.reshape(Sx, (1, 1, 1, 2 * M + 1, 2 * N + 1))
    Sy = backend.reshape(Sy, (1, 1, 1, 2 * M + 1, 2 * N + 1))
    phase = backend.reshape(phase, (1, 1, 1, 2 * M + 1, 2 * N + 1))

    # Contrast contribution
    area_factor = (w * h) / (Lx * Ly)
    
    delta_mat_mn = delta_mat * area_factor * Sx * Sy * phase  # (B, 3, 3, 2M+1, 2N+1)

    # Initialize with contrast term
    mat_mn = delta_mat_mn

    # --- Add background at (m=0,n=0), i.e. index (M,N) ---

    # bg_b: (B, 3, 3, 1, 1)
    # We add bg_b to mat_mn[:, M, N]
    # For Torch/NumPy this indexing is fine:
    if hasattr(mat_mn, "__setitem__"):
        # mat_{00} = matbg + Δmat * fill_fraction (already in mat_mn[:, M, N])
        mat_mn[..., M, N] = mat_mn[..., M, N] + bg_b[..., 0, 0]
    else:
        # if it is a different backend without in-place assignment,
        # should implement a functional update here.
        raise NotImplementedError("In-place assignment not supported for this backend.")

    return mat_mn

def matmap_fourier_ellipse(backend: "Backend", 
                           center: Tuple[float, float],
                           size: Tuple[float, float],
                           angle: float,
                           period: Tuple[float, float],
                           M: int, 
                           N: int, 
                           matval: complex, 
                           matbg: complex):
    """
    Closed-form Fourier coefficients matmap_{m,n} for a single ellipse
    in a periodic cell. 
    Uses the standard formula:
        Δmat_{mn} = Δmat * (π (w/2) (h/2) / (Lx*Ly))
                     * sinc(G_m w/2) * sinc(G_n h/2)
                     * exp(-j(G_m x_c + G_n y_c))
    and adds the background term matbg * δ_{m0} δ_{n0}.
    
    Parameters
    ----------
    backend : Backend
        Computational backend.
    center : tuple of float
        (x,y) coordinates of the ellipse's center (in the range [-Lx/2, Lx/2] x [-Ly/2, Ly/2]).
    size : tuple of float
        (width, height) of the ellipse.
    angle : float
        Rotation angle in radians.
    period : tuple of float
        (Lx, Ly) period of the unit cell.
    M, N : int
        Truncation order along x and y.
    matval : complex
        Material value tensor inside the ellipse (B, 3, 3).
    matbg : complex
        Background material value tensor (B, 3, 3).
    
    Returns
    -------
    mat_mn : backend tensor
        Fourier coefficients mat_{m,n}, shape (B, 3, 3, 2M+1, 2N+1), complex.
        Indices correspond to m ∈ [-M..M], n ∈ [-N..N].
    """
    
    w, h = size
    Lx, Ly = period
    
    #Wrap center to the unit cell
    center = wrap_center(backend, 
                         center[0], center[1], 
                         Lx, Ly)
    
    cx, cy = center
    
    # Shift center to [0, Lx] x [0, Ly] coordinates
    cx = cx + Lx / 2.0
    cy = cy + Ly / 2.0
    
    # Adjust shapes
    if matval.shape[0] != matbg.shape[0]:
        if matbg.shape[0] == 1:
            # replicate along wavelength dimension
            target_shape = (matval.shape[0],) + matbg.shape[1:]
            matbg = backend.expand(matbg, target_shape)
        else:
            raise ValueError("Material and background material must have the same number of wavelengths")
        
    val_b = backend.reshape(matval, (matval.shape[0], 3, 3, 1, 1)) # (B, 3, 3, 1, 1)
    bg_b  = backend.reshape(matbg,  (matbg.shape[0], 3, 3, 1, 1)) # (B, 3, 3, 1, 1)

    # Material contrast
    delta_mat = val_b - bg_b   # (B, 3, 3, 1, 1)

    # Get rotated Fourier grid and phase
    ku, kv, phase = get_fourier_rotated_grid(backend, (cx, cy), (Lx, Ly), angle, M, N)  # (2M+1, 2N+1)
    
    # Ellipse kernel: J1(rho) / rho
    a = w / 2.0
    b = h / 2.0
    rho2 = (a * ku)**2 + (b * kv)**2
    
    eps = backend.asarray(1e-9, complex=False)  
    rho = backend.sqrt(rho2 + eps)

    # J1(rho) / rho with safe limit at rho=0
    J1 = backend.besselj1(rho)

    ellipse_kernel = (2.0 * J1) / rho
    ellipse_kernel = backend.where(rho2 > eps, ellipse_kernel, backend.ones_like(rho2)) # lim_{rho→0} J1(rho)/rho = 1/2)

    #Broadcast to (B, 3, 3, 2M+1, 2N+1)
    ellipse_kernel = backend.reshape(ellipse_kernel, (1, 1, 1, 2 * M + 1, 2 * N + 1))
    phase = backend.reshape(phase, (1, 1, 1, 2 * M + 1, 2 * N + 1))

    # Contrast contribution
    area_factor = (backend.pi * a * b) / (Lx * Ly)
    
    delta_mat_mn = delta_mat * area_factor * ellipse_kernel * phase  # (B, 3, 3, 2M+1, 2N+1)

    # Initialize with contrast term
    mat_mn = delta_mat_mn

    # --- Add background at (m=0,n=0), i.e. index (M,N) ---

    # bg_b: (B, 3, 3, 1, 1)
    # We add bg_b to mat_mn[:, M, N]
    # For Torch/NumPy this indexing is fine:
    if hasattr(mat_mn, "__setitem__"):
        # mat_{00} = matbg + Δmat * fill_fraction (already in mat_mn[:, M, N])
        mat_mn[..., M, N] = mat_mn[..., M, N] + bg_b[..., 0, 0]
    else:
        # if it is a different backend without in-place assignment,
        # should implement a functional update here.
        raise NotImplementedError("In-place assignment not supported for this backend.")

    return mat_mn