#src/compute.py
# Main computational routines for RCWA

from typing import Tuple, Any
from src.backend import Backend

""" Fourier transform"""
def custom_fft(backend: Backend,
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
    matfunc_fft = backend.fft2(field, dim=(-2, -1))         # (..., Nx, Ny), complex
    matfunc_fft_shifted = backend.fftshift(matfunc_fft, dim=(-2, -1))  # center zero frequency   
    cx = Nx // 2
    cy = Ny // 2
    m_lo = cx - M
    m_hi = cx + M + 1
    n_lo = cy - N
    n_hi = cy + N + 1

    matfunc_crop = matfunc_fft_shifted[..., m_lo:m_hi, n_lo:n_hi]  # (..., 2M+1, 2N+1)

    norm = Nx * Ny
    matfunc_mn = matfunc_crop / norm

    return matfunc_mn

def fft_matfunc(
    backend: Backend,
    matfunc_xy,
    M: int,
    N: int,
):
    """
    Compute FFT for matfunc_{m,n} from real-space matfunc(x,y).
    matfunc_xy can be epsilon, mu, or any other material function.

    Parameters
    ----------
    backend : Backend
        Computational backend.

    matfunc_xy : array-like or backend tensor
        Material function map in real space. Can be real or complex.
        Shape: (wvl, 3, 3, Nx, Ny)

    M, N : int
        Truncation orders along x and y. Total numbers are (2M+1) and (2N+1).

    Returns
    -------
    matfunc_mn : backend tensor
        Fourier coefficients matfunc_{m,n}, shape (wvl, 3, 3, 2M+1, 2N+1),
        complex-valued.
    """
    matfunc_xy = backend.asarray(matfunc_xy, complex=True)

    shape = matfunc_xy.shape
    if len(shape) != 5:
        raise ValueError("matfunc_xy must have shape (wvl, 3, 3, Nx, Ny)")
    if shape[1] != 3 or shape[2] != 3:
        raise ValueError("matfunc_xy must have shape (wvl, 3, 3, Nx, Ny)")
    
    matfunc_mn = custom_fft(backend, matfunc_xy, M, N)  # (wvl, 3, 3, 2M+1, 2N+1)

    return matfunc_mn

"""K-vectors"""
def compute_k0xy(
    backend: Backend,          
    wavelengths,      
    theta,            
    phi,
    n_inc,              
    reduced: bool = False 
    ) -> Tuple[Any, Any]:
    """
    Compute k0x, k0y for a batch of wavelengths and given incidence angles.

    Parameters
    ----------
    backend : Backend
        Computational backend (e.g., TorchBackend instance).

    wavelengths : array-like
        Wavelengths in the same units you want k0 in (e.g. micrometers).
        Shape can be (n_lambda,) or any broadcastable shape.

    theta : array-like or scalar
        Polar angle from +z (normal incidence = 0).
        In radians.
        
    phi : array-like or scalar
        Azimuthal angle in the xy-plane (0 along +x).
        In radians.
        
    n_inc : array-like or scalar
        Refractive index of the incident medium.
        Can be scalar or broadcastable to `wavelengths`.
        Imag part is ignored.
        
    reduced : bool, optional
        If True, compute reduced wavevectors (divided by k0). Default is False.

    Returns
    -------
    k0x, k0y : torch.Tensor
        Tensors of shape (Nw, Nt, Np) where Nw is number of wavelengths,
        real-valued, on backend.device, with backend.dtype.
    """
    if isinstance(backend, Backend) is False:
        raise TypeError("backend must be an instance of Backend.")
    
    # Convert all inputs to backend tensors
    lam   = backend.asarray(wavelengths, complex=False)
    th    = backend.asarray(theta,       complex=False)
    ph    = backend.asarray(phi,         complex=False)
    n_inc = backend.asarray(n_inc,       complex=False)
    
    lam_shape = lam.shape
    n_shape   = n_inc.shape
    th_shape  = th.shape
    ph_shape  = ph.shape
    # --- Minimal sanity: wavelengths, n_inc, th, phi must be scalar or 1D ---
    if len(lam_shape) > 1:
        raise ValueError(f"wavelengths must be scalar or 1D, got shape={lam_shape}")
    if len(n_shape) > 1:
        raise ValueError(f"n_inc must be scalar or 1D, got shape={n_shape}")
    if len(th_shape) > 1:
        raise ValueError(f"theta must be scalar or 1D, got shape={th_shape}")
    if len(ph_shape) > 1:
        raise ValueError(f"phi must be scalar or 1D, got shape={ph_shape}")

    # If both lam and n_inc are 1D, enforce same length
    if len(lam_shape) == 1 and len(n_shape) == 1 and lam_shape[0] != n_shape[0]:
        raise ValueError(
            f"wavelengths and n_inc must have same length when both are 1D, "
            f"got {lam_shape[0]} and {n_shape[0]}"
        )
    
    # Force them into [Nw,1,1], [1,Nt,1], [1,1,Np] so that
    # broadcasting gives [Nw,Nt,Np] for everything.
    lam = backend.reshape(lam, (-1, 1, 1))   # [Nw,1,1]
    th = backend.reshape(th, (1, -1, 1))     # [1,Nt,1]
    ph = backend.reshape(ph, (1, 1, -1))     # [1,1,Np]
    
    # Number of wavelengths after reshaping
    Nw = lam.shape[0]

    if len(n_shape) == 0:
        n_inc = backend.ones_like(lam) * n_inc
    elif len(n_shape) == 1:
        if n_shape[0] not in (1, Nw):
            raise ValueError(f"n_inc length must be 1 or {Nw}")
        n_inc = backend.reshape(n_inc, (-1, 1, 1))
        n_inc = backend.expand(n_inc, (Nw, 1, 1))

    # Direction cosines
    sin_th = backend.sin(th)
    cos_ph = backend.cos(ph)
    sin_ph = backend.sin(ph)

    # These are ALWAYS real-valued
    kx_dir = n_inc * sin_th * cos_ph
    ky_dir = n_inc * sin_th * sin_ph

    if reduced:
        # reduced wavevectors = direction cosines × refractive index
        return kx_dir, ky_dir

    # Free-space wavenumber
    k0 = (2.0 * backend.pi) / lam

    k0x = k0 * kx_dir
    k0y = k0 * ky_dir
        
    return k0x, k0y

def compute_Kxy(
    backend: Backend,
    kx0: Any,
    ky0: Any,
    Lx: Any,
    Ly: Any,
    M: int,
    N: int,
) -> Tuple[Any, Any]:
    """
    Construct Kx, Ky grids for RCWA harmonics:
        Kx = kx0 + 2* pi* m / Lx
        Ky = ky0 + 2* pi* n / Ly
    with m ∈ [-M..M], n ∈ [-N..N].

    Parameters
    ----------
    backend : Backend
        Computational backend (e.g., TorchBackend).

    kx0, ky0 : torch.Tensor
        Incident wavevector components along x and y in the superstrate,
        typically shape (n_lambda,). Must be real-valued.

    Lx, Ly : array-like or scalar
        Fundamental lattice periods along x and y. Usually scalars.
        Will be converted to backend real arrays and broadcast if needed.

    M, N : int
        Truncation orders along x and y. Total numbers are (2M+1) and (2N+1).

    Returns
    -------
    Kx, Ky : torch.Tensor
        Harmonic wavevector components with shape (Nw, Nt, Np, 2M+1, 2N+1),
        real-valued, on backend.device, with backend.dtype.
    """
    # Ensure base k components are backend-compatible and 1D
    kx0 = backend.asarray(kx0, complex=False)  # (B,)
    ky0 = backend.asarray(ky0, complex=False)  # (B,)

    if kx0.shape != ky0.shape:
        raise ValueError(f"kx0 and ky0 must have the same shape, got {ky0.shape} and {ky0.shape}")
    
    if len(kx0.shape) != 3:
        raise ValueError(
            f"kx0 and ky0 must have shape (Nw, Nt, Np); "
            f"got ndim={len(kx0.shape)}, shape={kx0.shape}"
        )

    Nw, Nt, Np = kx0.shape

    # Periods → reciprocal lattice vectors
    Lx_t = backend.asarray(Lx, complex=False)
    Ly_t = backend.asarray(Ly, complex=False)

    Gx = (2.0 * backend.pi / Lx_t)
    Gy = (2.0 * backend.pi / Ly_t)

    # Harmonic indices m ∈ [-M..M], n ∈ [-N..N]
    m = backend.arange(-M, M+1)
    n = backend.arange(-N, N+1)

    # Reshape for broadcasting:
    kx0 = backend.reshape(kx0, (Nw, Nt, Np, 1, 1))
    ky0 = backend.reshape(ky0, (Nw, Nt, Np, 1, 1))

    # m : (1, 1, 1, 2M+1, 1)
    m = backend.reshape(m, (1, 1, 1, 2 * M + 1, 1))
    # n : (1, 1, 1, 1, 2N+1)
    n = backend.reshape(n, (1, 1, 1, 1, 2 * N + 1))

    # Gx, Gy : (1, 1, 1, 1, 1) so they broadcast with everything
    Gx = backend.reshape(Gx, (1, 1, 1, 1, 1))
    Gy = backend.reshape(Gy, (1, 1, 1, 1, 1))

    # Compute "lines":
    Kx_line = kx0 + m * Gx       # (*batch_shape, 2M+1, 1)
    Ky_line = ky0 + n * Gy       # (*batch_shape, 1, 2N+1)

    # (*batch_shape, 2M+1, 2N+1)
    full_shape = (Nw, Nt, Np, 2 * M + 1, 2 * N + 1)
    Kx = backend.expand(Kx_line, full_shape)
    Ky = backend.expand(Ky_line, full_shape)

    return Kx, Ky

def flatten_Kxy(backend: Backend, Kx: Any, Ky: Any) -> Tuple[Any, Any, int]:
    """
    Flattens the harmonic dimensions (2M+1, 2N+1) into a single 1D vector (N_rect).
    Result shape is (B, N_rect), where B = Nw*Nt*Np.
    """
    
    # 1. Save Batch Shape
    Nw, Nt, Np, Mm, Nn = Kx.shape
    N_rect = Mm * Nn
    
    # 2. Reshape to (B, N_rect)
    Kx_flat = backend.reshape(Kx, (Nw, Nt, Np, N_rect))
    Ky_flat = backend.reshape(Ky, (Nw, Nt, Np, N_rect))
    
    return Kx_flat, Ky_flat, N_rect

def diagonal_K_matrices(backend: Backend, Kx: Any, Ky: Any, circular: bool = True) -> Tuple[Any, Any]:
    """
    Converts the vector (B, size) into the final size x size diagonal matrices.
    Result shape is (B, size, size).
    
    Parameters
    ----------
    backend : Backend
        Computational backend.
    Kx, Ky : Any
        Tensors of shape (B, 2M+1, 2N+1).
    circular : bool, optional
        Whether the input vectors were constructed with circular truncation.
        Default is True.
        
    Returns
    -------
    Kx_mat, Ky_mat : Any
        Tensors of shape (B, size, size) where size is (2M+1)*(2N+1).
    """
    # Ensure Kx, Ky are backend tensors and make complex dtype
    Kx = backend.asarray(Kx, complex=True)
    Ky = backend.asarray(Ky, complex=True)
    
    # Sanity
    if Kx.shape != Ky.shape:
        raise ValueError("Kx and Ky must have the same shape")
    
    _, _, _, Mm, Nn = Kx.shape
    M = (Mm - 1) // 2
    N = (Nn - 1) // 2
    
    if circular:
        mx, ny = build_harmonic_grid(backend, M, N)
        mask = elliptical_truncation_mask(mx, ny, M, N)
    else:
        mask = backend.astype(backend.ones(Mm*Nn), backend.bool)
    
    # Flatten Kx, Ky
    Kx_flat, Ky_flat, _ = flatten_Kxy(backend, Kx, Ky)
    
    # Truncate to circular harmonics if needed
    Kx_trunc = Kx_flat[..., mask]   # (B..., Ncirc)
    Ky_trunc = Ky_flat[..., mask]
    
    # Build diagonal matrices
    Kx_mat = backend.diag_embed(Kx_trunc)
    Ky_mat = backend.diag_embed(Ky_trunc)
    
    return Kx_mat, Ky_mat

""" Toeplitz matrix construction """
def build_index_map(backend: Backend, M: int , N: int, circular: bool = True) -> Tuple[Any, Any]:
    """
    Precompute index lookup for 2D Toeplitz convolution.
    
    Parameters
    ----------
    backend : Backend
        Computational backend.
    M, N : int
        Truncation orders along x and y.
    circular : bool
        Whether to apply circular (elliptical) truncation.
        Default is True.
    Returns
    -------
    dm_map, dn_map : Any
        Index maps for m and n differences.
        Shape: ((2M+1)(2N+1), (2M+1)(2N+1))
        Output values are in [0..4M] and [0..4N].
    """
    mx, ny = build_harmonic_grid(backend, M, N)
    
    if circular:
        mask = elliptical_truncation_mask(mx, ny, M_cut=M, N_cut=N)
        mx = mx[mask]
        ny = ny[mask]
        
    Nsize = mx.shape[0]
    
    p_row_mat = backend.reshape(mx, (Nsize, 1))
    p_col_mat = backend.reshape(mx, (1, Nsize))

    q_row_mat = backend.reshape(ny, (Nsize, 1))
    q_col_mat = backend.reshape(ny, (1, Nsize))
    
    # Calculate Differences: The result is N_circ x N_circ
    dm_map = p_row_mat - p_col_mat + 2*M  # Delta P map + shift to make indices [0..4M]
    dn_map = q_row_mat - q_col_mat + 2*N  # Delta Q map + shift to make indices [0..4N]
    
    return backend.astype(dm_map, backend.long), backend.astype(dn_map, backend.long)

def build_harmonic_grid(backend: Backend, M: int, N: int) -> Tuple[Any, Any]:
    """
    Build flatten 2D grid of harmonic indices (m,n) for m ∈ [-M..M], n ∈ [-N..N].
    
    Parameters
    ----------
    backend : Backend
        Computational backend.
    M, N : int
        Truncation orders along x and y.
        
    Returns:
        mx_flat, ny_flat : tensors of shape (K,)
        where K = (2M+1)(2N+1)
    """
    m = backend.astype(backend.arange(-M, M+1), backend.long)
    n = backend.astype(backend.arange(-N, N+1), backend.long)

    mi, nj = backend.meshgrid(m, n, indexing='ij')

    # Flatten to 1D for convenience
    size = (2*M+1)*(2*N+1)
    mx_flat = backend.reshape(mi, (size,))
    ny_flat = backend.reshape(nj, (size,))

    return mx_flat, ny_flat

def elliptical_truncation_mask(mx_flat: Any, ny_flat: Any, M_cut: int, N_cut: int) -> Any:
    """
    Build an elliptical (circular) harmonic mask.
    
    Parameters
    ----------
    mx, ny : flattened harmonic indices (from build_harmonic_grid)
    M_cut, N_cut : ellipse semi-major axes (in harmonic units)

    Returns
    -------
    mask : boolean tensor of shape (K,)
    """
    # Compute ellipse equation
    left = (mx_flat / M_cut)**2 + (ny_flat / N_cut)**2

    # Keep points inside ellipse
    mask = left <= 1.0

    return mask

def toeplitz_2d(backend: Backend, eps: Any, dm_map: Any, dn_map: Any) -> Any:
    """
    Compute 2D Toeplitz matrix from Fourier coefficients using precomputed index maps.
    
    Parameters:
    ----------
    backend : Backend
        Computational backend.
    eps:
        (B, 4M+1, 4N+1)
    dm_map: 
        (size, size). Expected to have values in [0..4M]
    dn_map: 
        (size, size). Expected to have values in [0..4N]
    
    Returns:
    ----------
        Tmat: (B, (2M+1)*(2N+1), (2M+1)*(2N+1))
    """
    eps = backend.asarray(eps, complex=True)
    
    if dm_map.shape != dn_map.shape:
        raise ValueError("dm_map and dn_map must have the same shape")
    if dm_map.shape[0] != dm_map.shape[1]:
        raise ValueError("dm_map must be square")   
    if dn_map.shape[0] != dn_map.shape[1]:
        raise ValueError("dn_map must be square")
    
    return eps[:, dm_map, dn_map]  # (B, size, size)



