from typing import Tuple, Any
from src.backend import Backend

""" Fourier transform"""
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
        Number of harmonics along x and y.

    Returns
    -------
    matfunc_mn : backend tensor
        Fourier coefficients matfunc_{m,n}, shape (wvl, 3, 3, 2M+1, 2N+1),
        complex-valued.
    """
    matfunc_xy = backend.asarray(matfunc_xy, complex=True)

    shape = backend.shape(matfunc_xy)
    if len(shape) != 5:
        raise ValueError("matfunc_xy must have shape (wvl, 3, 3, Nx, Ny)")
    if shape[1] != 3 or shape[2] != 3:
        raise ValueError("matfunc_xy must have shape (wvl, 3, 3, Nx, Ny)")
    
    # Sanity: need enough points to support requested harmonics
    Nx, Ny = shape[-2], shape[-1]
    if (2 * M + 1) > Nx or (2 * N + 1) > Ny:
        raise ValueError(
            f"Grid too small for requested harmonics: "
            f"(2M+1, 2N+1)=({2*M+1}, {2*N+1}) vs (Nx,Ny)=({Nx}, {Ny})"
        )

    # FFT over x,y
    matfunc_fft = backend.fft2(matfunc_xy)         # (wvl, 3, 3, Nx, Ny), complex
    matfunc_fft_shifted = backend.fftshift(matfunc_fft)  # center zero frequency   
    cx = Nx // 2
    cy = Ny // 2
    m_lo = cx - M
    m_hi = cx + M + 1
    n_lo = cy - N
    n_hi = cy + N + 1

    matfunc_crop = matfunc_fft_shifted[:, m_lo:m_hi, n_lo:n_hi]  # (B, 2M+1, 2N+1)

    norm = Nx * Ny
    matfunc_mn = matfunc_crop / norm

    return matfunc_mn

def circ_fft_matfunc(
    backend: Backend,
    matfunc_xy,
    M: int,
    N: int,
):
    """
    Compute Fourier coefficients matfunc_{m,n} from real-space matfunc(x,y)
    using a circular crop in (m,n)-index space. matfunc_xy can be epsilon, mu,
    or any other material function.
    

    Parameters
    ----------
    backend : Backend
        Computational backend.

    matfunc_xy : array-like or backend tensor
        Material function map in real space. Can be real or complex.
        Shape: (Nr, Ntheta) or (B, Nr, Ntheta).

    M, N : int
        Number of harmonics along r and theta.

    Returns
    -------
    matfunc_mn : backend tensor
        Fourier coefficients matfunc_{m,n}, shape (B, 2M+1, 2N+1),
        complex-valued.
    """
    cropped_matfunc_mn = fft_matfunc(backend, matfunc_xy, M, N)
    
    # --- Circular mask in (m,n) index space ---
    # m_idx ∈ [-M..M], n_idx ∈ [-N..N]
    m_idx = backend.arange(-M, M + 1)              # (2M+1,)
    n_idx = backend.arange(-N, N + 1)              # (2N+1,)

    m_idx = backend.reshape(m_idx, (1, 2 * M + 1, 1))
    n_idx = backend.reshape(n_idx, (1, 1, 2 * N + 1))

    # radius in index space
    R = min(M, N)
    R2 = R * R

    m2 = m_idx * m_idx
    n2 = n_idx * n_idx
    r2 = m2 + n2                         # (1, 2M+1, 2N+1)

    # mask: inside circle => 1, outside => 0
    mask = backend.asarray(r2 <= R2, complex=False)   # real mask

    matfunc_mn = cropped_matfunc_mn * mask # (B, 2M+1, 2N+1)

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
    
    lam_shape = backend.shape(lam)
    n_shape   = backend.shape(n_inc)
    # --- Minimal sanity: wavelengths and n_inc must be scalar or 1D ---
    if len(lam_shape) > 1:
        raise ValueError(f"wavelengths must be scalar or 1D, got shape={lam_shape}")
    if len(n_shape) > 1:
        raise ValueError(f"n_inc must be scalar or 1D, got shape={n_shape}")

    # If both lam and n_inc are 1D, enforce same length
    if len(lam_shape) == 1 and len(n_shape) == 1 and lam_shape[0] != n_shape[0]:
        raise ValueError(
            f"wavelengths and n_inc must have same length when both are 1D, "
            f"got {lam_shape[0]} and {n_shape[0]}"
        )
    
    # Force them into [Nw,1,1], [1,Nt,1], [1,1,Np] so that
    # broadcasting gives [Nw,Nt,Np] for everything.
    if len(backend.shape(lam)) in (0, 1):
        lam = backend.reshape(lam, (-1, 1, 1))   # [Nw,1,1]
    if len(backend.shape(th)) in (0, 1):
        th = backend.reshape(th, (1, -1, 1))     # [1,Nt,1]
    if len(backend.shape(ph)) in (0, 1):
        ph = backend.reshape(ph, (1, 1, -1))     # [1,1,Np]
    
    # Number of wavelengths after reshaping
    Nw = backend.shape(lam)[0]
    
    # ---- Handle n_inc so it always ends up [Nw,1,1] ----
    n_shape = backend.shape(n_inc)
    if len(n_shape) == 0:
        # Scalar index: use same value for all wavelengths
        # lam/lam is a [Nw,1,1] array of ones
        n_inc = n_inc * (lam / lam)             # [Nw,1,1]
    elif len(n_shape) == 1:
        if n_shape[0] == 1:
            # Single value → use for all wavelengths
            n_inc = backend.reshape(n_inc, (1, 1, 1))  # [1,1,1]
            n_inc = n_inc * (lam / lam)               # [Nw,1,1]
        elif n_shape[0] == Nw:
            # Per-wavelength index
            n_inc = backend.reshape(n_inc, (Nw, 1, 1))  # [Nw,1,1]
        else:
            raise ValueError(
                f"n_inc 1D length must be 1 or Nw={Nw}, got {n_shape[0]}"
            )
    else:
        # already protected by earlier check, but keep for clarity
        raise ValueError("n_inc must be scalar or 1D.")

    # Direction cosines
    sin_th = backend.sin(th)
    cos_ph = backend.cos(ph)
    sin_ph = backend.sin(ph)

    # These are ALWAYS real-valued physically.
    kx_dir = n_inc * sin_th * cos_ph
    ky_dir = n_inc * sin_th * sin_ph

    if reduced:
        # reduced wavevectors = direction cosines × refractive index
        return kx_dir, ky_dir

    # Backend-controlled pi & scalar arithmetic
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
        Number of harmonics along x and y. Total orders are (2M+1) and (2N+1).

    Returns
    -------
    Kx, Ky : torch.Tensor
        Harmonic wavevector components with shape (Nw, Nt, Np, 2M+1, 2N+1),
        real-valued, on backend.device, with backend.dtype.
    """
    # Ensure base k components are backend-compatible and 1D
    kx0 = backend.asarray(kx0, complex=False)  # (B,)
    ky0 = backend.asarray(ky0, complex=False)  # (B,)

    if backend.shape(kx0) != backend.shape(ky0):
        raise ValueError(f"kx0 and ky0 must have the same shape, got {backend.shape(kx0)} and {backend.shape(ky0)}")
    
    if len(backend.shape(kx0)) != 3:
        raise ValueError(
            f"kx0 and ky0 must have shape (Nw, Nt, Np); "
            f"got ndim={len(backend.shape(kx0))}, shape={backend.shape(kx0)}"
        )

    Nw, Nt, Np = backend.shape(kx0)

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
    Nw, Nt, Np, Mm, Nn = backend.shape(Kx)
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
    Kx = backend.asarray(Kx, complex=False)
    Ky = backend.asarray(Ky, complex=False)
    
    if backend.shape(Kx) != backend.shape(Ky):
        raise ValueError("Kx and Ky must have the same shape")
    
    _, _, _, Mm, Nn = backend.shape(Kx)
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
def build_index_map(backend, M , N, circular=True):
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
        mask = elliptical_truncation_mask(mx=mx, ny=ny, M_cut=M, N_cut=N)
        mx = mx[mask]
        ny = ny[mask]
        
    Nsize = mx.shape[0]
    
    p_row_mat = backend.reshape(mx, (Nsize, 1))
    p_col_mat = backend.reshape(mx, (1, Nsize))

    q_row_mat = backend.reshape(ny, (Nsize, 1))
    q_col_mat = backend.reshape(ny, (1, Nsize))
    
    # Calculate Differences: The result is N_circ x N_circ
    dm_map = p_row_mat - p_col_mat + 2*M  # Delta P map
    dn_map = q_row_mat - q_col_mat + 2*N  # Delta Q map
    
    return backend.astype(dm_map, backend.long), backend.astype(dn_map, backend.long)

def build_harmonic_grid(backend, M, N):
    """
    Returns:
        mx_flat, ny_flat : tensors of shape (K,)
        where K = (2M+1)(2N+1)
    """
    m = backend.astype(backend.arange(-M, M+1), backend.long)
    n = backend.astype(backend.arange(-N, N+1), backend.long)

    mi, nj = backend.meshgrid(m, n, indexing='ij')

    # Flatten to 1D for convenience
    size = (2*M+1)*(2*N+1)
    mx = backend.reshape(mi, (size,))
    ny = backend.reshape(nj, (size,))

    return mx, ny

def elliptical_truncation_mask(mx, ny, M_cut, N_cut):
    """
    Build an elliptical (circular) harmonic mask in S4 sense.
    
    Parameters
    ----------
    mx, ny : flattened harmonic indices (from build_harmonic_grid)
    M_cut, N_cut : ellipse semi-major axes (in harmonic units)

    Returns
    -------
    mask : boolean tensor of shape (K,)
    """
    # Compute ellipse equation
    left = (mx / M_cut)**2 + (ny / N_cut)**2

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



