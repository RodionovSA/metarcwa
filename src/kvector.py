from typing import Tuple
import torch

from src.backend import Backend

def compute_k0xy(
    backend: Backend,          
    wavelengths,      
    theta,            
    phi,
    n_inc,              
    reduced: bool = False 
    ) -> Tuple[torch.Tensor, torch.Tensor]:
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
        In radians. Can be scalar or broadcastable to `wavelengths`.

    phi : array-like or scalar
        Azimuthal angle in the xy-plane (0 along +x).
        In radians. Can be scalar or broadcastable to `wavelengths`.
        
    n_inc : array-like or scalar
        Refractive index of the incident medium.
        Can be scalar or broadcastable to `wavelengths`.
        Imag part is ignored.
        
    reduced : bool, optional
        If True, compute reduced wavevectors (divided by k0). Default is False.

    Returns
    -------
    k0x, k0y : torch.Tensor
        Tensors of the same broadcasted shape as `wavelengths`,
        real-valued, on backend.device, with backend.dtype.
    """
    if isinstance(backend, Backend) is False:
        raise TypeError("backend must be an instance of Backend.")
    
    # Convert all inputs to backend tensors
    lam   = backend.asarray(wavelengths, complex=False)
    th    = backend.asarray(theta,       complex=False)
    ph    = backend.asarray(phi,         complex=False)
    n_inc = backend.asarray(n_inc,       complex=False)

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
    kx0: torch.Tensor,
    ky0: torch.Tensor,
    Lx,
    Ly,
    M: int,
    N: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
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
        Harmonic wavevector components with shape (n_lambda, 2M+1, 2N+1),
        real-valued, on backend.device, with backend.dtype.
    """
    # Ensure base k components are backend-compatible and 1D
    kx0 = backend.asarray(kx0, complex=False)  # (B,)
    ky0 = backend.asarray(ky0, complex=False)  # (B,)

    if backend.shape(kx0) != backend.shape(ky0):
        raise ValueError(f"kx0 and ky0 must have the same shape, got {backend.shape(kx0)} and {backend.shape(ky0)}")

    B = backend.shape(kx0)[0]  # batch over wavelengths

    # Periods → reciprocal lattice vectors
    Lx_t = backend.asarray(Lx, complex=False)
    Ly_t = backend.asarray(Ly, complex=False)

    Gx = (2.0 * backend.pi / Lx_t)
    Gy = (2.0 * backend.pi / Ly_t)

    # Harmonic indices m ∈ [-M..M], n ∈ [-N..N]
    m = backend.arange(-M, M+1)
    n = backend.arange(-N, N+1)

    # Reshape for broadcasting:
    kx0 = backend.reshape(kx0, (B, 1, 1))
    ky0 = backend.reshape(ky0, (B, 1, 1))

    m = backend.reshape(m, (1, 2*M+1, 1))
    n = backend.reshape(n, (1, 1, 2*N+1))

    Gx = backend.reshape(Gx, (1, 1, 1))
    Gy = backend.reshape(Gy, (1, 1, 1))

    # Compute lines
    Kx_line = kx0 + m * Gx       # (B, 2M+1, 1)
    Ky_line = ky0 + n * Gy       # (B, 1, 2N+1)

    # Expand using backend
    Kx = backend.expand(Kx_line, (B, 2*M+1, 2*N+1))
    Ky = backend.expand(Ky_line, (B, 2*M+1, 2*N+1))

    return Kx, Ky