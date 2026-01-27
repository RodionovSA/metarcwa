# src/model/geometry/harmonics.py
# Harmonics utilities for RCWA

from typing import Any, Tuple

from src.backend import Backend

def flatten_Kxy(backend: Backend, Kx: Any, Ky: Any) -> Tuple[Any, Any, int]:
    """
    Flattens the harmonic dimensions (2M+1, 2N+1) into a single 1D vector (N_rect).
    Result shape is (B, N_rect), where B = Nw,Nt,Np.
    """
    
    # 1. Save Batch Shape
    Nw, Nt, Np, Mm, Nn = Kx.shape
    N_rect = Mm * Nn
    
    # 2. Reshape to (B, N_rect)
    Kx_flat = backend.reshape(Kx, (Nw, Nt, Np, N_rect))
    Ky_flat = backend.reshape(Ky, (Nw, Nt, Np, N_rect))
    
    return Kx_flat, Ky_flat, N_rect

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