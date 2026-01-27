#src/solver/python/utils.py
# Main computational routines for RCWA

from typing import Tuple, Any
from src.backend import Backend
from src.model.geometry.harmonics import build_harmonic_grid, elliptical_truncation_mask, flatten_Kxy

"""K-vectors"""
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

def kz_sign(backend: Backend, eigvals: Any, mode: str = "positive", tol: float = 1e-12):
    """
    Select kz branch from RCWA eigenvalues (lambda^2).

    Forward  modes: Im(kz) > 0  (or Re(kz) > 0 if Im=0)
    Backward modes: Im(kz) < 0  (or Re(kz) < 0 if Im=0)

    eigvals shape: (..., N)
    For eigvals the folowing relation is implied: lambda^2 = (jk_z/k_0)^2
    """

    # Safe sqrt
    lam2 = backend.asarray(eigvals, complex=True)
    lam = backend.sqrt(lam2)
    kz = -1j*lam

    imag = backend.imag(kz)
    real = backend.real(kz)
    
    is_evan = backend.abs(imag) > tol

    if mode == "positive":
        sign = backend.where(
            is_evan,
            backend.sign(imag),
            backend.sign(real),
        )
    elif mode == "negative":
        sign = backend.where(
            is_evan,
            -backend.sign(imag),
            -backend.sign(real),
        )
    else:
        raise ValueError("modes must be 'positive' or 'negative'")

    # Avoid zero sign
    sign = backend.where(sign == 0, 1.0, sign)

    return kz * sign

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

""" S matrices """
def build_block(backend: Backend, A: Any, B: Any, C: Any, D: Any) -> Any:
    """ 
    Build 2x2 block from A, B, C, and D matrices.
    block = (A B; C D)
    
    """
    top = backend.cat([A, B], dim=-1)   # [B, Nh, 2Nh]
    bot = backend.cat([C, D], dim=-1)   # [B, Nh, 2Nh]
    block   = backend.cat([top, bot], dim=-2)     # [B, 2Nh, 2Nh]
    
    return block
    
def split_block(S: Any) -> Tuple[Any, Any, Any, Any]:
    """
    Split a 2x2 block matrix:
        S = [ A  B ]
            [ C  D ]

    Returns A, B, C, D with the same batch dims.
    """

    if S.shape[-1] % 2 != 0 or S.shape[-2] % 2 != 0:
        raise ValueError(
            f"Last two dims must be even, got {S.shape[-2:]}"
        )

    Nh_row = S.shape[-2] // 2
    Nh_col = S.shape[-1] // 2

    A = S[..., :Nh_row, :Nh_col]
    B = S[..., :Nh_row, Nh_col:]
    C = S[..., Nh_row:, :Nh_col]
    D = S[..., Nh_row:, Nh_col:]

    return A, B, C, D