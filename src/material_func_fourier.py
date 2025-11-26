from typing import Tuple
import torch

from src.backend import Backend


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
        Shape: (Nx, Ny) or (B, Nx, Ny).

    M, N : int
        Number of harmonics along x and y.

    Returns
    -------
    matfunc_mn : backend tensor
        Fourier coefficients matfunc_{m,n}, shape (B, 2M+1, 2N+1),
        complex-valued.
    """
    matfunc_xy = backend.asarray(matfunc_xy, complex=True)

    shape = backend.shape(matfunc_xy)
    if len(shape) == 2:
        Nx, Ny = shape
        matfunc_xy = backend.reshape(matfunc_xy, (1, Nx, Ny))
    elif len(shape) == 3:
        B, Nx, Ny = shape
    else:
        raise ValueError("matfunc_xy must have shape (Nx,Ny) or (B,Nx,Ny)")
    
    # Sanity: need enough points to support requested harmonics
    if (2 * M + 1) > Nx or (2 * N + 1) > Ny:
        raise ValueError(
            f"Grid too small for requested harmonics: "
            f"(2M+1, 2N+1)=({2*M+1}, {2*N+1}) vs (Nx,Ny)=({Nx}, {Ny})"
        )

    # FFT over x,y
    matfunc_fft = backend.fft2(matfunc_xy)         # (B, Nx, Ny), complex
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