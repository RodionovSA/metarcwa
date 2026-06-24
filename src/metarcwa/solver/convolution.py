# metarcwa/solver/convolution.py

import torch

def convolution_matrix(eps_grid: torch.Tensor, m_flat: torch.Tensor, n_flat: torch.Tensor):
    """
    Build the Toeplitz (Laurent) convolution matrix in the Fourier domain.

    For a spatially periodic permittivity eps(r), multiplication in real space
    becomes convolution in Fourier space. The (i, j) entry of the output is the
    Fourier coefficient eps_hat[m_i - m_j, n_i - n_j].

    ``norm='forward'`` divides the FFT by Nx*Ny so the coefficients equal the
    true Fourier series coefficients of eps.

    Axis–harmonic pairing:
        axis -2  (length Ny)  <->  n / b2 / a2  (y direction)
        axis -1  (length Nx)  <->  m / b1 / a1  (x direction)

    Parameters
    ----------
    eps_grid : torch.Tensor
        Permittivity sampled on the real-space grid, shape [..., Ny, Nx].
    m_flat : torch.Tensor
        Integer harmonic indices along x (b1 direction), shape [Nh].
    n_flat : torch.Tensor
        Integer harmonic indices along y (b2 direction), shape [Nh].

    Returns
    -------
    torch.Tensor
        Convolution matrix, shape [..., Nh, Nh].
    """
    Ny, Nx = eps_grid.shape[-2:]
    eps_hat = torch.fft.fft2(eps_grid, norm='forward')
    dm = (m_flat[:, None] - m_flat[None, :]) % Nx
    dn = (n_flat[:, None] - n_flat[None, :]) % Ny
    return eps_hat[..., dn, dm]
