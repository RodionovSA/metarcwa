# metarcwa/solver/convolution.py
# Description

import torch

def k_matrices(kx: torch.Tensor, ky: torch.Tensor):
    """ Computes diag Kx and Ky matrices of shape [..., Nh, Nh]"""
    diag_kx = torch.diag_embed(kx)
    diag_ky = torch.diag_embed(ky)
    return diag_kx, diag_ky

def convolution_matrix(eps_grid: torch.Tensor, m_flat: torch.Tensor, n_flat: torch.Tensor):
    """ Toeplitz convolution matrix in Fourier domain.
  
    eps_grid : [..., Ny, Nx]   (rows = y, cols = x)
    out      : [..., Nh, Nh]

    Axis–harmonic pairing for the FFT:
      axis -2  (length Ny)  ↔  n / b2 / a2  (y direction)
      axis -1  (length Nx)  ↔  m / b1 / a1  (x direction)
    """

    # Fourier transform eps_grid
    eps_fourier = torch.fft.fft2(eps_grid, dim = (-2,-1))

    # Shift eps_hat so that the indicies 0,0 is at the centre
    # at position Ny/2 and Nx/2

    eps_shift = torch.fft.fftshift(eps_fourier, dim=(-2,-1))

    # Need to change shape from [...,...,Ny,Nx] to [...,...,Nh,Nh]
    # Where Nh = (2*m_max + 1)*(2*n_max + 1)
    # And each indicie is m_row - m_column, n_row - n_column

    dm = m_flat[:,None] - m_flat[None,:]
    dn = n_flat[:,None] - n_flat[None,:]

    # dm and dn have negative harmonic labels but 
    # arrays cannot be indexed with negatives
    # Add Nx // 2 and Ny // 2 to dm and dn 
    # in order to convert the harmonic labels
    # to array indices

    Nx = eps_grid.shape[-1]
    Ny = eps_grid.shape[-2]
    
    ix = dm + Nx // 2
    iy = dn + Ny // 2

    convolution_matrix = eps_shift[...,iy,ix]

    return convolution_matrix