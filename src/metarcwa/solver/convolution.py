# metarcwa/solver/convolution.py
# Description

import torch

def k_matrices(kx: torch.Tensor, ky: torch.Tensor):
    """ Computes diag Kx and Ky matrices of shape [..., Nh, Nh]"""
    pass


def convolution_matrix(eps_grid: torch.Tensor, m_flat: torch.Tensor, n_flat: torch.Tensor):
    """ Toeplitz convolution matrix in Fourier domain.

    eps_grid : [..., Ny, Nx]   (rows = y, cols = x)
    out      : [..., Nh, Nh]

    Axis–harmonic pairing for the FFT:
      axis -2  (length Ny)  ↔  n / b2 / a2  (y direction)
      axis -1  (length Nx)  ↔  m / b1 / a1  (x direction)
    """
    pass