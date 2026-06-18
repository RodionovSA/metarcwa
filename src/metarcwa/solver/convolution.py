# metarcwa/solver/convolution.py
# Description

import torch

def k_matrices(kx: torch.Tensor, ky: torch.Tensor):
    """ Computes diag Kx and Ky matrices of shape [..., Nh, Nh]"""
    pass


def convolution_matrix(eps_grid: torch.Tensor, m_flat: torch.Tensor, n_flat: torch.Tensor):
    """ Toeplitz convolution matrix in Fourier domain. 
    eps_grid: [N_layers, ..., Nx, Ny], out: [N_layers, ..., Nh, Nh]
    """
    pass