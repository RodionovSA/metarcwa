# metarcwa/solver/harmonics.py
# Description

import torch

def reciprocal_lattice_vectors(a1: torch.Tensor, a2: torch.Tensor):
    """
    Compute 2D reciprocal lattice vectors b1 and b2
    using a1 and a2.

    a1 and a2 being the two lattice vectors used
    to define the in-plane periodicity of the unit
    cell of the direct lattice. 

    Orthogonality relations:

    b1 . a1 = 2pi
    b1 . a2 = 0
    b2 . a1 = 0
    b2 . a1 = 2pi

    Parameters:
    -------------------------
    a1 : torch.Tensor
        First direct lattice vector, shape [2]
    a2 : torch. Tensor
        Second direct lattice vector, shape [2]

    Outputs:
    ------------------------
    b1: torch.Tensor
        First reciprocal lattice vector, shape [2]
    b2: torch.Tensor
        Second reciprocal lattice vector, shape [2]

    """

    # Solve for b using matrices
    
    # Decompose lattice vectors into x and y componenets to calculate determinant
    a1x, a1y = a1[0], a1[1]
    a2x,a2y = a2[0], a2[1]
    
    det = a1x * a2y - a1y * a2x

    factor = 2 * torch.pi / det

    b1 = factor * torch.stack([a2y,-a2x])
    b2 = factor * torch.stack([-a1y, a1x])

    return b1, b2


def harmonic_index_map(m_max:int, n_max: int, device=None):
    """
    Create flattened harmonic index arrays meaning each harmonic
    is one entry in a Nh length list.

    Returns all harmonic orders:

    m = -m_max,...0,...,+m_max
    n = -n_max,...,0,...,+n_max

    Parameters:
    ----------------
    m_max: int
        Maximum harmonic order in the first reciprocal-lattice direction
    n_max: int
        Maximum harmonic order in the second reciprocal-lattice direction

    Returns:
    ------------------
    M_flat: torch.Tensor
        Flattened m indicies, shape [Nh]
    N_flat: torch.Tensor
        Flattened n indicies, shape [Nh]
    """

    m = torch.arange(-m_max, m_max + 1, device = device)
    n = torch.arange(-n_max, n_max + 1, device = device)

    # rows m, column n
    M, N = torch.meshgrid(m,n, indexing="ij")

    M_flat = torch.reshape(M, (-1,))
    N_flat = torch.reshape(N, (-1,))

    return M_flat,N_flat

def reciprocal_index_map(M_flat: torch.Tensor, N_flat: torch.Tensor, 
                         b1: torch.Tensor, b2: torch.Tensor):
    """
    Computs reciprocal lattice vectors G_mn for every harmonic order.

    Each harmonic (m,n) corresponds to:
    G_mn = m*b1 + n*b2

    Parameters:
    --------------------
    M_flat: torch.Tensor
        Flattened m indicies, shape [Nh]
    N_flat: torch.Tensor
        Flattened n indicies, shape [Nh]
    b1: torch.Tensor
        First reciprocal lattice vector, shape [2]
    b2: torch.Tensor
        Second reciprocal lattice vector, shape [2]

    Returns:
    ------------------------
    G: torch.Tensor
        Full reciprocal vectors, shape [Nh,2]
    G_x: torch.Tensor
        x-component of G_mn, shape[Nh]
    G_y: torch.Tensor
        y-component of G_mn, shape [Nh]
    """

    # G = m*b1 + n*b2
    # M and N, shape [Nh] -> shape [Nh,1]
    # b1 and b2, shape [2]
    # G, shape [Nh,2]
    G = M_flat[:,None] * b1 + N_flat[:,None] * b2

    Gx = G[:,0]
    Gy = G[:,1]

    return Gx, Gy, G

def harmonic_wavevectors(kx0: torch.Tensor, ky0: torch.Tensor, Gx: torch.Tensor,
                         Gy: torch.Tensor):
    """
    Compute the in-plane wavevector components for every harmonic.

    The incident wave has in-plane wavevector components kx0 and ky0. A
    periodic structure can add reciprocal lattice momentum G_mn, so each harmonic
    has:

    kx_mn = kx0 + Gx_mn
    ky_mn = ky0 + Gy_mn
    
    Parameters:
    ------------
    kx0: torch.Tensor
        x component of the incident in-plane wavevector
        Shape [N_wl, N_theta, N_phi]
    ky0: torch.Tensor
        y component of the incident in-plane wavevector
        Shape [N_wl, N_theta, N_phi]
    Gx: torch.Tensor
        x component of reciprocal lattice shifts
        Shape [Nh]
    Gy: torch.Tensor
        y component of reciprocal lattice shifts
        Shape [Nh]

    Returns:
    ---------------
    kx: torch.Tensor
        x-component of each harmonic wavevector
        Shape [N_wl, N_theta, N_phi, Nh]
    ky: torch.Tensor
        y-component of each harmonic wavevector
        Shape [N_wl, N_theta, N_phi, Nh]

    """

    # Add an extra dimension to kx0 and ky0
    # Shape [N_wl, N_theta, N_phi] -> [N_wl, N_theta, N_phi, 1]
    kx = kx0[...,None] + Gx         # Shape [N_wl, N_theta, N_phi, Nh]
    ky = ky0[...,None] + Gy         

    return kx, ky

