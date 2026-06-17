# metarcwa/solver/harmonics.py
# Description

import torch

def reciprocal_lattice_vectors(a1: torch.tensor, a2: torch.tensor):
    """
    Compute 2D reciprocal lattice vectors b1 and b2
    using a1 and a2.

    a1 and a2 being the two lattice vectors used
    to define the in-plane periodicity of the unit
    cell of the direct lattice. 

    Orthogonality relations:

    b1 * a1 = 2pi
    b1 * a2 = 0
    b2 * a1 = 0
    b2 * a1 = 2pi

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

    # Convert a1 and a2 to float before stacking
    a1 = torch.as_tensor(a1, dtype = torch.float32)
    a2 = torch.as_tensor(a2, dtype = torch.float32)

    # Define matrix a and r

    a = torch.stack([a1,a2],dim=0)
    r = 2 * torch.pi * torch.eye(2)

    # Solve the linear system ab = r
    
    b = torch.linalg.solve(a,r)

    b1 = b[:,0]
    b2 = b[:,1]

    return b1,b2

# Test

a1 = torch.tensor([15,0])
a2 = torch.tensor([0,15])

# Convert a1 and a2 to float before doing dot product
# so that a and b are of same type

a1 = torch.as_tensor(a1, dtype = torch.float32)
a2 = torch.as_tensor(a2, dtype = torch.float32)

b1, b2 = reciprocal_lattice_vectors(a1,a2)

print(b1)
print(b2)
print(torch.dot(b1, a1), f"should be 2pi")
print(torch.dot(b1,a2), f"should be 0")
print(torch.dot(b2,a1), f"should be 0")
print(torch.dot(b2,a2), f"should be 2pi") 


def harmonic_index_map(m_max:int, n_max: int):
    """
    Create 2D grid of harmonic indicies m and n.

    The harmonics are:

    m = -m_max,...0,...,+m_max
    n = -n_max,...,0,...,+n_max

    Each position (row, column) in the returned tensor corresponds
    to one harmonic order (m,n)

    Parameters:
    ----------------
    m_max: int
        Maximum harmonic order in the first reciprocal-lattice direction
    n_max: int
        Maximum harmonic order in the second reciprocal-lattice direction

    Returns:
    ------------------
    M: torch.Tensor
        Grid containing m indices, shape [2*m_max + 1, 2*n_max + 1]
    N: torch.Tensor
        Grid containing n indicies, shape [2*m_max + 1, 2*n_max +1]
    """

    m = torch.arange(-m_max, m_max + 1)
    n = torch.arange(-n_max, n_max + 1)

    M, N = torch.meshgrid(m,n)

    return M,N

# Test

M, N = harmonic_index_map(1,1)
print(M)
print(N)
print(M.shape)  # [3,3]

M,N = harmonic_index_map(2,2)
print(M)
print(N)
print(M.shape) #[5,5]

def reciprocal_index_map(M: torch.Tensor, N: torch.Tensor, b1: torch.Tensor, b2: torch.Tensor):
    """
    Computs reciprocal lattice vectors G_mn for every harmonic order.

    Each harmonic order (m,n) corresponds to a reciprocal lattice vector:

     G_mn = m*b1 + n*b2

     where b1 and b2 are the reciprocal lattice basis vectors.

     Since each G_mn is a 2D vector, the intermediate tensor G has shape:

    [N_m, N_n, 2]

    where:

    N_m = 2*m_max + 1
    N_n = 2*n_max + 1
    and the last dimension refers to [x,y]

    Parameters:
    --------------------
    M: torch.Tensor
        Grid of m indicies, shape [N_m, N_n]
    N: torch. Tensor
        Grid of n indicies, shape [N_m, N_n]
    b1: torch.Tensor
        First reciprocal lattice vector, shape [2]
    b2: torch.Tensor
        Second reciprocal lattice vector, shape [2]

    Returns:
    ------------------------
    G_x: torch.Tensor
        x-component of G_mn for every harmonic, shape [N_m, N_n]
    G_y: torch.Tensor
        y-component of G_m for every harmonic, shape [N_m, N_n]
    """
    # Build the reciprocal lattice vector from its
    # 2 basis vectors b1 and b2

    G = M[...,None] * b1 + N[...,None] * b2

    # (rows (m),columns(n),vector components(x,y) )

    G_x = G[...,0]
    G_y = G[...,1]

def harmonic_wavevectors(kx0: torch.Tensor, ky0: torch.Tensor, Gx: torch.Tensor,
                         Gy: torch.Tensor)
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
    ky0: torch.Tensor
        y component of the incident in-plane wavevector
    Gx: torch.Tensor
        x component of reciprocal lattice shifts, shape [N_m, N_n]
    Gy: torch.Tensor
        y component of reciprocal lattice shifts, shape [N_m, N_n]

    Returns:
    ---------------
    kx: torch.Tensor
        x-component of each harmonic wavevector, shape [N_m, N_n]
    ky: torch.Tensor
        y-component of each harmonic wavevector, shape [N_m, N_n]

    """

    kx = kx0 + Gx
    ky = ky0 + Gy

    return kx, ky

