# metarcwa/solver/harmonics.py

import torch

def reciprocal_lattice_vectors(a1: torch.Tensor, a2: torch.Tensor):
    """
    Compute 2D reciprocal lattice vectors b1 and b2 from direct lattice
    vectors a1 and a2.

    Orthogonality relations:

    b1 . a1 = 2pi
    b1 . a2 = 0
    b2 . a1 = 0
    b2 . a2 = 2pi

    Parameters
    ----------
    a1 : torch.Tensor
        First direct lattice vector, shape [2]
    a2 : torch.Tensor
        Second direct lattice vector, shape [2]

    Returns
    -------
    b1 : torch.Tensor
        First reciprocal lattice vector, shape [2]
    b2 : torch.Tensor
        Second reciprocal lattice vector, shape [2]
    """
    a1x, a1y = a1[0], a1[1]
    a2x, a2y = a2[0], a2[1]

    det = a1x * a2y - a1y * a2x
    factor = 2 * torch.pi / det

    b1 = factor * torch.stack([a2y, -a2x])
    b2 = factor * torch.stack([-a1y, a1x])

    return b1, b2


def harmonic_index_map(m_max: int, n_max: int, circular: bool = False, device=None):
    """
    Create flattened harmonic index arrays, one entry per harmonic order.

    Covers all orders:
        m = -m_max, ..., 0, ..., +m_max
        n = -n_max, ..., 0, ..., +n_max

    Parameters
    ----------
    m_max : int
        Maximum harmonic order in the first reciprocal-lattice direction.
    n_max : int
        Maximum harmonic order in the second reciprocal-lattice direction.
    circular : bool
        If True, uses elliptical truncation (keeps only harmonics inside the
        ellipse (m/m_max)^2 + (n/n_max)^2 <= 1). Default is rectangular.
    device : torch.device or str, optional
        Target device for the output tensors.

    Returns
    -------
    m_flat : torch.Tensor
        Flattened m indices, shape [Nh]
    n_flat : torch.Tensor
        Flattened n indices, shape [Nh]
    """
    m = torch.arange(-m_max, m_max + 1, device=device)
    n = torch.arange(-n_max, n_max + 1, device=device)

    # Lay the harmonic grid out to match the eps grid [..., Ny, Nx]:
    #   axis 0 (rows) -> n  (a2 / y direction)
    #   axis 1 (cols) -> m  (a1 / x direction)
    N, M = torch.meshgrid(n, m, indexing="ij")

    if circular:
        mask = (M / m_max) ** 2 + (N / n_max) ** 2 <= 1.0
    else:
        mask = torch.ones_like(M, dtype=torch.bool)

    m_flat, n_flat = M[mask], N[mask]
    return m_flat, n_flat


def reciprocal_index_map(m_flat: torch.Tensor, n_flat: torch.Tensor,
                         b1: torch.Tensor, b2: torch.Tensor):
    """
    Compute reciprocal lattice vectors G_mn for every harmonic order.

    Each harmonic (m, n) corresponds to:
        G_mn = m*b1 + n*b2

    Axis convention: m indexes b1/a1 (x direction) and n indexes b2/a2
    (y direction). In the eps grid (shape [..., Ny, Nx]), the FFT over
    axis -1 (Nx) corresponds to m/Gx, and over axis -2 (Ny) to n/Gy.

    Parameters
    ----------
    m_flat : torch.Tensor
        Flattened m indices, shape [Nh]
    n_flat : torch.Tensor
        Flattened n indices, shape [Nh]
    b1 : torch.Tensor
        First reciprocal lattice vector, shape [2]
    b2 : torch.Tensor
        Second reciprocal lattice vector, shape [2]

    Returns
    -------
    Gx : torch.Tensor
        x-component of G_mn, shape [Nh]
    Gy : torch.Tensor
        y-component of G_mn, shape [Nh]
    """
    Gx = m_flat * b1[0] + n_flat * b2[0]
    Gy = m_flat * b1[1] + n_flat * b2[1]

    return Gx, Gy


def harmonic_wavevectors(kx0: torch.Tensor, ky0: torch.Tensor,
                         Gx: torch.Tensor, Gy: torch.Tensor):
    """
    Compute the in-plane wavevector components for every harmonic.

    A periodic structure can add reciprocal lattice momentum G_mn, so each
    harmonic receives a shift:
        kx_mn = kx0 + Gx_mn
        ky_mn = ky0 + Gy_mn

    Parameters
    ----------
    kx0 : torch.Tensor
        x component of the incident in-plane wavevector, shape [...].
    ky0 : torch.Tensor
        y component of the incident in-plane wavevector, shape [...].
    Gx : torch.Tensor
        x component of reciprocal lattice shifts, shape [Nh].
    Gy : torch.Tensor
        y component of reciprocal lattice shifts, shape [Nh].

    Returns
    -------
    kx : torch.Tensor
        x-component of each harmonic wavevector, shape [..., Nh].
    ky : torch.Tensor
        y-component of each harmonic wavevector, shape [..., Nh].
    """
    kx = kx0[..., None] + Gx
    ky = ky0[..., None] + Gy

    return kx, ky


def compute_kxy(kx0: torch.Tensor, ky0: torch.Tensor,
                a1: torch.Tensor, a2: torch.Tensor,
                m_flat: torch.Tensor, n_flat: torch.Tensor):
    """
    Compute all harmonic wavevector components in a single call.

    Chains reciprocal_lattice_vectors -> reciprocal_index_map ->
    harmonic_wavevectors.

    Parameters
    ----------
    kx0 : torch.Tensor
        x component of the incident in-plane wavevector, shape [...].
    ky0 : torch.Tensor
        y component of the incident in-plane wavevector, shape [...].
    a1 : torch.Tensor
        First direct lattice vector, shape [2].
    a2 : torch.Tensor
        Second direct lattice vector, shape [2].
    m_flat : torch.Tensor
        Flattened m harmonic indices, shape [Nh].
    n_flat : torch.Tensor
        Flattened n harmonic indices, shape [Nh].

    Returns
    -------
    kx : torch.Tensor
        x-component of each harmonic wavevector, shape [..., Nh].
    ky : torch.Tensor
        y-component of each harmonic wavevector, shape [..., Nh].
    """
    b1, b2 = reciprocal_lattice_vectors(a1, a2)
    Gx, Gy = reciprocal_index_map(m_flat, n_flat, b1, b2)
    kx, ky = harmonic_wavevectors(kx0, ky0, Gx, Gy)

    return kx, ky
