# metarcwa/solver/harmonics.py

import torch
from typing import Tuple

def reciprocal_lattice_vectors(a1: torch.Tensor, a2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Reciprocal lattice vectors b1, b2 from direct vectors a1, a2.

    b1.a1 = b2.a2 = 2pi, b1.a2 = b2.a1 = 0.

    Parameters
    ----------
    a1, a2 : torch.Tensor
        Direct lattice vectors, shape [2].

    Returns
    -------
    b1, b2 : torch.Tensor
        Reciprocal lattice vectors, shape [2].
    """
    a1x, a1y = a1[0], a1[1]
    a2x, a2y = a2[0], a2[1]

    det = a1x * a2y - a1y * a2x
    factor = 2 * torch.pi / det

    b1 = factor * torch.stack([a2y, -a2x])
    b2 = factor * torch.stack([-a1y, a1x])

    return b1, b2


def harmonic_index_map(m_max: int, n_max: int, circular: bool = False, device="cpu") -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Flattened (m, n) harmonic index arrays covering m in [-m_max, m_max], n in [-n_max, n_max].

    Parameters
    ----------
    m_max, n_max : int
        Max harmonic order along a1/a2.
    circular : bool
        Elliptical truncation ``(m/m_max)^2 + (n/n_max)^2 <= 1`` if True, else rectangular.
    device : torch.device or str, optional
        Output device.

    Returns
    -------
    m_flat, n_flat : torch.Tensor
        Flattened harmonic indices, shape [Nh].
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
                         b1: torch.Tensor, b2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    G_mn = m*b1 + n*b2 for every harmonic, in physical units (1/length).

    m indexes b1/a1 (x), n indexes b2/a2 (y); matches the eps grid's
    axis -1 = m/Gx, axis -2 = n/Gy convention.

    Parameters
    ----------
    m_flat, n_flat : torch.Tensor
        Harmonic indices, shape [Nh].
    b1, b2 : torch.Tensor
        Reciprocal lattice vectors, shape [2].

    Returns
    -------
    Gx, Gy : torch.Tensor
        Reciprocal shift components, shape [Nh].
    """
    Gx = m_flat * b1[0] + n_flat * b2[0]
    Gy = m_flat * b1[1] + n_flat * b2[1]

    return Gx, Gy


def harmonic_wavevectors(kx0: torch.Tensor, ky0: torch.Tensor,
                         Gx: torch.Tensor, Gy: torch.Tensor,
                         k0: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    kx = kx0 + Gx/k0, ky = ky0 + Gy/k0.

    Gx/Gy are physical (1/length); kx0/ky0 are k0-normalized. k0 makes the
    reciprocal shift consistent before adding.

    Parameters
    ----------
    kx0, ky0 : torch.Tensor
        Incident in-plane wavevector, k0-normalized, shape [...].
    Gx, Gy : torch.Tensor
        Reciprocal lattice shifts, physical units, shape [Nh].
    k0 : torch.Tensor
        Free-space wavenumber 2*pi/wavelength, broadcastable against kx0's
        leading (wavelength) axis, e.g. [N_wvl].

    Returns
    -------
    kx, ky : torch.Tensor
        Per-harmonic wavevector, k0-normalized, shape [..., Nh].
    """
    k0 = k0.reshape(*k0.shape, *([1] * (kx0.ndim - k0.ndim + 1)))
    Gx = Gx / k0
    Gy = Gy / k0

    kx = kx0[..., None] + Gx
    ky = ky0[..., None] + Gy

    return kx, ky


def compute_kxy(kx0: torch.Tensor, ky0: torch.Tensor,
                a1: torch.Tensor, a2: torch.Tensor,
                m_flat: torch.Tensor, n_flat: torch.Tensor,
                k0: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Chains reciprocal_lattice_vectors -> reciprocal_index_map -> harmonic_wavevectors.

    Parameters
    ----------
    kx0, ky0 : torch.Tensor
        Incident in-plane wavevector, k0-normalized, shape [...].
    a1, a2 : torch.Tensor
        Direct lattice vectors, shape [2].
    m_flat, n_flat : torch.Tensor
        Harmonic indices, shape [Nh].
    k0 : torch.Tensor
        Free-space wavenumber 2*pi/wavelength.

    Returns
    -------
    kx, ky : torch.Tensor
        Per-harmonic wavevector, k0-normalized, shape [..., Nh].
    """
    b1, b2 = reciprocal_lattice_vectors(a1, a2)
    Gx, Gy = reciprocal_index_map(m_flat, n_flat, b1, b2)
    kx, ky = harmonic_wavevectors(kx0, ky0, Gx, Gy, k0=k0)

    return kx, ky
