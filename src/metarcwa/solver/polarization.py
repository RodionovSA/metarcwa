# metarcwa/solver/polarization.py
"""
polarization — s/p basis of a plane wave
=========================================

Tangential E of the unit s- and p-polarized plane waves with in-plane
wavevector ``(kx, ky)`` and normal component ``kz`` in a medium of index
``n`` (all k0-normalized)::

    s = (-ky, kx) / rho,     p_t = -(kz / n) (kx, ky) / rho,     rho = sqrt(kx² + ky²)

``s`` has no ``kz`` dependence; the tangential part of ``p`` does, so a wave
and its reflection (``-kz``) have different ``p_t``. At ``rho -> 0`` (normal
incidence) the direction is taken from the global azimuth ``phi0``.

Shared by :mod:`metarcwa.solver.background` (the incident s/p waves) and
:mod:`metarcwa.observables.base` (Jones matrices), so both use one basis.
"""

import torch

#: ``rho`` below which the in-plane direction falls back to ``phi0``.
RHO_EPS = 1e-8


def sp_vectors(kx: torch.Tensor, ky: torch.Tensor, kz: torch.Tensor,
               n: torch.Tensor, phi0: torch.Tensor):
    """Components of the s and p tangential-E unit vectors.

    Parameters
    ----------
    kx, ky : torch.Tensor
        In-plane wavevector (real), any broadcastable shapes.
    kz : torch.Tensor
        Normal wavevector component (complex); its sign selects the
        propagation direction.
    n : torch.Tensor
        Refractive index of the medium, broadcastable.
    phi0 : torch.Tensor
        Fallback azimuth for ``rho -> 0``, broadcastable.

    Returns
    -------
    s_x, s_y, p_x, p_y : torch.Tensor
        Complex components (dtype of ``kz``), broadcast shape of the inputs.
    """
    rho = torch.sqrt(kx ** 2 + ky ** 2)
    near_zero = rho < RHO_EPS
    rho_safe = torch.where(near_zero, torch.ones_like(rho), rho)
    ux = torch.where(near_zero, torch.cos(phi0), kx / rho_safe).to(kz.dtype)
    uy = torch.where(near_zero, torch.sin(phi0), ky / rho_safe).to(kz.dtype)
    kz_n = kz / n
    return -uy, ux, -kz_n * ux, -kz_n * uy


def sp_matrix(kx: torch.Tensor, ky: torch.Tensor, kz: torch.Tensor,
              n: torch.Tensor, phi0: torch.Tensor) -> torch.Tensor:
    """The s/p basis as a ``[..., 2, 2]`` matrix (rows ``Sx, Sy``; columns s, p).

    See :func:`sp_vectors` for the parameters.
    """
    sx, sy, px, py = sp_vectors(kx, ky, kz, n, phi0)
    return torch.stack([torch.stack([sx, px], dim=-1),
                        torch.stack([sy, py], dim=-1)], dim=-2)
