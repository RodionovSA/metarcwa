# metarcwa/model/source.py
# DESCRIPTION

import torch
import torch.nn as nn
from dataclasses import dataclass

from .nn_helpers import register

@dataclass(frozen=True)
class SourceSpec:
    """Solver-facing plane-wave illumination description.

    An immutable snapshot after angle-to-wavevector resolution. Polarization
    is deliberately not carried here: the solver only needs wavelength and
    in-plane wavevector to build the S-matrix — polarization amplitudes are
    applied afterward, downstream in ``results``, to the computed S-matrix.

    All batch axes follow the outer-product sweep convention set by
    ``Source.spec()``: ``[N_wl, N_theta, N_phi]``, with singleton axes for
    scalar parameters.

    Attributes
    ----------
    wavelength : Tensor | nn.Parameter
        Free-space wavelength, shape ``[N_wl, 1, 1]``. Retained so the
        free-space wavenumber ``k0 = 2*pi / wavelength`` can be recovered
        downstream (see ``kx0``/``ky0`` below).
    kx0, ky0 : Tensor | nn.Parameter
        In-plane wavevector components of the incident wave, **normalized by
        the free-space wavenumber** ``k0 = 2*pi / wavelength`` — i.e.
        dimensionless ``kx0 = n_inc * sin(theta) * cos(phi)``, *not* the
        physical ``k0 * n_inc * sin(theta) * cos(phi)``. The whole solver
        works in these k0-normalized units; multiply back by ``k0`` only
        where a physical (1/length) wavevector is required (e.g. final
        propagation phases). Already resolved using the incidence-medium
        index. Shape ``[N_wl, N_theta, N_phi]``.
    """

    wavelength: torch.Tensor
    kx0: torch.Tensor
    ky0: torch.Tensor


class Source(nn.Module):
    """A monochromatic plane wave illuminating the stack.

    ``wavelength``, ``theta``, and ``phi`` are **independent sweep axes**.
    Pass each as a 1-D tensor (or scalar) of any length; ``spec()`` forms the
    full outer-product grid with axis order ``[N_wl, N_theta, N_phi]``.
    Scalar (0-d) parameters collapse to a singleton axis and cost nothing.

    Polarization is not part of ``Source``: the solver's S-matrix does not
    depend on it, so s/p amplitudes are applied later, downstream in
    ``results``, to the computed S-matrix.

    Parameters
    ----------
    wavelength : float | Tensor | nn.Parameter
        Free-space wavelength(s). Becomes axis 0 of the sweep grid.
    theta : float | Tensor | nn.Parameter
        Polar angle of incidence in rad, from the normal. Becomes axis 1.
        Default 0 (normal incidence).
    phi : float | Tensor | nn.Parameter
        Azimuthal angle in rad. Becomes axis 2. Default 0.
    """

    def __init__(self, wavelength, theta=0.0, phi=0.0):
        super().__init__()
        register(self, "wavelength", wavelength)
        register(self, "theta", theta)
        register(self, "phi", phi)

    def spec(self, n_incidence: torch.Tensor) -> SourceSpec:
        """Build the solver-facing source description.

        Parameters
        ----------
        n_incidence : Tensor
            Real refractive index of the incidence medium, needed to convert
            angles to an in-plane wavevector.
        """
        # Place each swept parameter on its own broadcast axis:
        #   axis 0 — wavelength   [N_wl, 1,     1    ]
        #   axis 1 — theta        [1,    N_theta, 1    ]
        #   axis 2 — phi          [1,    1,       N_phi]
        # n_incidence was resolved at the stored wavelength, so it rides axis 0.
        #
        # kx0/ky0 are k0-NORMALIZED (dimensionless): the free-space wavenumber
        # k0 = 2*pi/wavelength is factored out, so kx0 = n*sin(th)*cos(ph), not
        # k0*n*sin(th)*cos(ph). The solver runs entirely in these units; k0 is
        # reintroduced downstream from `wavelength` only where a physical
        # wavevector is needed.
        wl = self.wavelength.reshape(-1, 1, 1)
        th = self.theta.reshape(1, -1, 1)
        ph = self.phi.reshape(1, 1, -1)
        n  = n_incidence.reshape(-1, 1, 1)

        kt  = n * torch.sin(th)          # [N_wl, N_theta, 1]  (k0-normalized)
        kx0 = kt * torch.cos(ph)         # [N_wl, N_theta, N_phi]
        ky0 = kt * torch.sin(ph)

        return SourceSpec(
            wavelength=self.wavelength,
            kx0=kx0,
            ky0=ky0,
        )
