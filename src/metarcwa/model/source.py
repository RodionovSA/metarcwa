# metarcwa/model/source.py
# DESCRIPTION

import torch
import torch.nn as nn

from .utils import register
from .spec import SourceSpec, PlaneWaveSpec


class Source(nn.Module):
    """Abstract base class for all illumination sources.

    A Source describes the physical excitation of the stack. Subclasses
    implement spec() to produce a SourceSpec — the solver-facing,
    pre-expansion description of the illumination. Concrete sources include
    PlaneWave; beam or multi-mode sources may be added as further
    subclasses without changing this base class or downstream code.

    The conversion from a physical description to per-harmonic amplitudes
    is the Solver's job; SourceSpec stays physical so that nothing
    solver-side (e.g. the harmonic count) leaks into the Model layer.
    """

    def spec(self, eps_incidence) -> SourceSpec:
        """Build the solver-facing source description.

        Parameters
        ----------
        eps_incidence
            Epsilon of the incidence medium, needed to convert
            angles to an in-plane wavevector.
        """
        raise NotImplementedError
    
class PlaneWave(Source):
    """A monochromatic plane wave illuminating the stack.

    Polarization is given as complex s- and p-amplitudes, each stored as
    two real fields so any component may independently be an nn.Parameter.

    ``wavelength``, ``theta``, and ``phi`` are **independent sweep axes**.
    Pass each as a 1-D tensor (or scalar) of any length; ``spec()`` forms the
    full outer-product grid with axis order ``[N_wl, N_theta, N_phi]``.
    Scalar (0-d) parameters collapse to a singleton axis and cost nothing.

    Parameters
    ----------
    wavelength : float | Tensor | nn.Parameter
        Free-space wavelength(s). Becomes axis 0 of the sweep grid.
    s_amp : complex | Tensor | nn.Parameter
        Complex amplitude of the s-polarized (TE) component.
    p_amp : complex | Tensor | nn.Parameter
        Complex amplitude of the p-polarized (TM) component.
    theta : float | Tensor | nn.Parameter
        Polar angle of incidence in rad, from the normal. Becomes axis 1.
        Default 0 (normal incidence).
    phi : float | Tensor | nn.Parameter
        Azimuthal angle in rad. Becomes axis 2. Default 0.
    """

    def __init__(self, wavelength, s_amp, p_amp, theta=0.0, phi=0.0):
        super().__init__()
        register(self, "wavelength", wavelength)
        register(self, "theta", theta)
        register(self, "phi", phi)
        self._register_complex("s", s_amp)
        self._register_complex("p", p_amp)

    def _register_complex(self, name, value):
        """Store a complex amplitude as two real fields: <name>_real, <name>_imag."""
        t = torch.as_tensor(value)
        real = t.real if t.is_complex() else t
        imag = t.imag if t.is_complex() else torch.zeros_like(t)
        register(self, f"{name}_real", real)
        register(self, f"{name}_imag", imag)

    @property
    def s(self) -> torch.Tensor:
        return torch.complex(self.s_real, self.s_imag)

    @property
    def p(self) -> torch.Tensor:
        return torch.complex(self.p_real, self.p_imag)

    def spec(self, eps_incidence) -> SourceSpec:
        # Place each swept parameter on its own broadcast axis:
        #   axis 0 — wavelength   [N_wl, 1,     1    ]
        #   axis 1 — theta        [1,    N_theta, 1    ]
        #   axis 2 — phi          [1,    1,       N_phi]
        # eps_incidence was evaluated at the stored wavelength, so it rides axis 0.
        wl      = self.wavelength.reshape(-1, 1, 1)
        th      = self.theta.reshape(1, -1, 1)
        ph      = self.phi.reshape(1, 1, -1)
        eps_inc = eps_incidence.reshape(-1, 1, 1)

        k0  = 2.0 * torch.pi / wl
        n   = torch.sqrt(eps_inc)
        kt  = n * k0 * torch.sin(th)          # [N_wl, N_theta, 1]
        kx0 = kt * torch.cos(ph)              # [N_wl, N_theta, N_phi]
        ky0 = kt * torch.sin(ph)

        return PlaneWaveSpec(
            wavelength=wl,
            kx0=kx0,
            ky0=ky0,
            s=self.s,
            p=self.p,
        )