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

    Parameters
    ----------
    wavelength : float | Tensor | nn.Parameter
        Free-space wavelength. May be batched for a sweep.
    s_amp : complex | Tensor | nn.Parameter
        Complex amplitude of the s-polarized (TE) component.
    p_amp : complex | Tensor | nn.Parameter
        Complex amplitude of the p-polarized (TM) component.
    theta : float | Tensor | nn.Parameter
        Polar angle of incidence in rad, from the normal. Default 0.
    phi : float | Tensor | nn.Parameter
        Azimuthal angle in rad. Default 0.
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
        k0 = 2.0 * torch.pi / self.wavelength
        n = torch.sqrt(eps_incidence)
        kt = n * k0 * torch.sin(self.theta)
        kx0 = kt * torch.cos(self.phi)
        ky0 = kt * torch.sin(self.phi)
        return PlaneWaveSpec(
            wavelength=self.wavelength,
            kx0=kx0,
            ky0=ky0,
            s=self.s,
            p=self.p,
        )