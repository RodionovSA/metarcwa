# metarcwa/model/material.py
# DESCRIPTION

import torch
import torch.nn as nn

from .utils import register

class Material(nn.Module):
    """Abstract base class for all materials.

    Subclasses implement epsilon() to provide the complex relative
    permittivity at a given wavelength. Concrete materials include
    dispersion models.

    Scope
    -----
    The current contract assumes isotropic, non-magnetic materials:
    epsilon() returns a scalar-like permittivity and mu is taken to be 1
    everywhere. This is a deliberate restriction, not a fundamental limit.

    Extension contract
    ------------------
    epsilon() returns a tensor *broadcastable* against the simulation grid.
    Isotropic materials return shape [] or [wl]; an anisotropic subclass
    may later return [..., 3, 3] without changing this base class or any
    shape-agnostic downstream code. Magnetic materials can be supported by
    adding a mu() method — an additive change that does not break the
    existing epsilon() contract.
    """

    def epsilon(self, wavelength: torch.Tensor) -> torch.Tensor:
        """Complex relative permittivity at the given wavelength(s)."""
        raise NotImplementedError
    
    
class ConstantMaterial(Material):
    """
    Material with a wavelength-independent permittivity.
    
    Parameters
    ----------
    eps_real : float | Tensor | nn.Parameter
        Real part of the relative permittivity. 
    eps_imag : float | Tensor | nn.Parameter, optional
        Imaginary part (loss). Defaults to 0.
    """

    def __init__(self, eps_real, eps_imag=0.0):
        super().__init__()
        register(self, "eps_real", eps_real)
        register(self, "eps_imag", eps_imag)

    def epsilon(self, wavelength: torch.Tensor) -> torch.Tensor:
        return torch.complex(self.eps_real, self.eps_imag)