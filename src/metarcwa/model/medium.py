# metarcwa/model/medium.py
# DESCRIPTION

import torch
import torch.nn as nn
from dataclasses import dataclass

from .utils import CallableModule, to_complex

@dataclass(frozen=True)
class MediumSpec:
    """Marker base — no fields. eps/mu shape contracts differ per
    variant and aren't safe to inherit."""
    pass


class Medium(nn.Module):
    """Base class for material response. Not instantiated directly.

    Subclass identity is the solver's dispatch key — it determines which
    mode equation is solved (isotropic scalar shortcut, anisotropic 3x3,
    or full magnetic eigenproblem with mu != I). 
    """
    pass

@dataclass(frozen=True)
class IsotropicMediumSpec(MediumSpec):
    eps: torch.Tensor                  # [*B] complex
    
    def refractive_index(self) -> torch.Tensor:
        """Isotropic n = sqrt(eps)."""
        return torch.sqrt(self.eps)

class IsotropicMedium(Medium):
    """Non-magnetic, isotropic medium (mu = 1).

    Parameters
    ----------
    eps_fn : CallableModule
        Permittivity, called as ``eps_fn(wavelength) -> eps``. Must return a
        complex tensor of exactly the same shape as ``wavelength`` 

        **Parameter visibility:** for the callable's tensors to appear in
        ``model.parameters()`` / ``model.buffers()`` and move with
        ``model.to()``, the callable must be an ``nn.Module`` (PyTorch only
        traverses modules, not plain functions or closures). Use
        ``from_dispertorch(disp)`` to wrap a DisperTorch model, or
        ``CallableModule(fn, dep1, dep2, …)`` to wrap any plain callable and
        register its dependencies. A bare ``lambda`` works numerically but
        its closed-over tensors are invisible to the model.
    """

    def __init__(self, eps_fn: CallableModule):
        super().__init__()
        if not callable(eps_fn):
            raise TypeError("eps_fn must be callable.")
        self.eps_fn = eps_fn
        
    def spec(self, wvl: torch.Tensor) -> IsotropicMediumSpec:
        eps = to_complex(self.eps_fn(wvl))
        if eps.shape != wvl.shape:
            raise ValueError(
                f"eps_fn must return a tensor of shape {wvl.shape} "
                f"(matching wavelength), got {eps.shape}."
            )
        return IsotropicMediumSpec(eps=eps)
        
