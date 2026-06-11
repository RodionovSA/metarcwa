# test for eps_fn

## Test epsilon with a material like gold:
## Create gold from DisperTorch, wrap it with `from_dispertorch`
## call `eps_fn(wl)` and inspect output

import matplotlib.pyplot as plt
from typing import Callable

import torch

from dispertorch import material, list_materials

print(list_materials())

def from_dispertorch(dispersion) -> Callable:
    """ 
    Convers a DisperTorch dispersion model into eps_fn(wavelength).
    """
    def eps_fn(wl):
        return dispersion.permittivity(wl)

    return eps_fn

au = material("Au")
wl = torch.linspace(400,800,50)

eps_fn = from_dispertorch(au)
eps = eps_fn(wl)

print(eps.shape)
print(eps.dtype)

plt.plot(wl, eps.real, label = "Real component of permittivity")
plt.plot(wl, eps.imag, label = "Imaginary component of permittivity")
plt.legend()
plt.show()