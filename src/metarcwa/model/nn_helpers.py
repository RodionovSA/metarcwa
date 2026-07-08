# metarcwa/model/nn_helpers.py
# Generic nn.Module registration helpers shared across the model package.

import torch
import torch.nn as nn
from typing import Callable


def register(module, name, value, dtype=torch.float32):
    """Register `value` on `module` under `name`.

    If `value` is an nn.Parameter it becomes an optimizable parameter;
    otherwise it is stored as a (non-gradient) buffer that still moves
    with .to() and is saved in state_dict().
    """
    if isinstance(value, nn.Parameter):
        setattr(module, name, value)
    else:
        module.register_buffer(name, torch.as_tensor(value, dtype=dtype))


class CallableModule(nn.Module):
    """Wrap a plain callable as an ``nn.Module`` so its dependencies are
    tracked by the owning model.

    When a callable closes over ``nn.Module`` or ``nn.Parameter`` objects,
    PyTorch cannot see them — they won't appear in ``model.parameters()``
    or ``model.buffers()`` and won't move with ``model.to()``.
    ``CallableModule`` solves this by registering those dependencies as
    submodules / parameters so the owning model traverses them normally.

    Parameters
    ----------
    fn : Callable
        The callable to delegate to in ``forward``.
    *deps : nn.Module | nn.Parameter
        Dependencies that ``fn`` closes over and that should be tracked.
        ``nn.Module`` deps are stored in an ``nn.ModuleList``; bare
        ``nn.Parameter`` deps are registered individually.
        Other types are silently ignored (no tensors to track).

    Examples
    --------
    Wrapping a dispersion model so its buffers propagate::

        eps_fn = CallableModule(dispersion.permittivity, dispersion)
        layer  = Layer(eps_solid_fn=eps_fn, ...)
        # model.buffers() now includes dispersion's coefficients

    Wrapping a function that closes over a learnable parameter::

        radius = nn.Parameter(torch.tensor(0.3))
        shape_fn = CallableModule(lambda lat, nx, ny: make_mask(lat, nx, ny, radius), radius)
        # model.parameters() now yields radius
    """

    def __init__(self, fn: Callable, *deps):
        super().__init__()
        if not callable(fn):
            raise TypeError(f"fn must be callable, got {type(fn)}")
        self.fn = fn
        mods = [d for d in deps if isinstance(d, nn.Module)]
        if mods:
            self._deps = nn.ModuleList(mods)
        for i, p in enumerate(d for d in deps if isinstance(d, nn.Parameter)):
            self.register_parameter(f"_dep_param_{i}", p)

    def forward(self, *args, **kwargs):
        return self.fn(*args, **kwargs)
