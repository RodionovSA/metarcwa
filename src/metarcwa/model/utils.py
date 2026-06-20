# metarcwa/model/utils.py
# DESCRIPTION

import warnings
import torch
import torch.nn as nn
from typing import Callable

# Maps a real floating dtype to its matching complex dtype. 
_REAL_TO_COMPLEX = {
    torch.float16: torch.complex32,
    torch.float32: torch.complex64,
    torch.float64: torch.complex128,
}


def to_complex(t: torch.Tensor) -> torch.Tensor:
    """Return ``t`` as a complex tensor, promoting if necessary.

    If ``t`` is already complex, it is returned unchanged (no copy). If it is
    real, it is cast to the matching complex dtype (e.g. float32 → complex64).
    """
    if t.is_complex():
        return t
    return t.to(_REAL_TO_COMPLEX.get(t.dtype, torch.complex64))


def to_real(t: torch.Tensor, *, name: str = "tensor", atol: float = 1e-8) -> torch.Tensor:
    """Return ``t`` as a real tensor, dropping the imaginary part if needed.

    If ``t`` is already real, it is returned unchanged. If it is complex and
    has a non-negligible imaginary part (``max|imag| > atol``), a
    ``UserWarning`` is emitted before dropping the imaginary part.

    Parameters
    ----------
    t : Tensor
        Input tensor.
    name : str
        Human-readable name used in the warning message.
    atol : float
        Threshold above which the imaginary part is considered non-negligible.
    """
    if not t.is_complex():
        return t
    if t.imag.abs().max() > atol:
        warnings.warn(
            f"{name} has a non-negligible imaginary part "
            f"(max |imag| = {t.imag.abs().max().item():.3g}); "
            "casting to real (medium treated as lossless).",
            UserWarning,
            stacklevel=2,
        )
    return t.real


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


def from_metashapes(shape, soft: bool, softness: float) -> CallableModule:
    """Convert a MetaShapes ``Shape`` into a ``shape_fn(lattice, nx, ny)`` callable.

    The returned ``CallableModule`` registers ``shape`` as a submodule so its
    geometry parameters (center, size, angle, …) appear in ``model.parameters()``
    / ``model.buffers()`` and move with ``model.to()``.

    Parameters
    ----------
    shape : metashapes.Shape
        MetaShapes shape object defining the geometry.
    soft : bool
        If ``True``, use a smooth (differentiable) transition at the
        shape boundary instead of a hard 0/1 step.
    softness : float
        Controls boundary smoothing width when ``soft=True``.

    Returns
    -------
    CallableModule
        A callable ``shape_fn(lattice, nx, ny) -> mask`` registered as an
        ``nn.Module``.
    """
    try:
        from metashapes import UnitCell
    except ImportError:
        raise ImportError("metashapes must be installed to use from_metashapes")

    def shape_fn(lattice, nx, ny):
        cell = UnitCell(lattice=lattice, scene=shape)
        return cell.mask(nx=nx, ny=ny, soft=soft, softness=softness)

    return CallableModule(shape_fn, shape)


def from_dispertorch(dispersion) -> CallableModule:
    """Convert a DisperTorch dispersion model into an ``eps_fn(wavelength)`` callable.

    The returned ``CallableModule`` registers ``dispersion`` as a submodule so
    its coefficients appear in ``model.parameters()`` / ``model.buffers()`` and
    move with ``model.to()``.

    Parameters
    ----------
    dispersion : dispertorch.DispersionModel
        A DisperTorch dispersion model (e.g. ``dispertorch.material("Au")``).

    Returns
    -------
    CallableModule
        A callable ``eps_fn(wavelength) -> complex eps`` registered as an
        ``nn.Module``.

    Notes
    -----
    ``DispersionModel`` is already callable (its ``forward`` aliases
    ``permittivity``), so you may also pass the model directly as
    ``eps_solid_fn`` without this adapter — both approaches propagate tensors
    correctly.
    """
    return CallableModule(dispersion.permittivity, dispersion)
