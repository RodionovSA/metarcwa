# metarcwa/model/adapters.py
# Adapters converting external-package objects (MetaShapes, DisperTorch)
# into metarcwa-native CallableModule callables.

from .nn_helpers import CallableModule


def from_metashapes(shape, soft: bool = False, softness: float = 0.0) -> CallableModule:
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
