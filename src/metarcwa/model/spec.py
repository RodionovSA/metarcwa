# metarcwa/model/spec.py
# DESCRIPTION

from dataclasses import dataclass, fields
from typing import Iterator
import torch
import torch.nn as nn


class Spec:
    """Base for solver-facing spec containers.

    Provides generic tensor traversal and mapping over all tensor leaves,
    recursing into nested ``Spec`` instances automatically.
    """

    def named_tensors(self, prefix: str = "") -> Iterator[tuple[str, torch.Tensor]]:
        """Yield ``(name, tensor)`` for every tensor leaf in this spec.

        Recurses into nested ``Spec`` fields using dotted names, e.g.
        ``"stack.a1"``, ``"source.wavelength"``.  ``nn.Parameter`` values
        are yielded as-is (``nn.Parameter`` is a ``Tensor`` subclass).
        """
        for f in fields(self):
            value = getattr(self, f.name)
            name = f"{prefix}{f.name}"
            if isinstance(value, Spec):
                yield from value.named_tensors(prefix=f"{name}.")
            elif isinstance(value, torch.Tensor):
                yield name, value

    def tensors(self) -> Iterator[torch.Tensor]:
        """Yield every tensor leaf (values only, no names)."""
        for _, t in self.named_tensors():
            yield t


@dataclass(frozen=True)
class StackSpec(Spec):
    """Resolved structure crossing the Model -> Solver boundary.

    An immutable snapshot of the stack after geometry has been rasterized
    and materials evaluated at the source wavelength. The Solver reads this
    and cannot reach back into the Model.

    Attributes
    ----------
    layer_eps : Tensor | nn.Parameter
        Per-layer permittivity on the grid, shape ``[N_layers, ..., Ny, Nx]``
        (rows = y, cols = x). Leading dims after ``N_layers`` may carry a
        geometry/wavelength batch. **Always complex** (promoted by
        ``Stack.spec()`` if the user supplies a real eps callable).
    layer_thickness : Tensor | nn.Parameter
        Thickness of each finite layer, shape [N_layers] (broadcastable
        against any batch).
    eps_incidence : Tensor | nn.Parameter
        Permittivity of the semi-infinite incidence medium (uniform).
        **Always real** — the incidence medium is treated as lossless so that
        ``n = sqrt(eps_inc)`` and the resulting wavevectors are real.
        A ``UserWarning`` is emitted if the callable returns a complex value
        with non-negligible imaginary part.
    eps_transmission : Tensor | nn.Parameter
        Permittivity of the semi-infinite transmission medium (uniform).
        **Always complex** (same promotion as ``layer_eps``).
    a1 : Tensor | nn.Parameter
        First lattice vector, shape [2]. Needed for reciprocal-lattice
        construction.
    a2 : Tensor | nn.Parameter
        Second lattice vector, shape [2]. Same convention as ``a1``.
    """

    layer_eps: torch.Tensor
    layer_thickness: torch.Tensor
    eps_incidence: torch.Tensor
    eps_transmission: torch.Tensor
    a1: torch.Tensor
    a2: torch.Tensor


@dataclass(frozen=True)
class SourceSpec(Spec):
    """Abstract base for the solver-facing illumination description.

    A SourceSpec carries the *physical*, pre-expansion description of the
    illumination. Expansion onto Fourier harmonics is the Solver's job, so
    nothing solver-side (e.g. the harmonic count) appears here. Concrete
    variants — PlaneWaveSpec, and beam specs in future — subclass this.

    Attributes
    ----------
    wavelength : Tensor | nn.Parameter
        Free-space wavelength. May be batched.
    """

    wavelength: torch.Tensor


@dataclass(frozen=True)
class PlaneWaveSpec(SourceSpec):
    """Plane-wave illumination.

    All batch axes follow the outer-product sweep convention set by
    ``PlaneWave.spec()``: ``[N_wl, N_theta, N_phi]``, with singleton axes
    for scalar parameters.

    Attributes
    ----------
    wavelength : Tensor | nn.Parameter
        Free-space wavelength, shape ``[N_wl, 1, 1]``.
    kx0, ky0 : Tensor | nn.Parameter
        In-plane wavevector components of the incident wave, already
        resolved using the incidence-medium index. Shape ``[N_wl, N_theta, N_phi]``.
    s, p : Tensor | nn.Parameter
        Complex s- and p-polarization amplitudes.
    """

    kx0: torch.Tensor
    ky0: torch.Tensor
    s: torch.Tensor
    p: torch.Tensor


@dataclass(frozen=True)
class ModelSpec(Spec):
    """Complete Model output: structure + illumination, for the Solver."""

    stack: StackSpec
    source: SourceSpec
