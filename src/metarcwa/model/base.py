# metarcwa/model/base.py
# DESCRIPTION

import torch
import torch.nn as nn
import dataclasses
from dataclasses import dataclass

from .stack import Stack
from .source import Source
from .._dtypes import _REAL_TO_COMPLEX
from .medium import MediumSpec
from .layer import PatternedLayer, HomogeneousLayer

@dataclass(frozen=True)
class ModelSpec:
    """Complete Model -> Solver snapshot: structure + illumination.

    An immutable snapshot after geometry rasterization and material evaluation at the
    source wavelength; Solver cannot reach back into the Model.

    Structure
    ---------
    layers : tuple[HomogeneousLayer | PatternedLayer, ...]
        Ordered finite layers, incidence side first.
    incidence : MediumSpec
        Semi-infinite incidence medium. Lossless.
    transmission : MediumSpec
        Semi-infinite transmission medium. May be lossy.
    a1, a2 : Tensor | nn.Parameter
        Lattice vectors, shape [2] each.

    Illumination
    ------------
    wavelength : Tensor | nn.Parameter
        Free-space wavelength, shape ``[N_wl, 1, 1]``. Sole carrier of
        physical scale — k0 = 2*pi/wavelength is reconstructed downstream.
    kx0, ky0 : Tensor | nn.Parameter
        k0-normalized in-plane wavevector, shape ``[N_wl, N_theta, N_phi]``.
    """

    # structure (source-independent)
    layers: tuple[HomogeneousLayer | PatternedLayer, ...]
    incidence: MediumSpec
    transmission: MediumSpec
    a1: torch.Tensor
    a2: torch.Tensor
    # illumination (Source-specific — see note in base.py header)
    wavelength: torch.Tensor
    kx0: torch.Tensor
    ky0: torch.Tensor

class Model(nn.Module):
    """
    Simulation model for RCWA.
    """

    def __init__(self, stack: Stack, source: Source):
        super().__init__()
        self.stack = stack
        self.source = source

    # ------------------------------------------------------------------
    # Device / dtype
    # ------------------------------------------------------------------

    def to(self, *args, **kwargs) -> "Model":
        """Move and/or cast all model parameters and buffers.

        Behaves like ``nn.Module.to`` but is **type-aware**: when a real
        floating dtype is requested, complex tensors (e.g. permittivity
        buffers) are cast to the matching complex dtype instead of having
        their imaginary part silently discarded.

        Pass a real floating dtype (``torch.float32``, ``torch.float64``,
        …); complex tensors will follow automatically:

        =========  =============
        real dtype  complex dtype
        =========  =============
        float16    complex32
        float32    complex64
        float64    complex128
        =========  =============

        Parameters
        ----------
        *args, **kwargs
            Same as ``nn.Module.to`` — accepts device, dtype, or both.

        Returns
        -------
        Model
            ``self``, so calls are chainable (``model.to(...).spec()``).
        """
        device, dtype, non_blocking, convert_to_format = torch._C._nn._parse_to(*args, **kwargs)

        if dtype is not None and not dtype.is_floating_point:
            raise TypeError(
                f"Model.to() expects a real floating dtype (e.g. torch.float64); got {dtype}. "
                "Complex tensors are cast to the matching complex dtype automatically."
            )

        complex_dtype = _REAL_TO_COMPLEX.get(dtype)

        def convert(t: torch.Tensor) -> torch.Tensor:
            d = None
            if dtype is not None:
                if t.is_complex():
                    d = complex_dtype
                elif t.is_floating_point():
                    d = dtype
                # non-float tensors (int indices etc.) left untouched
            if convert_to_format is not None and t.dim() in (4, 5):
                return t.to(device, d, non_blocking, memory_format=convert_to_format)
            return t.to(device, d, non_blocking)

        return self._apply(convert)

    # ------------------------------------------------------------------
    # Spec
    # ------------------------------------------------------------------

    def spec(self, nx: int, ny: int) -> ModelSpec:
        """Build the complete Model -> Solver description.

        Returns an immutable ``ModelSpec`` snapshot using the model's
        current device and dtype (set via ``model.to(...)``).

        Parameters
        ----------
        nx : int
            Grid resolution along x for patterned layers' shape_fn.
        ny : int
            Grid resolution along y, same caveat as nx.

        Returns
        -------
        ModelSpec
            Immutable spec container ready for the Solver.

        Notes
        -----
        Each ``ModelSpec`` field is pulled automatically from whichever of
        ``stack_spec``/``source_spec`` declares it (matched by name via
        ``dataclasses.fields``), instead of being copied field-by-field by
        hand — adding a field only means adding it to the sub-spec and to
        ``ModelSpec`` itself, not also updating this method
        (ANALYSIS.md A6).
        """
        wavelength = self.source.wavelength
        stack_spec = self.stack.spec(wavelength, nx, ny)
        source_spec = self.source.spec(stack_spec.incidence.refractive_index().real) # Ignore imag part for the incidence
        sub_specs = (stack_spec, source_spec)
        kwargs = {}
        for f in dataclasses.fields(ModelSpec):
            for sub in sub_specs:
                if hasattr(sub, f.name):
                    kwargs[f.name] = getattr(sub, f.name)
                    break
            else:
                raise AttributeError(
                    f"ModelSpec.{f.name!r} not found on stack_spec or source_spec"
                )
        return ModelSpec(**kwargs)
