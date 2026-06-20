# metarcwa/model/base.py
# DESCRIPTION

import torch
import torch.nn as nn

from .stack import Stack
from .source import Source
from .spec import ModelSpec
from .utils import _REAL_TO_COMPLEX  

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

    def spec(self) -> ModelSpec:
        """Build the complete Model -> Solver description.

        Returns an immutable ``ModelSpec`` snapshot using the model's
        current device and dtype (set via ``model.to(...)``).

        Returns
        -------
        ModelSpec
            Immutable spec container ready for the Solver.
        """
        wavelength = self.source.wavelength
        stack_spec = self.stack.spec(wavelength)
        source_spec = self.source.spec(stack_spec.eps_incidence)
        return ModelSpec(stack=stack_spec, source=source_spec)
