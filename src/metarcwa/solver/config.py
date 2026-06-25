# metarcwa/solver/config.py
"""
config — solver hyperparameters for the RCWA pipeline
======================================================

Two public dataclasses:

  Factorization   — TVF (Tangent Vector Field) factorization settings.
  Config          — Top-level solver configuration.

Both are plain ``@dataclass`` types; pass a ``Config`` instance to
``LayerSolver`` to control grid resolution, harmonic truncation,
factorization method, and eigenvalue solver behaviour.
"""

import yaml
import torch
from dataclasses import dataclass, field
from pathlib import Path

_DTYPE_TO_STR: dict[torch.dtype, str] = {
    torch.float32:    "float32",
    torch.float64:    "float64",
    torch.complex64:  "complex64",
    torch.complex128: "complex128",
}
_STR_TO_DTYPE: dict[str, torch.dtype] = {v: k for k, v in _DTYPE_TO_STR.items()}


@dataclass
class Factorization:
    """
    Settings for the TVF (Tangent Vector Field) Li-factorization rule.

    Controls how the permittivity Fourier convolution matrix is computed
    for patterned layers.  The TVF smoothly interpolates the factorization
    direction field across material interfaces, improving convergence of the
    Fourier series.

    Attributes
    ----------
    method : str
        TVF algorithm.  One of ``"Jones"``, ``"Pol"``, ``"Normal"``,
        ``"Jones_direct"``.  Default ``"Jones"``.
    optimizer : str
        Optimiser used to fit the TVF field.  Default ``"newton"``.
    steps : int
        Number of optimiser steps.  Default ``1`` (exact for the Newton
        quadratic, sufficient in most cases).
    alpha : float
        Alignment loss weight (gradient-field alignment term).  Default ``1.0``.
    beta : float
        Fourier regularisation weight (band-limit smoothness).  Default ``0.05``.
    gamma : float
        Smoothness loss weight (spatial smoothness).  Default ``0.05``.
    """

    method: str    = "Jones"    # "Normal" | "Pol" | "Jones" | "Jones_direct"
    optimizer: str = "newton"
    steps: int     = 1
    alpha: float   = 1.0
    beta: float    = 0.05
    gamma: float   = 0.05

    def to_dict(self) -> dict:
        return {
            "method": self.method,
            "optimizer": self.optimizer,
            "steps": self.steps,
            "alpha": self.alpha,
            "beta": self.beta,
            "gamma": self.gamma,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Factorization":
        return cls(**d)


@dataclass
class Config:
    """
    Top-level solver configuration for the RCWA pipeline.

    Pass an instance of this class to ``LayerSolver`` to control grid
    resolution, Fourier truncation, TVF factorization, and eigenvalue
    solver behaviour.

    Attributes
    ----------
    dtype : torch.dtype
        Floating-point precision for all computations.  Default
        ``torch.float32``; use ``torch.float64`` for higher accuracy.
    device : str or torch.device
        Target device, e.g. ``"cpu"`` or ``"cuda"``.  Default ``"cpu"``.
    nx : int
        Real-space grid resolution along the **a1** lattice direction
        (number of pixels).  Default ``128``.
    ny : int
        Real-space grid resolution along the **a2** lattice direction.
        Default ``128``.
    m : int
        Number of retained Fourier harmonics along **a1**.  The total
        harmonic count along a1 is ``2m + 1``.  Default ``12``.
    n : int
        Number of retained Fourier harmonics along **a2**.  Default ``12``.
    truncation : str
        Harmonic truncation scheme.  ``"circular"`` keeps harmonics inside
        an ellipse (smoother convergence); ``"rectangular"`` keeps all
        ``(2m+1)×(2n+1)`` harmonics.  Default ``"circular"``.
    factorization : Factorization or None
        TVF Li-factorization settings.  ``None`` disables TVF and uses the
        plain Laurent convolution rule.  Default ``Factorization()``.
    modesolver : str
        Mode-solving strategy for patterned layers.  Currently only
        ``"eig"`` (full eigendecomposition via :func:`eigsolver`) is
        supported.  Default ``"eig"``.
    eigsolver_stable : bool
        If ``True`` (default), use :class:`Eig` with Lorentzian-broadened
        gradients for stability near degenerate eigenvalues.  Set to
        ``False`` to use ``torch.linalg.eig`` directly (faster but
        gradients can be NaN near degeneracies).
    """

    dtype:            torch.dtype         = torch.float32
    device:           torch.device        = "cpu"
    nx:               int                 = 128
    ny:               int                 = 128
    m:                int                 = 12
    n:                int                 = 12
    truncation:       str                 = "circular"    # "circular" | "rectangular"
    factorization:    Factorization|None  = field(default_factory=Factorization)
    modesolver:       str                 = "eig"         # "eig"
    eigsolver_stable: bool                = True

    def __post_init__(self) -> None:
        if not isinstance(self.device, torch.device):
            self.device = torch.device(self.device)

    def to_dict(self) -> dict:
        return {
            "dtype": _DTYPE_TO_STR[self.dtype],
            "device": str(self.device),
            "nx": self.nx,
            "ny": self.ny,
            "m": self.m,
            "n": self.n,
            "truncation": self.truncation,
            "factorization": self.factorization.to_dict() if self.factorization is not None else None,
            "modesolver": self.modesolver,
            "eigsolver_stable": self.eigsolver_stable,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Config":
        d = dict(d)
        d["dtype"] = _STR_TO_DTYPE[d["dtype"]]
        d["device"] = torch.device(d["device"])
        if d.get("factorization") is not None:
            d["factorization"] = Factorization.from_dict(d["factorization"])
        return cls(**d)

    def to_yaml(self, path: str | Path) -> None:
        with open(path, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Config":
        with open(path) as f:
            return cls.from_dict(yaml.safe_load(f))
