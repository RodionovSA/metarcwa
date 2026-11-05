# metarcwa/model/pml.py
"""Lateral perfectly matched layer (PML) of a rectangular supercell.

The PML occupies a band of width ``w`` at each edge of the cell along every
enabled axis. Inside it the in-plane coordinate is complex-stretched by a
factor ``s(d)``, where ``d`` is the normalized distance into the PML (``0`` at
its inner edge, ``1`` at the cell edge); ``s = 1`` in the physical region.
The profile ``s(d)`` is pluggable. See ``docs/PML.md``.
"""

import math
import torch
import torch.nn as nn
from dataclasses import dataclass

from .nn_helpers import register
from .lattice import Lattice
from .._dtypes import to_complex


class StretchAbsorbProfile(nn.Module):
    """Real stretch times imaginary absorption (the fmmax profile)::

        s(d) = (1 + a_max * d**p) * (1 + 1j * sigma_max * sin(pi/2 * d)**2)

    Parameters
    ----------
    a_max : float | Tensor
        Strength of the real stretch (``>= 0``).
    p : float | Tensor
        Grading order of the real stretch (``> 0``).
    sigma_max : float | Tensor
        Absorption strength (``> 0``).
    """

    def __init__(self, a_max=4.0, p=4.0, sigma_max=1.0):
        super().__init__()
        register(self, "a_max", a_max)
        register(self, "p", p)
        register(self, "sigma_max", sigma_max)
        if self.a_max < 0 or self.p <= 0 or self.sigma_max <= 0:
            raise ValueError("require a_max >= 0, p > 0, sigma_max > 0")

    def forward(self, d: torch.Tensor) -> torch.Tensor:
        real = 1 + self.a_max * d ** self.p
        imag = self.sigma_max * torch.sin(math.pi / 2 * d) ** 2
        return to_complex(real) * (1 + 1j * imag)


class PolynomialProfile(nn.Module):
    """Graded absorption without real stretch::

        s(d) = 1 + 1j * sigma_max * d**p

    Parameters
    ----------
    sigma_max : float | Tensor
        Absorption strength (``> 0``).
    p : float | Tensor
        Grading order (``> 0``).
    """

    def __init__(self, sigma_max=1.0, p=3.0):
        super().__init__()
        register(self, "sigma_max", sigma_max)
        register(self, "p", p)
        if self.sigma_max <= 0 or self.p <= 0:
            raise ValueError("require sigma_max > 0, p > 0")

    def forward(self, d: torch.Tensor) -> torch.Tensor:
        return 1 + 1j * to_complex(self.sigma_max * d ** self.p)


@dataclass(frozen=True)
class PMLSpec:
    """Stretch profiles sampled on the pattern grid.

    Attributes
    ----------
    sx : Tensor | None
        Complex stretch factor along x, shape ``[Nx]`` (grid axis -1), or
        ``None`` if the PML is disabled along x.
    sy : Tensor | None
        Complex stretch factor along y, shape ``[Ny]`` (grid axis -2), or
        ``None`` if the PML is disabled along y.
    """
    sx: torch.Tensor | None
    sy: torch.Tensor | None


class PML(nn.Module):
    """Lateral PML at the edges of a rectangular supercell.

    Parameters
    ----------
    width : float | tuple[float, float] | Tensor | nn.Parameter
        PML width along (x, y), in the lattice's length units. A scalar
        applies to both axes; ``0`` disables the PML along that axis. To
        make it optimizable, pass a 2-element ``nn.Parameter``; a scalar
        ``nn.Parameter`` is rejected.
    profile : nn.Module, optional
        Stretch profile, called as ``profile(d) -> s`` with ``d`` a real
        tensor in ``[0, 1]``; returns the complex ``s`` of the same shape.
        Must satisfy ``s(0) = 1``, ``Re s > 0`` and ``Im s >= 0``. Wrap a
        plain callable in ``CallableModule``. Defaults to
        ``StretchAbsorbProfile()``.
    method : str
        PML implementation. Only ``"stretch"`` (complex coordinate
        stretching of the wavevector operators) is available.
    """

    METHODS = ("stretch",)

    def __init__(self, width, profile: nn.Module | None = None, method: str = "stretch"):
        super().__init__()
        if method not in self.METHODS:
            raise ValueError(f"method must be one of {self.METHODS}, got {method!r}")
        profile = profile if profile is not None else StretchAbsorbProfile()
        if not isinstance(profile, nn.Module):
            raise TypeError("profile must be an nn.Module (wrap callables in CallableModule).")
        width = torch.as_tensor(width, dtype=torch.get_default_dtype())
        if width.numel() == 1:
            if isinstance(width, nn.Parameter) or width.requires_grad:
                raise TypeError(
                    "width must be a plain scalar, not a Parameter or a "
                    "grad-carrying tensor: a scalar width is broadcast to both "
                    "axes, and nn.Parameter must be a leaf tensor, so the "
                    "connection to it would be silently severed. For an "
                    "optimizable width, pass both axes directly: "
                    "PML(width=nn.Parameter(torch.tensor([wx, wy])))."
                )
            width = width.reshape(1).expand(2).clone()
        register(self, "width", width)
        self.profile = profile
        self.method = method

        if self.width.shape != (2,):
            raise ValueError(f"width must be a scalar or have 2 entries, got shape {tuple(self.width.shape)}")
        if (self.width < 0).any() or not (self.width > 0).any():
            raise ValueError("width must be >= 0 on both axes and > 0 on at least one")

    @staticmethod
    def _periods(lattice: Lattice) -> tuple[torch.Tensor, torch.Tensor]:
        """Periods (px, py) of a rectangular lattice; raise for any other lattice."""
        a1, a2 = lattice.a1, lattice.a2
        if a1[1] != 0 or a2[0] != 0:
            raise ValueError(
                "PML requires a rectangular lattice (a1 along x, a2 along y), "
                "e.g. Lattice.rectangular(px, py)."
            )
        return a1[0].abs(), a2[1].abs()

    @staticmethod
    def _distance(period: torch.Tensor, n: int, width: torch.Tensor) -> torch.Tensor:
        """Normalized distance into the PML at the ``n`` pixel centres, shape ``[n]``."""
        x = (torch.arange(n, dtype=period.dtype, device=period.device) + 0.5) / n * period - period / 2
        if width == 0:
            return torch.zeros_like(x)
        return ((x.abs() - (period / 2 - width)) / width).clamp(0.0, 1.0)

    def check(self, lattice: Lattice) -> None:
        """Validate the PML against ``lattice``: rectangular, and widths fit in the cell."""
        px, py = self._periods(lattice)
        if 2 * self.width[0] >= px or 2 * self.width[1] >= py:
            raise ValueError(
                f"PML widths {self.width.tolist()} leave no physical region in a "
                f"cell of size ({float(px)}, {float(py)})"
            )

    def spec(self, lattice: Lattice, nx: int, ny: int) -> PMLSpec:
        """Sample the stretch profile on the ``[ny, nx]`` pattern grid."""
        px, py = self._periods(lattice)
        s0 = self.profile(torch.zeros(1, dtype=px.dtype, device=px.device))
        if not torch.allclose(s0, torch.ones_like(s0)):
            raise ValueError(f"PML profile must satisfy s(0) = 1, got {s0.item()}")
        s = []
        for period, n, width in ((px, nx, self.width[0]), (py, ny, self.width[1])):
            if width == 0:
                s.append(None)
                continue
            s_axis = self.profile(self._distance(period, n, width))
            if (s_axis.real <= 0).any() or (s_axis.imag < 0).any():
                raise ValueError("PML profile must satisfy Re s > 0 and Im s >= 0")
            s.append(s_axis)
        return PMLSpec(sx=s[0], sy=s[1])

    def pad(self, pattern: torch.Tensor, lattice: Lattice) -> torch.Tensor:
        """Edge-pad ``pattern`` (``[..., Ny, Nx]``) into the PML region.

        Every PML pixel takes the value of the nearest physical-region pixel
        along the PML normal, so the geometry does not vary along the normal
        inside the PML.
        """
        px, py = self._periods(lattice)
        ny, nx = pattern.shape[-2:]
        idx = []
        for period, n, width in ((px, nx, self.width[0]), (py, ny, self.width[1])):
            inside = (self._distance(period, n, width) == 0).nonzero().squeeze(-1)
            if inside.numel() == 0:
                raise ValueError(f"PML leaves no physical pixels on a grid of {n} points")
            i = torch.arange(n, device=pattern.device)
            idx.append(i.clamp(int(inside.min()), int(inside.max())))
        ix, iy = idx
        return pattern[..., iy[:, None], ix[None, :]]
