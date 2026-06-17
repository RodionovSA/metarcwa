# metarcwa/model/lattice.py
# DESCRIPTION

import math
import torch
import torch.nn as nn

from .utils import register

class Lattice(nn.Module):
    """In-plane periodicity of the unit cell.
    Defined by two lattice vectors. A rectangular lattice is the special
    case of axis-aligned vectors.

    Parameters
    ----------
    a1 : float | Tensor | nn.Parameter
        First lattice vector, shape [2]. Stored as a buffer by default;
        pass an ``nn.Parameter`` to make it optimizable.
    a2 : float | Tensor | nn.Parameter
        Second lattice vector, shape [2]. Same convention as ``a1``.
    """
    def __init__(self, a1, a2):
        super().__init__()
        register(self, "a1", a1)
        register(self, "a2", a2)
        if self.cell_area <= 1e-12:
            raise ValueError("a1, a2 must be linearly independent")
        
    @property
    def device(self): return self.a1.device
    @property
    def dtype(self):  return self.a1.dtype

    @classmethod
    def rectangular(cls, px, py) -> "Lattice":
        """Build an axis-aligned rectangular lattice from two periods.

        Parameters
        ----------
        px, py : float | nn.Parameter
            Period along x and y. Pass ``nn.Parameter`` to make the
            corresponding lattice vector optimizable.
        """
        px_v = px.detach().item() if isinstance(px, torch.Tensor) else float(px)
        py_v = py.detach().item() if isinstance(py, torch.Tensor) else float(py)
        a1 = torch.tensor([px_v, 0.0])
        a2 = torch.tensor([0.0, py_v])
        if isinstance(px, nn.Parameter):
            a1 = nn.Parameter(a1)
        if isinstance(py, nn.Parameter):
            a2 = nn.Parameter(a2)
        return cls(a1=a1, a2=a2)

    @classmethod
    def hexagonal(cls, a, *, orientation: str = "pointy") -> "Lattice":
        """Build a hexagonal (triangular) lattice from the lattice constant.

        Parameters
        ----------
        a : float | nn.Parameter
            Nearest-neighbour distance (lattice constant). Pass
            ``nn.Parameter`` to make both lattice vectors optimizable.
        orientation : str
            ``"pointy"`` — Wigner-Seitz cell has vertices at top and bottom;
            lattice vectors at 0° and 60°::

                a1 = [a,       0          ]
                a2 = [a/2,     a·√3/2     ]

            ``"flat"`` — Wigner-Seitz cell has flat edges at top and bottom;
            lattice vectors at 30° and 90°::

                a1 = [a·√3/2,  a/2        ]
                a2 = [0,       a           ]
        """
        a_v = a.detach().item() if isinstance(a, torch.Tensor) else float(a)
        if orientation == "pointy":
            a1 = torch.tensor([a_v, 0.0])
            a2 = torch.tensor([a_v / 2.0, a_v * math.sqrt(3) / 2.0])
        elif orientation == "flat":
            a1 = torch.tensor([a_v * math.sqrt(3) / 2.0, a_v / 2.0])
            a2 = torch.tensor([0.0, a_v])
        else:
            raise ValueError(f"orientation must be 'pointy' or 'flat', got {orientation!r}")
        if isinstance(a, nn.Parameter):
            a1, a2 = nn.Parameter(a1), nn.Parameter(a2)
        return cls(a1=a1, a2=a2)

    # --- matrix form: columns are the lattice vectors ----------------
    @property
    def matrix(self) -> torch.Tensor:
        """2x2 matrix A with columns [a1 | a2]."""
        return torch.stack([self.a1, self.a2], dim=1)

    @property
    def cell_area(self) -> torch.Tensor:
        return torch.linalg.det(self.matrix).abs()

    # --- coordinate transforms ---------------------------------------
    def to_fractional(self, x: torch.Tensor, y: torch.Tensor):
        """Cartesian (x, y) -> fractional (f1, f2)."""
        p = torch.stack([x, y], dim=0)
        f = torch.einsum('ij,j...->i...', torch.linalg.inv(self.matrix), p)
        return f[0], f[1]

    def to_cartesian(self, f1: torch.Tensor, f2: torch.Tensor):
        """Fractional (f1, f2) -> Cartesian (x, y)."""
        f = torch.stack([f1, f2], dim=0)
        p = torch.einsum('ij,j...->i...', self.matrix, f)
        return p[0], p[1]

    # --- periodic copies ---------------------------------------------
    def offset(self, i: int, j: int) -> torch.Tensor:
        """Cartesian translation for lattice cell (i, j)."""
        return i * self.a1 + j * self.a2

    def neighbor_offsets(self, ring: int = 1):
        r = range(-ring, ring + 1)
        return [(i, j) for i in r for j in r]