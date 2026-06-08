# metarcwa/model/lattice.py
# DESCRIPTION

import math
import torch
import torch.nn as nn

class Lattice(nn.Module):
    """In-plane periodicity of the unit cell. Fixed (non-optimizable).
    Defined by two lattice vectors. A rectangular lattice is the special
    case of axis-aligned vectors.

    Parameters
    ----------
    a1 : Tensor
        First lattice vector, shape [2].
    a2 : Tensor
        Second lattice vector, shape [2].
    """
    def __init__(self, a1, a2):
        super().__init__()
        a1 = torch.as_tensor(a1, dtype=torch.float32)
        a2 = torch.as_tensor(a2, dtype=torch.float32)
        if torch.linalg.det(torch.stack([a1, a2], 1)).abs() <= 1e-12:
            raise ValueError("a1, a2 must be linearly independent")
        self.register_buffer("a1", a1)
        self.register_buffer("a2", a2)
        
    @property
    def device(self): return self.a1.device
    @property
    def dtype(self):  return self.a1.dtype

    @classmethod
    def rectangular(cls, px: float, py: float) -> "Lattice":
        """Build an axis-aligned rectangular lattice from two periods."""
        return cls(
            a1=torch.tensor([px, 0.0]),
            a2=torch.tensor([0.0, py]),
        )

    @classmethod
    def hexagonal(cls, a: float, *, orientation: str = "pointy") -> "Lattice":
        """Build a hexagonal (triangular) lattice from the lattice constant.

        Parameters
        ----------
        a : float
            Nearest-neighbour distance (lattice constant).
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
        if orientation == "pointy":
            return cls(
                a1=torch.tensor([a, 0.0]),
                a2=torch.tensor([a / 2.0, a * math.sqrt(3) / 2.0]),
            )
        elif orientation == "flat":
            return cls(
                a1=torch.tensor([a * math.sqrt(3) / 2.0, a / 2.0]),
                a2=torch.tensor([0.0, a]),
            )
        else:
            raise ValueError(f"orientation must be 'pointy' or 'flat', got {orientation!r}")

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