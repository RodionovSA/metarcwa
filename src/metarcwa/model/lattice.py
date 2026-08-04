# metarcwa/model/lattice.py
# DESCRIPTION

import math
import torch
import torch.nn as nn

from .nn_helpers import register

def _scalar(name: str, value) -> float:
    """Coerce a constructor scalar to float, refusing grad-carrying tensors.

    ``rectangular``/``hexagonal`` derive a1/a2 from a scalar, and
    ``nn.Parameter`` must be a leaf tensor -- so a Parameter passed here
    can never stay connected to the stored lattice vectors regardless of
    how it's wrapped. Rather than silently building a disconnected
    Parameter (which optimizes nothing), reject it with a pointer to the
    construction path that does work.
    """
    if isinstance(value, torch.Tensor):
        if isinstance(value, nn.Parameter) or value.requires_grad:
            raise TypeError(
                f"{name} must be a plain scalar, not a Parameter or a "
                f"grad-carrying tensor: a1/a2 are derived from {name}, and "
                f"nn.Parameter must be a leaf tensor, so the connection to "
                f"{name} would be silently severed. For an optimizable "
                f"lattice, construct it directly: "
                f"Lattice(a1=nn.Parameter(...), a2=nn.Parameter(...))."
            )
        if value.numel() != 1:
            raise ValueError(f"{name} must be a scalar, got shape {tuple(value.shape)}")
    return float(value)


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
        if self.a1.shape != (2,) or self.a2.shape != (2,):
            raise ValueError(
                f"a1 and a2 must have shape [2], got {tuple(self.a1.shape)} "
                f"and {tuple(self.a2.shape)}"
            )
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
        px, py : float
            Period along x and y. For an optimizable lattice, construct
            it directly instead: ``Lattice(a1=nn.Parameter(...), a2=nn.Parameter(...))``.
        """
        px_v, py_v = _scalar("px", px), _scalar("py", py)
        a1 = torch.tensor([px_v, 0.0])
        a2 = torch.tensor([0.0, py_v])
        return cls(a1=a1, a2=a2)

    @classmethod
    def hexagonal(cls, a, *, orientation: str = "pointy") -> "Lattice":
        """Build a hexagonal (triangular) lattice from the lattice constant.

        Parameters
        ----------
        a : float
            Nearest-neighbour distance (lattice constant). For an
            optimizable lattice, construct it directly instead:
            ``Lattice(a1=nn.Parameter(...), a2=nn.Parameter(...))`` --
            note that optimizing a1/a2 independently does not preserve
            hexagonal symmetry.
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
        a_v = _scalar("a", a)
        if orientation == "pointy":
            a1 = torch.tensor([a_v, 0.0])
            a2 = torch.tensor([a_v / 2.0, a_v * math.sqrt(3) / 2.0])
        elif orientation == "flat":
            a1 = torch.tensor([a_v * math.sqrt(3) / 2.0, a_v / 2.0])
            a2 = torch.tensor([0.0, a_v])
        else:
            raise ValueError(f"orientation must be 'pointy' or 'flat', got {orientation!r}")
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