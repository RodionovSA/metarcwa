# metarcwa/model/lattice.py
# DESCRIPTION

from dataclasses import dataclass
import torch

@dataclass(frozen=True)
class Lattice:
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

    a1: torch.Tensor
    a2: torch.Tensor

    @classmethod
    def rectangular(cls, px: float, py: float) -> "Lattice":
        """Build an axis-aligned rectangular lattice from two periods."""
        return cls(
            a1=torch.tensor([px, 0.0]),
            a2=torch.tensor([0.0, py]),
        )