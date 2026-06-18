# metarcwa/solver/utils.py
# Description

import torch
from typing import Literal

def matrix_solve(
    A: torch.Tensor,
    B: torch.Tensor,
    method: Literal["solve", "inv"] = "solve",
) -> torch.Tensor:
    """Solve A X = B for X (i.e. X = A⁻¹ B).

    Args:
        A: [..., n, n] batched square matrix.
        B: [..., n, k] right-hand side (or [..., n, n] for full inverse via B=I).
        method: "solve" uses torch.linalg.solve (preferred — no explicit inverse);
                "inv" forms torch.linalg.inv(A) @ B (slower, less stable, worse backward).

    Returns:
        X: [..., n, k].
    """
    pass