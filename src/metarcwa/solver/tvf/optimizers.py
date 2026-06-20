# metarcwa/solver/tvf/optimizers.py
# Optimizer wrappers for TVF optimization

from typing import Callable
from abc import ABC, abstractmethod
import torch


# ----- Abstract base -----
class TVFOptimizer(ABC):
    @abstractmethod
    def minimize(self, params: torch.Tensor, loss_fn: Callable, steps: int) -> torch.Tensor:
        """
        Minimize loss_fn w.r.t. params in-place for the given number of steps.

        Parameters
        ----------
        params : torch.Tensor
            Optimization variable (leaf tensor with requires_grad=True).
        loss_fn : Callable
            Maps params -> scalar (or [B] batch) loss.
        steps : int
            Number of optimizer steps.

        Returns
        -------
        params : torch.Tensor
            The (updated) params tensor (same object).
        """
        ...


# ----- PyTorch LBFGS -----
class TorchLBFGS(TVFOptimizer):
    """
    TVF optimizer using PyTorch's L-BFGS algorithm.

    The optimizer is created **once** and reused across all ``steps``
    so that L-BFGS can accumulate curvature history.

    For batched inputs the loss function should return a tensor of shape [B];
    it is summed to a scalar before backward so all batch elements are
    optimized jointly.

    Parameters
    ----------
    lr : float
        Learning rate (step length). Default 1.0.
    max_iter : int
        Maximum number of L-BFGS iterations per step call. Default 20.
    tolerance_grad : float
        Stop if max-norm of gradient falls below this. Default 1e-8.
    tolerance_change : float
        Stop if absolute change in loss falls below this. Default 1e-8.
    line_search_fn : str or None
        Line search to use, e.g. ``"strong_wolfe"``. Default None.
    """

    def __init__(
        self,
        lr: float = 1.0,
        max_iter: int = 20,
        tolerance_grad: float = 1e-8,
        tolerance_change: float = 1e-8,
        line_search_fn=None,
    ):
        self.lr = lr
        self.max_iter = max_iter
        self.tolerance_grad = tolerance_grad
        self.tolerance_change = tolerance_change
        self.line_search_fn = line_search_fn

    def minimize(self, params: torch.Tensor, loss_fn: Callable, steps: int) -> torch.Tensor:
        """
        Run L-BFGS for ``steps`` optimizer steps, reusing the same optimizer
        instance so curvature history is preserved across steps.

        Parameters
        ----------
        params : torch.Tensor
            Leaf tensor with ``requires_grad=True``.
        loss_fn : Callable
            Maps params -> torch.Tensor of shape [] or [B].
        steps : int
            Number of optimizer .step() calls.

        Returns
        -------
        params : torch.Tensor
        """
        opt = torch.optim.LBFGS(
            [params],
            lr=self.lr,
            max_iter=self.max_iter,
            tolerance_grad=self.tolerance_grad,
            tolerance_change=self.tolerance_change,
            line_search_fn=self.line_search_fn,
        )

        def closure():
            opt.zero_grad()
            loss = loss_fn(params).sum()   # sum over batch dim if present
            loss.backward()
            return loss

        for _ in range(steps):
            opt.step(closure)

        return params


# ----- Factory -----
def make_optimizer(name: str, **kwargs) -> TVFOptimizer:
    """
    Create a TVFOptimizer by name.

    Parameters
    ----------
    name : str
        Optimizer name (case-insensitive). Supported: ``"lbfgs"``.
    **kwargs
        Forwarded to the optimizer constructor.

    Returns
    -------
    optimizer : TVFOptimizer
    """
    name = name.lower()
    if name == "lbfgs":
        return TorchLBFGS(**kwargs)
    raise ValueError(f"Unknown optimizer '{name}'. Supported: 'lbfgs'")
