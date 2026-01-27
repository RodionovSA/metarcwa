from typing import Any, Callable, Tuple
from abc import ABC, abstractmethod
import torch

from src.backend import Backend

# ----- Optimizers -----
class TVFOptimizer(ABC):
    @abstractmethod
    def step(self, params: Any, loss_fn: Callable): ...

#----- PyTorch LBFGS Optimizer -----
class TorchLBFGS(TVFOptimizer):
    """
    TVF optimizer using PyTorch's LBFGS algorithm.
    In the case of batched inputs, the loss function should return a tensor of shape [B],
    but the optimizer will sum over the batch dimension to perform a single optimization step.
    """
    def __init__(self, 
                 lr=1.0, 
                 max_iter=20, 
                 tolerance_grad=1e-8, 
                 tolerance_change=1e-8, 
                 line_search_fn=None):
        """
        Parameters:
            lr: float
                Learning rate.
            max_iter: int
                Maximum number of iterations per optimization step.
            tolerance_grad: float
                Tolerance for gradient norm.
            tolerance_change: float
                Tolerance for change in loss value.
            line_search_fn: str
                Line search function to use.
        """
        self.lr = lr
        self.max_iter = max_iter
        self.tolerance_grad = tolerance_grad
        self.tolerance_change = tolerance_change
        self.line_search_fn = line_search_fn

    def step(self, params: Any, loss_fn: Callable) -> Any:
        opt = torch.optim.LBFGS([params], 
                                lr=self.lr, 
                                max_iter=self.max_iter,
                                tolerance_grad=self.tolerance_grad,
                                tolerance_change=self.tolerance_change,
                                line_search_fn=self.line_search_fn)

        def closure():
            opt.zero_grad()
            loss_batched = loss_fn(params)      # shape [B]
            loss = loss_batched.sum()            # scalar
            loss.backward()
            return loss

        opt.step(closure)
        return params
    
# ----- Optimizers factory -----
def make_optimizer(backend: Backend, name: str, **kwargs) -> TVFOptimizer:
    name = name.lower()

    if backend.name == "torch":
        if name == "lbfgs":
            return TorchLBFGS(**kwargs)
        else:
            raise ValueError(f"Unknown optimizer '{name}' for Torch backend")

    elif backend.name == "jax":
        raise NotImplementedError("JAX optimizers are not implemented yet")

    elif backend.name == "numpy":
        raise NotImplementedError("Numpy optimizers are not implemented yet")

    else:
        raise ValueError(f"Unsupported backend '{backend.name}'")