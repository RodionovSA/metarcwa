# metarcwa/solver/tvf/optimizers.py
# Optimizer wrappers for TVF optimization

from typing import Callable
from abc import ABC, abstractmethod
import torch
from torch.func import grad as func_grad, jvp, vmap


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
        Line search to use.  Default ``"strong_wolfe"``.

        Without a line search (``None``), L-BFGS uses a fixed step of size
        ``lr = 1.0``.  The optimization variable is the raw in-band FFT
        coefficients, whose magnitudes are O(D²) = O(10⁴) for a D×D grid
        while the loss is O(1).  The resulting gradient is ~1e-4, so the
        fixed step of 1.0 moves parameters by only ~1e-4 per outer iteration
        — far too small relative to the coefficient scale — causing the
        optimizer to freeze (output equals the initial field).  The Wolfe
        conditions in ``"strong_wolfe"`` adapt the step to local curvature,
        making convergence independent of the absolute loss scale and producing
        the same result as Newton for any (alpha, beta, gamma) with the same
        ratio.
    """

    def __init__(
        self,
        lr: float = 1.0,
        max_iter: int = 20,
        tolerance_grad: float = 1e-8,
        tolerance_change: float = 1e-8,
        line_search_fn: str = "strong_wolfe",
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


# ----- Exact Newton solve -----
class NewtonExact(TVFOptimizer):
    """
    Exact one-step Newton optimizer for the TVF quadratic loss.

    The TVF alignment + Fourier-regularization + smoothness loss is a *real*
    quadratic function of the Fourier coefficients (real/imag parts), so a
    single Newton step (solve H·Δx = g) gives the exact global minimum.

    Each batch element is solved independently.  For robustness the Hessian
    is regularised with a small diagonal shift before the solve.

    Parameters
    ----------
    regularization : float
        Diagonal regularization added to H before the solve. Default 1e-12.
    steps : int
        Number of Newton steps.  Default 1 (exact for quadratic losses).
    """

    def __init__(self, regularization: float = 1e-12, steps: int = 1):
        self.regularization = regularization
        self.steps = steps

    def minimize(self, params: torch.Tensor, loss_fn: Callable, steps: int) -> torch.Tensor:
        """
        Run ``steps`` exact Newton iterations.

        The loss is a real quadratic that is decoupled across the batch, so the
        Hessian is block-diagonal with one ``[flat, flat]`` block per batch
        element. This allows a fully vectorized solve:

        * A single backward pass computes all per-sample gradients.
        * ``vmap`` over ``flat`` JVP calls assembles all Hessian blocks at once.
        * A single batched ``torch.linalg.solve`` replaces the per-sample loop.

        No Python loops are needed, regardless of batch size or parameter count.

        Parameters
        ----------
        params : torch.Tensor
            Leaf tensor with ``requires_grad=True``.  Shape ``[B, ...]``.
        loss_fn : Callable
            Maps params -> torch.Tensor of shape ``[B]``.
        steps : int
            Number of Newton iterations.

        Returns
        -------
        params : torch.Tensor
            Updated params (same object, data updated via no_grad).
        """
        B = params.shape[0]
        shape_per = params.shape[1:]   # shape of one batch element
        flat = params[0].numel()

        # Sum over batch so grad() returns a [B, *shape_per] tensor —
        # valid because the loss is decoupled across b.
        def scalar_loss(p: torch.Tensor) -> torch.Tensor:
            return loss_fn(p).sum()

        grad_fn = func_grad(scalar_loss)

        for _ in range(steps):
            x = params.detach()        # [B, *shape_per], pure functional primal

            # ── Gradient: one backward pass for the whole batch ──────────────
            g = grad_fn(x)             # [B, *shape_per]

            # ── Hessian columns via vmapped JVP ──────────────────────────────
            # basis[k] is the k-th standard basis vector reshaped to shape_per.
            basis = torch.eye(flat, dtype=x.dtype, device=x.device).reshape(
                flat, *shape_per
            )                          # [flat, *shape_per]

            # hvp_col(v): tangent v has shape [*shape_per]; broadcast to
            # [B, *shape_per] so the JVP hits every batch element at once.
            # Returns jvp output shape [B, *shape_per] = H_b @ v for each b.
            def hvp_col(v: torch.Tensor) -> torch.Tensor:
                v_batch = v.unsqueeze(0).expand(B, *shape_per)
                return jvp(grad_fn, (x,), (v_batch,))[1]

            # cols[k, b, ...] = k-th column of H_b  → shape [flat, B, *shape_per]
            cols = vmap(hvp_col)(basis)

            # Reshape to [B, flat, flat]: H[b, j, k] = cols[k, b, j]
            H = cols.reshape(flat, B, flat).permute(1, 2, 0)   # [B, flat, flat]

            # ── Regularize and solve H Δx = g for the whole batch ────────────
            H_reg = H + self.regularization * torch.eye(
                flat, dtype=H.dtype, device=H.device
            )                          # [B, flat, flat] (eye broadcasts over batch)
            delta = torch.linalg.solve(H_reg, g.reshape(B, flat))  # [B, flat]

            with torch.no_grad():
                params -= delta.reshape(B, *shape_per)

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
    if name in ("newton", "newtonexact"):
        return NewtonExact(**kwargs)
    raise ValueError(f"Unknown optimizer '{name}'. Supported: 'lbfgs', 'newton'")
