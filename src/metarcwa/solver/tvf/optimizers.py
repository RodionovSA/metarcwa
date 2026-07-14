# metarcwa/solver/tvf/optimizers.py
# Optimizer wrappers for TVF optimization

import warnings
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
    chunk_size : int or None
        Number of Hessian columns assembled per ``vmap`` pass in
        :meth:`minimize` (see there for the memory/speed tradeoff).
        ``None`` (default) assembles all columns in a single pass — the
        original behavior, fastest but with peak memory scaling as
        ``flat = numel(params[0])``. A positive int caps memory at
        roughly ``flat / chunk_size`` of that, at some runtime cost.
    """

    def __init__(self, regularization: float = 1e-12, steps: int = 1,
                 chunk_size: int | None = None):
        self.regularization = regularization
        self.steps = steps
        self.chunk_size = chunk_size

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
            # chunk_size caps how many basis columns are evaluated in one
            # vmap pass — each column's hvp_col forward+tangent pass touches
            # the full [B, D0, D1, 2] grid (via loss_fn/total_loss), so an
            # unchunked vmap over all `flat` columns at once is the dominant
            # peak-memory cost of TVF. Chunking is mathematically identical
            # (same H, same solve), it only trades some speed for memory.
            cols = vmap(hvp_col, chunk_size=self.chunk_size)(basis)

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


# ----- Matrix-free Newton-CG solve -----
class NewtonCG(TVFOptimizer):
    """
    Matrix-free Newton optimizer for the TVF quadratic loss, using conjugate
    gradients (CG) to solve ``H·Δx = g`` from Hessian-vector products alone.

    Like :class:`NewtonExact`, this relies on the TVF loss being a *real*
    quadratic function of the Fourier coefficients, so solving the Newton
    system gives the exact global minimum (up to CG convergence tolerance).
    Unlike :class:`NewtonExact`, it never materializes the dense
    ``[B, flat, flat]`` Hessian or runs an ``O(flat**3)`` batched solve —
    each CG iteration costs one Hessian-vector product (one ``jvp`` through
    the loss), so peak memory is ``O(flat + grid)`` instead of ``O(flat**2)``.
    This is the optimizer to use at high harmonic truncation
    (``flat = (2m+1)(2n+1)*4``), where :class:`NewtonExact`'s dense Hessian
    and cubic solve become the bottleneck even with chunking.

    Each batch element is solved independently but the CG iteration is fully
    vectorized across the batch (per-element dot products / stopping
    criteria), so this is one Python loop over CG iterations regardless of
    batch size or parameter count.

    Parameters
    ----------
    regularization : float
        Diagonal regularization added to the Hessian-vector product before
        the solve (same role as ``NewtonExact.regularization``). Default
        ``1e-12``.
    max_iter : int or None
        Maximum CG iterations. ``None`` (default) uses ``2 * flat`` — for an
        exact quadratic, CG should converge within ``flat`` iterations in
        exact arithmetic; the factor of 2 gives headroom for floating-point
        loss of conjugacy on ill-conditioned Hessians.
    tol : float
        Relative-residual stopping tolerance:
        ``||r|| <= tol * ||g||`` per batch element (all elements must meet
        this to stop early). Default ``1e-8``. Use a tighter tolerance in
        ``float64`` for high-accuracy runs; ``float32`` will typically not
        reach residuals much below ``~1e-6`` regardless of ``tol``.
    steps : int
        Number of Newton steps. Default 1 (exact for quadratic losses).
    """

    def __init__(self, regularization: float = 1e-12, max_iter: int | None = None,
                 tol: float = 1e-8, steps: int = 1):
        self.regularization = regularization
        self.max_iter = max_iter
        self.tol = tol
        self.steps = steps

    def minimize(self, params: torch.Tensor, loss_fn: Callable, steps: int) -> torch.Tensor:
        """
        Run ``steps`` Newton iterations, each solving ``H·Δx = g`` via
        matrix-free batched CG (see class docstring).

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
        max_iter = 2 * flat if self.max_iter is None else self.max_iter
        dims = tuple(range(1, len(shape_per) + 1))   # per-element reduction dims

        def scalar_loss(p: torch.Tensor) -> torch.Tensor:
            return loss_fn(p).sum()

        grad_fn = func_grad(scalar_loss)

        def dot(u: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
            return (u * w).sum(dim=dims)   # [B]

        def bview(v: torch.Tensor) -> torch.Tensor:
            return v.reshape(B, *([1] * len(shape_per)))

        for _ in range(steps):
            x = params.detach()        # [B, *shape_per], pure functional primal
            g = grad_fn(x)              # [B, *shape_per]

            def hvp(v: torch.Tensor) -> torch.Tensor:
                return jvp(grad_fn, (x,), (v,))[1] + self.regularization * v

            # ── Batched conjugate gradients: solve H·delta = g, x0 = 0 ───────
            delta = torch.zeros_like(g)
            r = g.clone()
            p = r.clone()
            rs_old = dot(r, r)                          # [B]
            g_norm = dot(g, g).sqrt().clamp_min(1e-300)  # [B]

            for _n_iter in range(max_iter):
                Hp = hvp(p)
                pHp = dot(p, Hp).clamp_min(1e-300)
                alpha = bview(rs_old / pHp)
                delta = delta + alpha * p
                r = r - alpha * Hp
                rs_new = dot(r, r)
                converged = bool((rs_new.sqrt() <= self.tol * g_norm).all())
                if converged:
                    # Update rs_old to the just-computed (lower) residual before
                    # breaking, so the post-loop report/warning reflects the
                    # actual converged value rather than the prior iteration's
                    # (stale, larger) one.
                    rs_old = rs_new
                    break
                beta = bview(rs_new / rs_old.clamp_min(1e-300))
                p = r + beta * p
                rs_old = rs_new

            residual = (rs_old.sqrt() / g_norm).max().item()
            if residual > self.tol:
                warnings.warn(
                    f"NewtonCG: did not converge within {max_iter} CG iterations "
                    f"(worst-case relative residual {residual:.3e} > tol {self.tol:.1e}). "
                    "Consider raising max_iter or newton_cg_tol tolerance, or switch to "
                    "'newton' for this harmonic count.",
                    RuntimeWarning,
                )

            with torch.no_grad():
                params -= delta

        return params


# ----- Factory -----
def make_optimizer(name: str, **kwargs) -> TVFOptimizer:
    """
    Create a TVFOptimizer by name.

    Parameters
    ----------
    name : str
        Optimizer name (case-insensitive). Supported: ``"lbfgs"``, ``"newton"``
        (:class:`NewtonExact`), ``"newton_cg"`` (:class:`NewtonCG`, matrix-free,
        for high harmonic truncation).
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
    if name in ("newton_cg", "newtoncg"):
        return NewtonCG(**kwargs)
    raise ValueError(
        f"Unknown optimizer '{name}'. Supported: 'lbfgs', 'newton', 'newton_cg'"
    )
