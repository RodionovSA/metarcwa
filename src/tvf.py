from src.backend import Backend
from typing import Any

class TVF2D:
    """
    Tangent Vector Field (TVF) layer for 2D periodic structures.

    Input:
        eps_or_s : Tensor of shape [B, H, W] or [B, 1, H, W]
            Scalar field describing geometry (e.g., real(eps), fill fraction, etc.)

    Output:
        t : Tensor of shape [B, 2, H, W]
            Tangent unit vector field (t_x, t_y) at each grid point.
            Periodic, smooth, differentiable w.r.t. input field.
    """

    def __init__(self, 
                 backend: Backend, 
                 alpha=20.0, 
                 beta=1.0, 
                 max_iters=50, 
                 eps=1e-8,
                 solver: str = 'jacobi'):
        """
        Parameters
        ----------
        backend : Backend
            Backend for tensor operations.
        alpha : float
            Weight for data fidelity term (matching t0 near edges).
        beta : float
            Weight for smoothness term.
        max_iters : int
            Max number of smoothing iterations.
        eps : float
            Small constant to avoid division by zero.
        solver : str
            Solver type ('jacobi' supported).
        """
        self.alpha = alpha      # data fidelity (match t0 near edges)
        self.beta = beta        # smoothness strength
        self.max_iters = max_iters
        self.eps = eps
        self.backend = backend
        self.solver = solver

    def _grad_periodic(self, s):
        """
        Compute gradient of scalar field s on a periodic grid.

        s: [B, 1, H, W]
        Returns:
            sx, sy: [B, 1, H, W]
        """
        # central differences with periodic boundary conditions
        sx = 0.5 * (self.backend.roll(s, shifts=-1, dims=-1) - self.backend.roll(s, shifts=1, dims=-1))
        sy = 0.5 * (self.backend.roll(s, shifts=-1, dims=-2) - self.backend.roll(s, shifts=1, dims=-2))
        return sx, sy

    def _laplacian_periodic(self, t):   
        """
        2D Laplacian with periodic BC for a vector field t.

        t: [B, C, H, W]
        Returns:
            lap_t: [B, C, H, W]
        """
        t_xp = self.backend.roll(t, shifts=1, dims=-1)
        t_xm = self.backend.roll(t, shifts=-1, dims=-1)
        t_yp = self.backend.roll(t, shifts=1, dims=-2)
        t_ym = self.backend.roll(t, shifts=-1, dims=-2)
        lap_t = (t_xp + t_xm + t_yp + t_ym - 4.0 * t)
        return lap_t
    
    def Jacobi(self,
               w: Any, 
               t0: Any,
               verbose: bool = False) -> Any:
        """
        Compute tangent tensor with Jacobi iteration approach.

        Parameters
        ----------
        w : Any
            Edge weight tensor.
        t0 : Any
            Initial tangent vector field.
        verbose : bool, default=False
            Whether to print residuals during iteration.
        Returns
        -------
        t_new : Any
            Updated tangent vector field after one Jacobi iteration.
        
        """
        t = self.backend.clone(t0)   # initialization 
        
        # Broadcast w to 2 channels
        w2 = self.backend.cat([w, w], dim=1)
        
        for _ in range(self.max_iters):
            lap_t = self._laplacian_periodic(t)       # [B,2,H,W]

            neighbors = lap_t + 4.0 * t               # [B, 2, H, W]
            
            # Numerator and denominator per pixel, broadcasting w over channel dim
            numer = self.alpha * w2 * t0 + self.beta * neighbors    # [B, 2, H, W]
            denom = self.alpha * w2 + 4.0 * self.beta  # 4 from 2D Laplacian stencil
            denom = denom + self.eps

            t = numer / denom

            if verbose:
                # b(x) = α w t0
                b = self.alpha * w2 * t0

                # A(t) = α w t - β Δ t
                A_t = self.alpha * w2 * t - self.beta * lap_t

                # residual r = A(t) - b
                r = A_t - b

                # L2 residual
                residual = self.backend.sqrt(self.backend.sum(r * r))

                # optional: relative residual
                bnorm = self.backend.sqrt(self.backend.sum(b * b)) + 1e-12
                residual_rel = residual / bnorm

                print(f"Jacobi iteration residual: abs={residual}, rel={residual_rel}")

        # Normalize to unit length
        t_norm = self.backend.sqrt(self.backend.sum(t * t, dim=1, keepdim=True) + self.eps)
        t_unit = t / t_norm

        return t_unit  # [B, 2, H, W]

    def compute(self, eps_or_s: Any, requires_grad: bool = False, verbose: bool = False) -> Any:
        """
        Compute tangent vector field.

        Parameters
        ----------
        eps_or_s : torch.Tensor
            Shape [B, H, W] or [B, 1, H, W].
            Can be epsilon, real(epsilon), or any scalar shape field.
        requires_grad : bool, default=False
            Whether to track gradients during computation.
        verbose : bool, default=False
            Whether to print debug information.

        Returns
        -------
        t : Any
            Shape [B, 2, H, W], tangent unit vectors.
        """
        s = self.backend.asarray(eps_or_s, complex=False)
        shape_s = self.backend.shape(s)
        
        if not requires_grad:
            s = self.backend.detach(s)

        # Ensure shape [B, 1, H, W]
        if len(shape_s) == 3:          # [B, H, W]
            s = self.backend.reshape(s, (shape_s[0], 1, shape_s[1], shape_s[2]))
        elif len(shape_s) != 4:
            raise ValueError("Input must have shape [B,H,W] or [B,1,H,W]")

        # Normalize s to [0,1] to have a scale-invariant gradient
        s_min = s.amin(dim=(-2, -1), keepdim=True)
        s_max = s.amax(dim=(-2, -1), keepdim=True)
        s = (s - s_min) / (s_max - s_min + self.eps)

        # ---- 1. Raw normal from gradient of s ----
        sx, sy = self._grad_periodic(s)                # [B,1,H,W] each
        grad_mag = self.backend.sqrt(sx * sx + sy * sy + self.eps)

        nx0 = sx / grad_mag
        ny0 = sy / grad_mag

        # ---- 2. Raw tangent t0 = (-ny, nx) ----
        tx0 = -ny0
        ty0 = nx0
        t0 = self.backend.cat([tx0, ty0], dim=1)              # [B,2,H,W]

        # ---- 3. Edge weight w: large near interfaces ----
        edge_strength = grad_mag                      # [B,1,H,W]
        # Normalize edge_strength to [0,1]
        edge_norm = edge_strength / (edge_strength.amax(dim=(-2, -1), keepdim=True) + self.eps)
        w = edge_norm  # can also use w = edge_norm**γ to sharpen

        # ---- 4. Solve approx (α W - β Δ) t = α W t0 ----
        if self.solver == 'jacobi':
            t_unit = self.Jacobi(w, t0, verbose=verbose)  # [B,2,H,W]
        else:
            raise ValueError(f"Unknown solver: {self.solver}")

        return t_unit  # [B, 2, H, W]