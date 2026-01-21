from src.backend import Backend
from src.model.geometry.geometry import Lattice

from typing import Any, Callable, Tuple
from abc import ABC, abstractmethod
import torch

class TVF:
    """
    Class for Tangent Vector Field (TVF) computation.
    Supports the following methods:
    - "Jones": Antos 2009 https://doi.org/10.1364/OE.17.007269.
    - "Pol": S4 https://doi.org/10.1016/j.cpc.2012.04.026.
    - "Normal": Schuster 2007 https://doi.org/10.1364/JOSAA.24.002880.
    - "Jones_direct": FMMax https://doi.org/10.1364/OE.503481.
    
    The approach was inspired by the implementations in S4 and FMMax.
    """
    def __init__(self, 
                 backend: Backend,
                 lattice: Lattice,
                 M: int,
                 N: int, 
                 method: str, 
                 optimizer: str = "LBFGS"):
        """
        Parameters:
            backend: Backend
                Computational backend (e.g., PyTorch, JAX, NumPy).
            lattice: Lattice
                Lattice object defining the grid and spacing.
            M: int
                Truncation order in x direction for Fourier filtering.
            N: int
                Truncation order in y direction for Fourier filtering.
            method: str
                Method for TVF computation (Jones, Pol, Normal, Jones_direct).
            optimizer: str
                Optimizer to use for TVF optimization (default: "LBFGS").
        """
        self._backend = backend
        self._lattice = lattice
        self._M = M
        self._N = N
        self._method = method
        if method not in ["Jones", "Pol", "Normal", "Jones_direct"]:
            raise ValueError(f"Unknown method '{method}' for TVF computation")
        self._optimizer = make_optimizer(backend, optimizer)
    
    @property
    def backend(self):
        return self._backend
    @property
    def lattice(self):
        return self._lattice
    @property
    def M(self):
        return self._M
    @property
    def N(self):
        return self._N
    @property
    def method(self):
        return self._method
    @property
    def optimizer(self):
        return self._optimizer
        
    def _prepare_field(self, field: Any) -> Tuple[Any, Any, Any]:
        """
        Prepare the vector field before processing.
        
        Parameters:
            field: Any
                Input scalar field. Shape: [B, Nx, Ny].
        
        Returns:
            target_field: Any
                Prepared target vector field. Shape: [B, Nx, Ny, 2].
            initial_field_fft: Any
                Initial vector field in Fourier domain for optimization. Shape: [B, Nx, Ny, 2].
            weights: Any
                Weights for alignment loss. Shape: [B, Nx, Ny, 1].
        """
        # Step 1: Compute gradients assuming periodic boundaries
        gradx, grady = _grad_periodic(self.backend, self.lattice, field)  # [B, Nx, Ny]
        
        # Step 2: Filter and resample gradients ((4M+1, 4N+1) window)
        gradx_f, grady_f = low_pass_filter(self.backend, (gradx, grady), M=2*self.M, N=2*self.N)  # [B, Nx, Ny]
        
        # Step 3: Global normalization
        grad_n = normalize_max_global(self.backend, self.backend.stack((gradx_f, grady_f), dim=-1))
        gradx_n, grady_n = grad_n[...,0], grad_n[...,1]  # [B, Nx, Ny]
        
        # Step 4: Define target field
        target_field = self.backend.stack((grady_n, -gradx_n), dim=-1)  # [B, Nx, Ny, 2]
        
        # Step 5: Normalize elementwise
        target_field = normalize_elementwise(self.backend, target_field)  # [B, Nx, Ny, 2]
        
        # Step 6: Define initial field
        if self.method == "Jones_direct":
            target_field = normalize_jones(self.backend, target_field)  # [B, Nx, Ny, 2]
            initial_field = target_field
        else:
            initial_field = self.backend.stack((grady_n, -gradx_n), dim=-1)  # [B, Nx, Ny, 2]
        
        # Step 7: Shift initial field to fourier domain
        initial_field_fft = self.backend.fftshift(self.backend.fft2(initial_field, dim=(-3, -2)), dim=(-3, -2))  # [B, Nx, Ny, 2]
            
        # Step 8: Compute alignment loss weights
        weights = _field_magnitude(self.backend, self.backend.stack((gradx_n, grady_n), dim=-1))  # [B, Nx, Ny, 1]
            
        return target_field, initial_field_fft, weights
    
    def _optimize(self, 
                  field: Any, 
                  alpha: float = 1.0, 
                  beta: float = 1e-8, 
                  gamma: float = 1.0,
                  steps: int = 1) -> Any:
        """
        Optimize the Tangent Vector Field (TVF) from the input scalar field.
        Optimization works on geometry-based TVF derived from Re(field) in Fourier domain.
        The resulting TVF does not keep computational graph from field.

        Parameters:
            field: Any
                Input scalar field. Shape: [B, Nx, Ny].
            alpha: float
                Weight for alignment loss.
            beta: float
                Weight for fourier regularization loss.
            gamma: float
                Weight for smoothness loss.
            steps: int
                Number of optimization steps.
                
        Returns:
            optimized_field: Any
                Computed TVF. Shape: [B, Nx, Ny, 2].
        """
        # Sanity check
        if len(field.shape) != 3:
            raise ValueError("Input field must be a 3D tensor with shape [B, Nx, Ny].")
        
        # Clone and detach grads
        field = self.backend.real(self.backend.detach(field))
        
        # Prepare fields
        target_field, initial_field_fft, weights = self._prepare_field(field)
        initial_field = self.backend.stack([self.backend.real(initial_field_fft), 
                                            self.backend.imag(initial_field_fft)], dim=-1) #Split real and imag parts
        
        def loss_fn(current_field: Any) -> Any:
            return total_loss(self.backend, 
                              current_field, 
                              target_field, 
                              weights=weights, 
                              lattice=self.lattice,
                              alpha=alpha,
                              beta=beta,
                              gamma=gamma)
        
        optimized_field = self.backend.requires_grad(self.backend.detach(initial_field), set=True)
        for _ in range(steps):
            optimized_field = self.optimizer.step(optimized_field, loss_fn)
        
        optimized_field = self.backend.detach(optimized_field) # Detach from computation graph
        optimized_field = optimized_field[...,0] + 1j * optimized_field[...,1]  # Combine real and imag parts
        optimized_field = self.backend.ifft2(self.backend.ifftshift(optimized_field, dim=(-3, -2)), dim=(-3, -2))  # Back to spatial domain
        
        return optimized_field
    
    def compute(self, 
                field: Any, 
                alpha: float = 1.0, 
                beta: float = 1e-8, 
                gamma: float = 1.0,
                steps: int = 1) -> Tuple[Any, Any]:
        """
        Compute the Tangent Vector Field (TVF) from the input scalar field.
        Parameters:
            field: Any
                Input scalar field. Shape: [B, Nx, Ny].
            alpha: float
                Weight for alignment loss.
            beta: float
                Weight for fourier regularization loss.
            gamma: float
                Weight for smoothness loss.
            steps: int
                Number of optimization steps.
        Returns:
            Tx: Any
                Tangent vector field x-component. Shape: [B, Nx, Ny].
            Ty: Any
                Tangent vector field y-component. Shape: [B, Nx, Ny].
            
        """
        optimized_field = normalize_max_global(self.backend, self._optimize(field, alpha, beta, gamma, steps))
        
        if self.method == "Jones":
            # Transform to Jones field
            optimized_field = normalize_jones(self.backend, optimized_field)
        
        elif self.method == "Pol":
            # Global normalization
            optimized_field = normalize_max_global(self.backend, self.backend.real(optimized_field))
            
        elif self.method == "Normal":
            # Elementwise normalization
            optimized_field = normalize_elementwise(self.backend, self.backend.real(optimized_field))
            
        elif self.method == "Jones_direct":
            # Already Jones-normalized
            pass 

        return optimized_field[...,0], optimized_field[...,1]  # Return Tx, Ty components

""" Main functions for TVF operations """

# ----- Gradient computation for 2D scalar fields -----
def _grad_periodic(backend: Backend, lattice: Lattice, s: Any) -> tuple[Any, Any]:
    """
    Compute gradient of scalar field s.
    Use central differences with periodic boundary conditions.
    Normalize by lattice spacing to avoid incorrect gradient magnitude in anisotropic lattices.

    Parameters:
        backend: Backend
            Computational backend.
        lattice: Lattice
            Lattice object defining the grid and spacing.
        s: input scalar field
            shape: [B, Nx, Ny]
    Returns:
        sx, sy: [B, Nx, Ny]
    """
    # central differences with periodic boundary conditions
    gradx = 0.5 * (backend.roll(s, shifts=-1, dims=-2) - backend.roll(s, shifts=1, dims=-2))
    grady = 0.5 * (backend.roll(s, shifts=-1, dims=-1) - backend.roll(s, shifts=1, dims=-1))
    
    # Normalize by lattice spacing
    delta_x, delta_y = lattice.delta
    
    gradx = gradx / delta_x
    grady = grady / delta_y
    return gradx, grady

# ----- Low-pass filtering functions for 2D vector fields -----
def low_pass_mask(backend: Backend, Nx: int, Ny: int, M: int, N: int) -> Any:
    """
    Create a low-pass filter mask in Fourier domain.

    Parameters:
        backend: Backend
            Computational backend.
        Nx: int
            Number of grid points in x direction.
        Ny: int
            Number of grid points in y direction.
        M: int
            Truncation order in x direction.
        N: int
            Truncation order in y direction.
    Returns:
        mask: [Nx, Ny], with ones in low-pass region and zeros elsewhere.
    """
    kx = backend.arange(0, Nx) - Nx//2
    ky = backend.arange(0, Ny) - Ny//2
    kx, ky = backend.meshgrid(kx, ky, indexing="ij")

    mask = (backend.abs(kx) <= M) & (backend.abs(ky) <= N)
    return mask

def low_pass_filter(backend: Backend, grad: tuple[Any, Any], M: int, N: int) -> Any:
    """
    Low-pass filter 2D data in Fourier domain.

    Parameters:
        backend: Backend
            Computational backend.
        grad: tuple[Any, Any]
            Gradients to be filtered. Each of shape [B, Nx, Ny].
        M: int
            Truncation order in x direction.
        N: int
            Truncation order in y direction.
    Returns:
        filtered_data: [B, Nx, Ny], but only low-pass components are non-zero.
    """
    # Sanity check
    if grad is None or len(grad) != 2:
        raise ValueError("Gradient must be a tuple of two components.")
    if grad[0].shape != grad[1].shape:
        raise ValueError("Gradient components must have the same shape.")
    
    gradx, grady = grad[0], grad[1]
    
    _, Nx, Ny = gradx.shape
    gradx_fft = backend.fft2(gradx, dim=(-2, -1))  # [B, Nx, Ny]
    grady_fft = backend.fft2(grady, dim=(-2, -1))  # [B, Nx, Ny]
    
    # Shift zero frequency component to center
    gradx_fft_shifted = backend.fftshift(gradx_fft, dim=(-2, -1))
    grady_fft_shifted = backend.fftshift(grady_fft, dim=(-2, -1))
    
    mask = low_pass_mask(backend, Nx, Ny, M, N)  # [Nx, Ny]
    mask = backend.astype(mask, gradx.dtype)
    
    filteredx_fft_shifted = gradx_fft_shifted * mask  # [B, Nx, Ny]
    filteredy_fft_shifted = grady_fft_shifted * mask  # [B, Nx, Ny]
    
    # Shift back
    filteredx_fft = backend.ifftshift(filteredx_fft_shifted, dim=(-2, -1))
    filteredy_fft = backend.ifftshift(filteredy_fft_shifted, dim=(-2, -1))
    
    # Inverse FFT to get filtered data
    filteredx = backend.ifft2(filteredx_fft)  # [B, Nx, Ny]
    filteredy = backend.ifft2(filteredy_fft)  # [B, Nx, Ny]
    
    return (filteredx, filteredy)

# ----- Normalization functions for 2D vector fields -----
def _field_magnitude(backend: Backend, field: Any) -> Any:
    """
    Compute the magnitude of a 2D vector field.
    Safely handles sqrt for autograd.

    Parameters:
        backend: Backend
            Computational backend.
        field: Any
            Vector field. Shape: [B, Nx, Ny, 2].
    Returns:
        magnitude: Any
            Magnitude of the vector field. Shape: [B, Nx, Ny, 1].
    """
    # compute magnitude squared
    mag_sq = backend.sum(backend.abs(field) ** 2, dim=-1, keepdim=True)
    
    # avoid NaNs for grad at zero field
    is_zero = mag_sq == 0
    mag_sq_safe = backend.where(is_zero, backend.ones_like(mag_sq), mag_sq) # replace zeros with ones for safe sqrt
    mag = backend.where(is_zero, backend.zeros_like(mag_sq), backend.sqrt(mag_sq_safe)) # calculate magnitude
    
    return mag

def normalize_max_global(backend: Backend, field: Any) -> Any:
    """
    Normalize a 2D vector field by its global maximum magnitude.
    This is a global normalization (not elementwise), numerically safe.

    Parameters:
        backend: Backend
            Computational backend.
        vector_field: Any
            Vector field to be normalized. Shape: [B, Nx, Ny, 2].
    Returns:
        normalized_vector_field: Any
            Normalized vector field. Shape: [B, Nx, Ny, 2].
    """
    mag = _field_magnitude(backend, field)  # [B, Nx, Ny, 1]
    
    # global maximum magnitude
    max_mag = backend.amax(mag, dim=(-3, -2), keepdim=True)
    
    # avoid division by zero
    max_mag_safe = backend.where(max_mag == 0, 1, max_mag)

    return field / max_mag_safe

def normalize_elementwise(backend: Backend, field: Any) -> Any:
    """
    Normalize a 2D vector field so that each element's magnitude is 1.
    This is an elementwise normalization, numerically safe.

    Parameters:
        backend: Backend
            Computational backend.
        vector_field: Any
            Vector field to be normalized. Shape: [B, Nx, Ny, 2].
    Returns:
        normalized_vector_field: Any
            Normalized vector field. Shape: [B, Nx, Ny, 2].
    """
    # compute magnitude squared
    mag_sq = backend.sum(backend.abs(field) ** 2, dim=-1, keepdim=True)
    
    # avoid NaNs for zero field
    is_zero = mag_sq == 0
    mag_sq_safe = backend.where(is_zero, backend.ones_like(mag_sq), mag_sq) # replace zeros with ones for safe sqrt
    mag = backend.sqrt(mag_sq_safe) # calculate magnitude

    return field / mag

# ----- Loss functions for 2D vector fields -----
def alignment_loss(backend: Backend, field: Any, target_field: Any, weights: Any) -> Any:
    """
    Compute alignment loss between field and target_field, weighted by weights.
    Loss = mean(weights * |field - target_field|^2)
    
    Parameters:
        backend: Backend
            Computational backend.
        field: Any
            Current vector field. Shape: [B, Nx, Ny, 2].
        target_field: Any
            Target vector field. Shape: [B, Nx, Ny, 2].
        weights: Any
            Weights for each element. Shape: [B, Nx, Ny, 1] or [B, Nx, Ny].
    
    Returns:
        loss: Any
            Alignment loss. Shape: [B].
    """
    if field.shape != target_field.shape:
        raise ValueError("field and target_field must have the same shape.")
    if weights.shape != field.shape[:-1] and weights.shape != field.shape[:-1] + (1,):
        raise ValueError("weights must have shape [B, Nx, Ny] or [B, Nx, Ny, 1].")
    
    diff = field - target_field  # [B, Nx, Ny, 2]
    diff_sq = backend.sum(backend.abs(diff) ** 2, dim=-1, keepdim=True)  # [B, Nx, Ny, 1]
    weighted_diff_sq = weights * diff_sq  # [B, Nx, Ny, 1] or [B, Nx, Ny]
    
    if weights.shape != field.shape[:-1]:
        weighted_diff_sq = backend.squeeze(weighted_diff_sq, dim=-1)  # [B, Nx, Ny]
        
    loss = backend.mean(weighted_diff_sq, dim=(-1, -2)) # [B]
    return loss

def fourier_regularization_loss(backend: Backend, field: Any, period: tuple[float, float]) -> Any:
    """
    Compute Fourier regularization loss for a 2D vector field.
    Loss = mean(|k|^2 * |F(field)|^2), where F is the 2D FFT.

    Parameters:
        backend: Backend
            Computational backend.
        field: Any
            Vector field. Shape: [B, Nx, Ny, 2].
        period: tuple[float, float]
            Physical period of the lattice in x and y directions.
    """
    Lx, Ly = period
    
    # Calculate |k|^2
    _, Nx, Ny, _ = field.shape
    kx = 2* backend.pi / Lx * (backend.arange(0, Nx) - Nx//2)
    ky = 2* backend.pi / Ly * (backend.arange(0, Ny) - Ny//2)
    Kx, Ky = backend.meshgrid(kx, ky, indexing="ij")
    K_norm2 = Lx*Ly*(backend.abs(Kx) ** 2 + backend.abs(Ky) ** 2)  # [Nx, Ny]
    
    loss = backend.mean(K_norm2 * backend.sum(backend.abs(field) ** 2, dim=-1), dim=(-2, -1))  # [B]
    return loss

def smoothness_loss(backend: Backend, lattice: Lattice, field: Any) -> Any:
    """
    Compute smoothness loss for a 2D vector field.
    Loss = mean(|grad(field)|^2)

    Parameters:
        backend: Backend
            Computational backend.
        field: Any
            Vector field. Shape: [B, Nx, Ny, 2].
            
    Returns:
        loss: Any
            Smoothness loss. Shape: [B].
    """
    field_x = field[:,:,:,0]  # [B, Nx, Ny]
    field_y = field[:,:,:,1]  # [B, Nx, Ny]
    
    gradx_x, grady_x = _grad_periodic(backend, lattice, field_x)  # [B, Nx, Ny]
    gradx_y, grady_y = _grad_periodic(backend, lattice, field_y)  # [B, Nx, Ny]
    
    grad_sq = backend.abs(gradx_x) ** 2 + backend.abs(grady_x) ** 2 + backend.abs(gradx_y) ** 2 + backend.abs(grady_y) ** 2  # [B, Nx, Ny]
    
    loss = backend.mean(grad_sq, dim=(-1, -2))  # [B]
    return loss

def total_loss(backend: Backend, 
               field: Any, 
               target_field: Any, 
               weights: Any, 
               lattice: Lattice,
               alpha: float = 1.0,
               beta: float = 1e-3,
               gamma: float = 1.0) -> Any:
    """
    Compute total loss as a weighted sum of alignment loss, fourier regularization loss, and smoothness loss.
    
    Parameters:
        backend: Backend
            Computational backend.
        field: Any
            Current vector field. Shape: [B, Nx, Ny, 2].
        target_field: Any
            Target vector field. Shape: [B, Nx, Ny, 2].
        weights: Any
            Weights for alignment loss. Shape: [B, Nx, Ny, 1] or [B, Nx, Ny].
        lattice: Lattice
            Lattice object defining the grid and spacing.
        alpha: float
            Weight for alignment loss.
        beta: float
            Weight for fourier regularization loss.
        gamma: float
            Weight for smoothness loss.
    
    Returns:
        loss: Any
            Total loss. Shape: [B].
    """
    # Prepare field
    field_comb = field[...,0] + 1j * field[...,1]  # Combine real and imag parts
    field_real = backend.ifft2(backend.ifftshift(field_comb, dim=(-3, -2)), dim=(-3, -2))  # Back to spatial domain
    
    loss_al = alignment_loss(backend, field_real, target_field, weights)
    loss_f = fourier_regularization_loss(backend, field_comb, lattice.period)
    loss_s = smoothness_loss(backend, lattice, field_real)
    
    total = alpha * loss_al + beta * loss_f + gamma * loss_s

    return total  

# ----- Jones normalization -----
def normalize_jones(backend: Backend, field: Any) -> Any:
    """
    Generate a Jones vector field following the "Jones" method of Antos 2009 https://doi.org/10.1364/OE.17.007269.

    Parameters
    ----------
    field : Any
        Vector field, shape [B, Nx, Ny, 2], possibly complex.

    Returns
    -------
    jones_field : Any
        Jones-normalized vector field, shape [B, Nx, Ny, 2], complex.
    """
    # Sanity check
    if field.shape[-1] != 2:
        raise ValueError("Last dimension of field must be size 2.")
    
    if len(field.shape) != 4:
        raise ValueError("Input field must be a 4D tensor with shape [B, Nx, Ny, 2].")

    # Global normalization 
    field = normalize_max_global(backend, field)

    # Magnitude |field|
    magnitude = _field_magnitude(backend, field)  # [B, Nx, Ny, 1]

    # Handle near-zero magnitude safely
    magnitude_near_zero = backend.isclose(magnitude, backend.asarray(0.0))
    magnitude_safe = backend.where(
        magnitude_near_zero,
        backend.ones_like(magnitude),
        magnitude,
    )

    inv_sqrt2 = 1.0 / backend.sqrt(backend.asarray(2.0))

    # Normalized components
    tx_norm = backend.where(
        magnitude_near_zero,
        inv_sqrt2,
        field[..., 0:1] / magnitude_safe,
    )

    ty_norm = backend.where(
        magnitude_near_zero,
        inv_sqrt2,
        field[..., 1:2] / magnitude_safe,
    )

    # Phase parameters
    phi = backend.pi / 8.0 * (1.0 + backend.cos(backend.pi * magnitude))

    # Complex angle of normalized vector
    theta = backend.angle(tx_norm + 1j * ty_norm)

    # Jones construction
    exp_i_theta = backend.exp(1j * theta)

    jx = exp_i_theta * (
        tx_norm * backend.cos(phi)
        - ty_norm * 1j * backend.sin(phi)
    )

    jy = exp_i_theta * (
        ty_norm * backend.cos(phi)
        + tx_norm * 1j * backend.sin(phi)
    )

    return backend.cat([jx, jy], dim=-1)

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