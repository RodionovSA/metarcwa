#metarcwa/solver/tvf/tvf.py
# DESCRIPTION

from metarcwa.model import Lattice
from .optimizers import make_optimizer
from .tvf_utils import _grad_periodic, low_pass_filter, normalize_elementwise, normalize_jones, normalize_max_global, _field_magnitude, total_loss

import torch
from typing import Any, Tuple


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
                 lattice: Lattice,
                 method: str, 
                 optimizer: str = "LBFGS"):
        """
        Parameters:
            lattice: Lattice
                Lattice object defining the grid and spacing.
            method: str
                Method for TVF computation (Jones, Pol, Normal, Jones_direct).
            optimizer: str
                Optimizer to use for TVF optimization (default: "LBFGS").
        """
        self.lattice = lattice
        self.M = lattice.M
        self.N = lattice.N
        self.method = method
        if method not in ["Jones", "Pol", "Normal", "Jones_direct"]:
            raise ValueError(f"Unknown method '{method}' for TVF computation")
        self.optimizer = make_optimizer(optimizer)
        
    def _prepare_field(self, field: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Prepare the vector field before processing.
        
        Parameters:
            field: torch.Tensor
                Input scalar field. Shape: [B, Nx, Ny].
        
        Returns:
            target_field: torch.Tensor
                Prepared target vector field. Shape: [B, Nx, Ny, 2].
            initial_field_fft: torch.Tensor
                Initial vector field in Fourier domain for optimization. Shape: [B, Nx, Ny, 2].
            weights: torch.Tensor
                Weights for alignment loss. Shape: [B, Nx, Ny, 1].
        """
        # Step 1: Compute gradients assuming periodic boundaries
        gradx, grady = _grad_periodic(self.lattice, field)  # [B, Nx, Ny]
        
        # Step 2: Filter and resample gradients ((4M+1, 4N+1) window)
        gradx_f, grady_f = low_pass_filter((gradx, grady), M=2*self.M, N=2*self.N)  # [B, Nx, Ny]
        
        # Step 3: Global normalization
        grad_n = normalize_max_global(torch.stack((gradx_f, grady_f), dim=-1))
        gradx_n, grady_n = grad_n[...,0], grad_n[...,1]  # [B, Nx, Ny]
        
        # Step 4: Define target field
        target_field = self.backend.stack((grady_n, -gradx_n), dim=-1)  # [B, Nx, Ny, 2]
        
        # Step 5: Normalize elementwise
        target_field = normalize_elementwise(target_field)  # [B, Nx, Ny, 2]
        
        # Step 6: Define initial field
        if self.method == "Jones_direct":
            target_field = normalize_jones(target_field)  # [B, Nx, Ny, 2]
            initial_field = target_field
        else:
            initial_field = torch.stack((grady_n, -gradx_n), dim=-1)  # [B, Nx, Ny, 2]
        
        # Step 7: Shift initial field to fourier domain
        initial_field_fft = torch.fftshift(torch.fft2(initial_field, dim=(-3, -2)), dim=(-3, -2))  # [B, Nx, Ny, 2]
            
        # Step 8: Compute alignment loss weights
        weights = _field_magnitude(torch.stack((gradx_n, grady_n), dim=-1))  # [B, Nx, Ny, 1]
            
        return target_field, initial_field_fft, weights
    
    def _optimize(self, 
                  field: torch.Tensor, 
                  alpha: float = 1.0, 
                  beta: float = 1e-8, 
                  gamma: float = 1.0,
                  steps: int = 1) -> torch.Tensor:
        """
        Optimize the Tangent Vector Field (TVF) from the input scalar field.
        Optimization works on geometry-based TVF derived from Re(field) in Fourier domain.
        The resulting TVF does not keep computational graph from field.

        Parameters:
            field: torch.Tensor
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
            optimized_field: torch.Tensor
                Computed TVF. Shape: [B, Nx, Ny, 2].
        """
        # Sanity check
        if len(field.shape) != 3:
            raise ValueError("Input field must be a 3D tensor with shape [B, Nx, Ny].")
        
        # Clone and detach grads
        field = torch.real(field.detach())
        
        # Prepare fields
        target_field, initial_field_fft, weights = self._prepare_field(field)
        initial_field = torch.stack([torch.real(initial_field_fft), 
                                    torch.imag(initial_field_fft)], dim=-1) #Split real and imag parts
        
        def loss_fn(current_field: Any) -> Any:
            return total_loss(current_field, 
                              target_field, 
                              weights=weights, 
                              lattice=self.lattice,
                              alpha=alpha,
                              beta=beta,
                              gamma=gamma)
        
        optimized_field = initial_field.detach().requires_grad(True)
        for _ in range(steps):
            optimized_field = self.optimizer.step(optimized_field, loss_fn)
        
        optimized_field = optimized_field.detach() # Detach from computation graph
        optimized_field = optimized_field[...,0] + 1j * optimized_field[...,1]  # Combine real and imag parts
        optimized_field = torch.ifft2(torch.ifftshift(optimized_field, dim=(-3, -2)), dim=(-3, -2))  # Back to spatial domain
        
        return optimized_field
    
    def compute(self, 
                field: torch.Tensor, 
                alpha: float = 1.0, 
                beta: float = 1e-8, 
                gamma: float = 1.0,
                steps: int = 1) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the Tangent Vector Field (TVF) from the input scalar field.
        Parameters:
            field: torch.Tensor
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
            Tx: torch.Tensor
                Tangent vector field x-component. Shape: [B, Nx, Ny].
            Ty: torch.Tensor
                Tangent vector field y-component. Shape: [B, Nx, Ny].
            
        """
        optimized_field = normalize_max_global(self._optimize(field, alpha, beta, gamma, steps))
        
        if self.method == "Jones":
            # Transform to Jones field
            optimized_field = normalize_jones(optimized_field)
        
        elif self.method == "Pol":
            # Global normalization
            optimized_field = normalize_max_global(torch.real(optimized_field))
            
        elif self.method == "Normal":
            # Elementwise normalization
            optimized_field = normalize_elementwise(torch.real(optimized_field))
            
        elif self.method == "Jones_direct":
            # Already Jones-normalized
            pass 

        return optimized_field[...,0], optimized_field[...,1]  # Return Tx, Ty components