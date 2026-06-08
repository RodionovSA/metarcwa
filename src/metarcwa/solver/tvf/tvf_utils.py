# metarcwa/solver/tvf/tvf_utils.py
# Main functions for TVF operations 

from typing import Any, Tuple
import torch

from metarcwa.model.geometry.lattice import Lattice

# ----- Gradient computation for 2D scalar fields -----
def _grad_periodic(lattice: Lattice, s: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute gradient of scalar field s.
    Use central differences with periodic boundary conditions.
    Normalize by lattice spacing to avoid incorrect gradient magnitude in anisotropic lattices.

    Parameters:
        lattice: Lattice
            Lattice object defining the grid and spacing.
        s: input scalar field
            shape: [B, Nx, Ny]
    Returns:
        sx, sy: [B, Nx, Ny]
    """
    # central differences with periodic boundary conditions
    gradx = 0.5 * (torch.roll(s, shifts=-1, dims=-2) - torch.roll(s, shifts=1, dims=-2))
    grady = 0.5 * (torch.roll(s, shifts=-1, dims=-1) - torch.roll(s, shifts=1, dims=-1))
    
    # Normalize by lattice spacing
    delta_x, delta_y = lattice.delta
    
    gradx = gradx / delta_x
    grady = grady / delta_y
    return gradx, grady

# ----- Low-pass filtering functions for 2D vector fields -----
def low_pass_mask(Nx: int, Ny: int, M: int, N: int) -> torch.Tensor:
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
    kx = torch.arange(0, Nx) - Nx//2
    ky = torch.arange(0, Ny) - Ny//2
    kx, ky = torch.meshgrid(kx, ky, indexing="ij")

    mask = (torch.abs(kx) <= M) & (torch.abs(ky) <= N)
    return mask

def low_pass_filter(grad: tuple[torch.Tensor, torch.Tensor], M: int, N: int) -> Tuple[torch.Tensor,torch.Tensor]:
    """
    Low-pass filter 2D data in Fourier domain.

    Parameters:
        grad: tuple[torch.Tensor, torch.Tensor]
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
    gradx_fft = torch.fft2(gradx, dim=(-2, -1))  # [B, Nx, Ny]
    grady_fft = torch.fft2(grady, dim=(-2, -1))  # [B, Nx, Ny]
    
    # Shift zero frequency component to center
    gradx_fft_shifted = torch.fftshift(gradx_fft, dim=(-2, -1))
    grady_fft_shifted = torch.fftshift(grady_fft, dim=(-2, -1))
    
    mask = low_pass_mask(Nx, Ny, M, N)  # [Nx, Ny]
    mask = torch.astype(mask, gradx.dtype)
    
    filteredx_fft_shifted = gradx_fft_shifted * mask  # [B, Nx, Ny]
    filteredy_fft_shifted = grady_fft_shifted * mask  # [B, Nx, Ny]
    
    # Shift back
    filteredx_fft = torch.ifftshift(filteredx_fft_shifted, dim=(-2, -1))
    filteredy_fft = torch.ifftshift(filteredy_fft_shifted, dim=(-2, -1))
    
    # Inverse FFT to get filtered data
    filteredx = torch.ifft2(filteredx_fft)  # [B, Nx, Ny]
    filteredy = torch.ifft2(filteredy_fft)  # [B, Nx, Ny]
    
    return (filteredx, filteredy)

# ----- Normalization functions for 2D vector fields -----
def _field_magnitude(field: torch.Tensor) -> torch.Tensor:
    """
    Compute the magnitude of a 2D vector field.
    Safely handles sqrt for autograd.

    Parameters:
        field: torch.Tensor
            Vector field. Shape: [B, Nx, Ny, 2].
    Returns:
        magnitude: torch.Tensor
            Magnitude of the vector field. Shape: [B, Nx, Ny, 1].
    """
    # compute magnitude squared
    mag_sq = torch.sum(torch.abs(field) ** 2, dim=-1, keepdim=True)
    
    # avoid NaNs for grad at zero field
    is_zero = mag_sq == 0
    mag_sq_safe = torch.where(is_zero, torch.ones_like(mag_sq), mag_sq) # replace zeros with ones for safe sqrt
    mag = torch.where(is_zero, torch.zeros_like(mag_sq), torch.sqrt(mag_sq_safe)) # calculate magnitude
    
    return mag

def normalize_max_global(field: torch.Tensor) -> torch.Tensor:
    """
    Normalize a 2D vector field by its global maximum magnitude.
    This is a global normalization (not elementwise), numerically safe.

    Parameters:
        vector_field: torch.Tensor
            Vector field to be normalized. Shape: [B, Nx, Ny, 2].
    Returns:
        normalized_vector_field: torch.Tensor
            Normalized vector field. Shape: [B, Nx, Ny, 2].
    """
    mag = _field_magnitude(field)  # [B, Nx, Ny, 1]
    
    # global maximum magnitude
    max_mag = torch.amax(mag, dim=(-3, -2), keepdim=True)
    
    # avoid division by zero
    max_mag_safe = torch.where(max_mag == 0, 1, max_mag)

    return field / max_mag_safe

def normalize_elementwise(field: torch.Tensor) -> torch.Tensor:
    """
    Normalize a 2D vector field so that each element's magnitude is 1.
    This is an elementwise normalization, numerically safe.

    Parameters:
        vector_field: torch.Tensor
            Vector field to be normalized. Shape: [B, Nx, Ny, 2].
    Returns:
        normalized_vector_field: torch.Tensor
            Normalized vector field. Shape: [B, Nx, Ny, 2].
    """
    # compute magnitude squared
    mag_sq = torch.sum(torch.abs(field) ** 2, dim=-1, keepdim=True)
    
    # avoid NaNs for zero field
    is_zero = mag_sq == 0
    mag_sq_safe = torch.where(is_zero, torch.ones_like(mag_sq), mag_sq) # replace zeros with ones for safe sqrt
    mag = torch.sqrt(mag_sq_safe) # calculate magnitude

    return field / mag

# ----- Loss functions for 2D vector fields -----
def alignment_loss(field: torch.Tensor, 
                   target_field: torch.Tensor, 
                   weights: torch.Tensor) -> torch.Tensor:
    """
    Compute alignment loss between field and target_field, weighted by weights.
    Loss = mean(weights * |field - target_field|^2)
    
    Parameters:
        field: torch.Tensor
            Current vector field. Shape: [B, Nx, Ny, 2].
        target_field: torch.Tensor
            Target vector field. Shape: [B, Nx, Ny, 2].
        weights: torch.Tensor
            Weights for each element. Shape: [B, Nx, Ny, 1] or [B, Nx, Ny].
    
    Returns:
        loss: torch.Tensor
            Alignment loss. Shape: [B].
    """
    if field.shape != target_field.shape:
        raise ValueError("field and target_field must have the same shape.")
    if weights.shape != field.shape[:-1] and weights.shape != field.shape[:-1] + (1,):
        raise ValueError("weights must have shape [B, Nx, Ny] or [B, Nx, Ny, 1].")
    
    diff = field - target_field  # [B, Nx, Ny, 2]
    diff_sq = torch.sum(torch.abs(diff) ** 2, dim=-1, keepdim=True)  # [B, Nx, Ny, 1]
    weighted_diff_sq = weights * diff_sq  # [B, Nx, Ny, 1] or [B, Nx, Ny]
    
    if weights.shape != field.shape[:-1]:
        weighted_diff_sq = torch.squeeze(weighted_diff_sq, dim=-1)  # [B, Nx, Ny]
        
    loss = torch.mean(weighted_diff_sq, dim=(-1, -2)) # [B]
    return loss

def fourier_regularization_loss(field: torch.Tensor, 
                                period: tuple[float, float]) -> torch.Tensor:
    """
    Compute Fourier regularization loss for a 2D vector field.
    Loss = mean(|k|^2 * |F(field)|^2), where F is the 2D FFT.

    Parameters:
        field: torch.Tensor
            Vector field. Shape: [B, Nx, Ny, 2].
        period: tuple[float, float]
            Physical period of the lattice in x and y directions.
    """
    Lx, Ly = period
    
    # Calculate |k|^2
    _, Nx, Ny, _ = field.shape
    kx = 2* torch.pi / Lx * (torch.arange(0, Nx) - Nx//2)
    ky = 2* torch.pi / Ly * (torch.arange(0, Ny) - Ny//2)
    Kx, Ky = torch.meshgrid(kx, ky, indexing="ij")
    K_norm2 = Lx*Ly*(torch.abs(Kx) ** 2 + torch.abs(Ky) ** 2)  # [Nx, Ny]
    
    loss = torch.mean(K_norm2 * torch.sum(torch.abs(field) ** 2, dim=-1), dim=(-2, -1))  # [B]
    return loss

def smoothness_loss(lattice: Lattice, field: torch.Tensor) -> torch.Tensor:
    """
    Compute smoothness loss for a 2D vector field.
    Loss = mean(|grad(field)|^2)

    Parameters:
        field: torch.Tensor
            Vector field. Shape: [B, Nx, Ny, 2].
            
    Returns:
        loss: torch.Tensor
            Smoothness loss. Shape: [B].
    """
    field_x = field[:,:,:,0]  # [B, Nx, Ny]
    field_y = field[:,:,:,1]  # [B, Nx, Ny]
    
    gradx_x, grady_x = _grad_periodic(lattice, field_x)  # [B, Nx, Ny]
    gradx_y, grady_y = _grad_periodic(lattice, field_y)  # [B, Nx, Ny]
    
    grad_sq = torch.abs(gradx_x) ** 2 + torch.abs(grady_x) ** 2 + torch.abs(gradx_y) ** 2 + torch.abs(grady_y) ** 2  # [B, Nx, Ny]
    
    loss = torch.mean(grad_sq, dim=(-1, -2))  # [B]
    return loss

def total_loss(field: torch.Tensor, 
               target_field: torch.Tensor, 
               weights: torch.Tensor, 
               lattice: Lattice,
               alpha: float = 1.0,
               beta: float = 1e-3,
               gamma: float = 1.0) -> torch.Tensor:
    """
    Compute total loss as a weighted sum of alignment loss, fourier regularization loss, and smoothness loss.
    
    Parameters:
        field: torch.Tensor
            Current vector field. Shape: [B, Nx, Ny, 2].
        target_field: torch.Tensor
            Target vector field. Shape: [B, Nx, Ny, 2].
        weights: torch.Tensor
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
        loss: torch.Tensor
            Total loss. Shape: [B].
    """
    # Prepare field
    field_comb = field[...,0] + 1j * field[...,1]  # Combine real and imag parts
    field_real = torch.ifft2(torch.ifftshift(field_comb, dim=(-3, -2)), dim=(-3, -2))  # Back to spatial domain
    
    loss_al = alignment_loss(field_real, target_field, weights)
    loss_f = fourier_regularization_loss(field_comb, lattice.period)
    loss_s = smoothness_loss(lattice, field_real)
    
    total = alpha * loss_al + beta * loss_f + gamma * loss_s

    return total  

# ----- Jones normalization -----
def normalize_jones(field: torch.Tensor) -> torch.Tensor:
    """
    Generate a Jones vector field following the "Jones" method of Antos 2009 https://doi.org/10.1364/OE.17.007269.

    Parameters
    ----------
    field : torch.Tensor
        Vector field, shape [B, Nx, Ny, 2], possibly complex.

    Returns
    -------
    jones_field : torch.Tensor
        Jones-normalized vector field, shape [B, Nx, Ny, 2], complex.
    """
    # Sanity check
    if field.shape[-1] != 2:
        raise ValueError("Last dimension of field must be size 2.")
    
    if len(field.shape) != 4:
        raise ValueError("Input field must be a 4D tensor with shape [B, Nx, Ny, 2].")

    # Global normalization 
    field = normalize_max_global(field)

    # Magnitude |field|
    magnitude = _field_magnitude(field)  # [B, Nx, Ny, 1]

    # Handle near-zero magnitude safely
    magnitude_near_zero = torch.isclose(magnitude, torch.asarray(0.0))
    magnitude_safe = torch.where(
        magnitude_near_zero,
        torch.ones_like(magnitude),
        magnitude,
    )

    inv_sqrt2 = 1.0 / torch.sqrt(torch.asarray(2.0))

    # Normalized components
    tx_norm = torch.where(
        magnitude_near_zero,
        inv_sqrt2,
        field[..., 0:1] / magnitude_safe,
    )

    ty_norm = torch.where(
        magnitude_near_zero,
        inv_sqrt2,
        field[..., 1:2] / magnitude_safe,
    )

    # Phase parameters
    phi = torch.pi / 8.0 * (1.0 + torch.cos(torch.pi * magnitude))

    # Complex angle of normalized vector
    theta = torch.angle(tx_norm + 1j * ty_norm)

    # Jones construction
    exp_i_theta = torch.exp(1j * theta)

    jx = exp_i_theta * (
        tx_norm * torch.cos(phi)
        - ty_norm * 1j * torch.sin(phi)
    )

    jy = exp_i_theta * (
        ty_norm * torch.cos(phi)
        + tx_norm * 1j * torch.sin(phi)
    )

    return torch.cat([jx, jy], dim=-1)

