# metarcwa/solver/tvf/tvf.py
# Tangent Vector Field (TVF) solver for RCWA factorization rules.
#
# Interface: all arguments are plain torch.Tensor or Python types.
# Device and dtype are inherited from input tensors via PyTorch ops.

import torch
from typing import Tuple

from .optimizers import make_optimizer
from .tvf_utils import (
    _grad_periodic,
    low_pass_filter,
    low_pass_mask,
    normalize_elementwise,
    normalize_jones,
    normalize_max_global,
    _field_magnitude,
    total_loss,
)


class TVF:
    """
    Tangent Vector Field (TVF) computation for RCWA factorization rules.

    The TVF is derived from the real part of the permittivity field and is
    used to implement the Li factorization rules for patterned layers.
    Supports any 2-D Bravais lattice (rectangular, hexagonal, oblique, ...).

    Grid convention
    ---------------
    The input field ``eps`` of shape ``[B, D0, D1]`` is assumed to be sampled
    on the fractional unit-cell parallelogram grid::

        sample (i/D0, j/D1)  ->  Cartesian position  f2*a2 + f1*a1
            axis -2 (size D0)  <->  a2 direction  (f2 = i/D0)
            axis -1 (size D1)  <->  a1 direction  (f1 = j/D1)

    This matches the metashapes rasterization convention (indexing="xy"
    returns [ny, nx] where ny <-> a2 and nx <-> a1).

    Supported methods
    -----------------
    ``"Jones"``
        Antos 2009 (DOI 10.1364/OE.17.007269). Smooth complex Jones field.
    ``"Pol"``
        S4 (DOI 10.1016/j.cpc.2012.04.026). Globally normalised real field.
    ``"Normal"``
        Schuster 2007 (DOI 10.1364/JOSAA.24.002880). Elementwise unit field.
    ``"Jones_direct"``
        FMMax (DOI 10.1364/OE.503481). Jones field from the filtered gradient.

    Parameters
    ----------
    a1 : torch.Tensor
        First lattice vector, shape ``[2]``. Corresponds to axis -1 of the
        field (fractional coord f1).  Any direction/magnitude is supported.
    a2 : torch.Tensor
        Second lattice vector, shape ``[2]``. Corresponds to axis -2 of the
        field (fractional coord f2).
    M : int
        Fourier truncation order in the **a1** direction: harmonics m with
        ``|m| <= M`` are kept.
    N : int
        Fourier truncation order in the **a2** direction: harmonics n with
        ``|n| <= N`` are kept.
    method : str
        TVF method. One of ``"Jones"``, ``"Pol"``, ``"Normal"``,
        ``"Jones_direct"``.
    optimizer : str
        Optimizer name (case-insensitive). Default ``"newton"``.
    alpha : float or None
            Alignment loss weight.  ``None`` → 1.0.
    beta : float or None
        Fourier regularization weight.  ``None`` → 0.05.  Both ``beta`` and
        ``gamma`` weight the physical Dirichlet energy ``∫‖∇f‖²dA`` and
        are on the same scale; typical values are 0.01–0.1.
    gamma : float or None
        Smoothness loss weight.  ``None`` → 0.05.  See ``beta``.
    steps : int
        Optimizer steps.  Default 1 (exact for the Newton solve on a
        quadratic loss).
    """

    METHODS = ("Jones", "Pol", "Normal", "Jones_direct")

    def __init__(
        self,
        a1: torch.Tensor,
        a2: torch.Tensor,
        M: int,
        N: int,
        method: str,
        optimizer: str = "newton",
        alpha: float | None = None,
        beta: float | None = None,
        gamma: float | None = None,
        steps: int = 1,
    ):
        if method not in self.METHODS:
            raise ValueError(
                f"Unknown TVF method '{method}'. Choose from {self.METHODS}."
            )
        self.a1 = a1
        self.a2 = a2
        self.M = int(M)
        self.N = int(N)
        self.method = method
        self.optimizer = make_optimizer(optimizer)
        self.alpha = 1.0 if alpha is None else alpha
        self.beta  = 0.05 if beta is None else beta
        self.gamma = 0.05 if gamma is None else gamma
        self.steps = steps

    def _prepare_field(
        self, field: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Derive target, initial (Fourier-domain), and weight tensors from a
        real scalar field.

        The field must follow the class-level grid convention:
        axis -2 (size D0) along a2, axis -1 (size D1) along a1.

        Parameters
        ----------
        field : torch.Tensor
            Real scalar field (Re of permittivity). Shape ``[B, D0, D1]``.

        Returns
        -------
        target_field : torch.Tensor
            Spatial target vector field, shape ``[B, D0, D1, 2]``.
        initial_field_fft : torch.Tensor
            Centred Fourier transform of the initial vector field,
            shape ``[B, D0, D1, 2]``, complex.
        weights : torch.Tensor
            Alignment loss weights (gradient magnitude),
            shape ``[B, D0, D1, 1]``.
        """
        # Step 1: Periodic gradients
        gradx, grady = _grad_periodic(field, self.a1, self.a2)          

        # Step 2: Low-pass filter (window 2M+1 × 2N+1)
        gradx_f, grady_f = low_pass_filter((gradx, grady), M=self.M, N=self.N)

        # Step 3: Global normalisation
        grad_n = normalize_max_global(torch.stack((gradx_f, grady_f), dim=-1))
        gradx_n = grad_n[..., 0]
        grady_n = grad_n[..., 1]

        # Step 4 & 5: Target = perpendicular to gradient, elementwise-normalised
        target_field = normalize_elementwise(
            torch.stack((grady_n, -gradx_n), dim=-1)                    # [B, Ny, Nx, 2]
        )

        # Step 6: Initial field, with optional Jones conversion for direct method
        if self.method == "Jones_direct":
            target_field = normalize_jones(target_field)
            # Use the full complex Jones field as the initial guess 
            initial_field = target_field                                  # [B, Ny, Nx, 2] complex
        else:
            initial_field = torch.stack((grady_n, -gradx_n), dim=-1)   # [B, Ny, Nx, 2]

        # Step 7: Centred 2-D FFT of the 2-component initial field
        # FFT is applied over the two spatial dims (-3, -2); the vector dim is -1.
        initial_field_fft = torch.fft.fftshift(
            torch.fft.fft2(initial_field, dim=(-3, -2)),
            dim=(-3, -2),
        )                                                                # [B, Ny, Nx, 2] complex

        # Step 8: Gradient magnitude as alignment weights
        weights = _field_magnitude(
            torch.stack((gradx_n, grady_n), dim=-1)                     # [B, Ny, Nx, 2]
        )                                                                # [B, Ny, Nx, 1]

        return target_field, initial_field_fft, weights

    def _optimize(
        self,
        field: torch.Tensor,
    ) -> torch.Tensor:
        """
        Optimize the TVF from a real scalar permittivity field.

        Optimization is performed **exclusively over the in-band Fourier
        coefficients** (the ``(2M+1)×(2N+1)`` harmonics kept by the
        simulation), enforcing a hard band limit.

        Parameters
        ----------
        field : torch.Tensor
            Input permittivity field (complex accepted; imaginary part is
            discarded). Shape ``[B, Ny, Nx]``.

        Returns
        -------
        optimized_field : torch.Tensor
            TVF in spatial domain, shape ``[B, Ny, Nx, 2]``, **complex**.
        """
        if field.ndim != 3:
            raise ValueError("field must be 3-D with shape [B, Ny, Nx].")

        # Work on real part; detach from the model graph
        field_real = torch.real(field).detach()
        B, D0, D1 = field_real.shape

        # Derive target, initial Fourier field, and loss weights
        target_field, initial_field_fft, weights = self._prepare_field(field_real)
        # initial_field_fft: [B, D0, D1, 2] complex, centred

        # ── Hard band-limit: gather only in-band coefficients ──────────────────
        # mask shape [D0, D1]; True for the (2M+1)(2N+1) kept harmonics
        mask = low_pass_mask(D0, D1, self.M, self.N, device=field_real.device)  # [D0, D1]
        # flat indices of in-band positions
        in_band_idx = mask.flatten().nonzero(as_tuple=True)[0]  # [num_terms]

        # Extract in-band complex coefficients: [B, num_terms, 2]
        fft_flat = initial_field_fft.reshape(B, D0 * D1, 2)       # [B, D0*D1, 2]
        init_inband = fft_flat[:, in_band_idx, :]                   # [B, num_terms, 2] complex

        # Optimization variable: real/imag of in-band coeffs
        # Shape: [B, num_terms, 2, 2], last dim = (real, imag)
        params = torch.stack(
            [init_inband.real, init_inband.imag], dim=-1
        ).detach().requires_grad_(True)                             # [B, num_terms, 2, 2]

        def loss_fn(p: torch.Tensor) -> torch.Tensor:
            # Reconstruct complex in-band field: [B, num_terms, 2]
            inband_complex = p[..., 0] + 1j * p[..., 1]

            # Scatter back into full [B, D0*D1, 2] grid (zeros outside band)
            full_fft_flat = torch.zeros(B, D0 * D1, 2,
                                        dtype=inband_complex.dtype,
                                        device=p.device)
            full_fft_flat[:, in_band_idx, :] = inband_complex

            field_fft = full_fft_flat.reshape(B, D0, D1, 2)       # centred [B,D0,D1,2]

            # Unpack into the [B, D0, D1, 2, 2] format expected by total_loss
            params_full = torch.stack(
                [field_fft.real, field_fft.imag], dim=-1
            )

            return total_loss(
                params_full,
                target_field,
                weights=weights,
                a1=self.a1,
                a2=self.a2,
                alpha=self.alpha,
                beta=self.beta,
                gamma=self.gamma,
            )

        params = self.optimizer.minimize(params, loss_fn, steps=self.steps)
        params = params.detach()

        # Scatter optimized in-band coeffs back to full grid
        inband_opt = params[..., 0] + 1j * params[..., 1]          # [B, num_terms, 2]
        full_fft_flat = torch.zeros(B, D0 * D1, 2,
                                    dtype=inband_opt.dtype,
                                    device=params.device)
        full_fft_flat[:, in_band_idx, :] = inband_opt
        field_fft = full_fft_flat.reshape(B, D0, D1, 2)            # [B, D0, D1, 2] centred

        # Inverse FFT back to spatial domain
        optimized_field = torch.fft.ifft2(
            torch.fft.ifftshift(field_fft, dim=(-3, -2)), dim=(-3, -2)
        )                                                            # [B, D0, D1, 2] complex

        return optimized_field

    def compute(
        self,
        field: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the Tangent Vector Field from a permittivity field.

        The result is **detached** from the input field's computation graph
        by design — the TVF is a geometry-derived precomputed quantity.

        Parameters
        ----------
        field : torch.Tensor
            Permittivity field (complex; imaginary part is discarded for
            geometry extraction). Shape ``[B, Ny, Nx]``.

        Returns
        -------
        Tx : torch.Tensor
            x-component of TVF. Shape ``[B, Ny, Nx]``.
            Complex for ``"Jones"`` and ``"Jones_direct"``; real for ``"Pol"``
            and ``"Normal"``.
        Ty : torch.Tensor
            y-component of TVF. Shape ``[B, Ny, Nx]``. Same dtype as ``Tx``.
        """
        optimized = self._optimize(field)    # [B, Ny, Nx, 2] complex

        if self.method == "Jones":
            # Apply Jones normalization to the real part of the optimized field.
            optimized = normalize_jones(optimized.real)                  # [B, Ny, Nx, 2] complex

        elif self.method == "Pol":
            optimized = normalize_max_global(optimized.real)             # [B, Ny, Nx, 2] real

        elif self.method == "Normal":
            optimized = normalize_elementwise(optimized.real)            # [B, Ny, Nx, 2] real

        elif self.method == "Jones_direct":
            # The optimized field is already complex (Jones initial + target);
            # normalize to max magnitude 1.
            optimized = normalize_max_global(optimized)                  # [B, Ny, Nx, 2] complex

        return optimized[..., 0], optimized[..., 1]
