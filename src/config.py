
from dataclasses import dataclass


from dataclasses import dataclass, field
from typing import Literal, Optional, Tuple

@dataclass
class EigenConfig:
    """
    Configuration parameters for the RCWA eigensolver.
    
    Parameters
    ----------
    M, N : int
        Number of Fourier harmonics along x and y directions.
    closed_form : bool
        Whether to use analytic Fourier coefficients for simple shapes.
    Factorization : bool
        Apply Li's factorization rules (True/False).
    fourier_rule : {'TF', 'EF', 'NONE'}
        Type of Fourier factorization for Li's rule (Total/Even/None).
    subpixel : bool
        Use anisotropic subpixel smoothing for material interfaces.
    solver : {'eigh', 'eig', 'svd', 'custom'}
        Eigenvalue solver backend.
    stabilization : bool
        Use eigenvalue stabilization (e.g., Kato shift).
    kato_shift : float
        Shift size for stabilization.
    verbosity : int
        0 = silent, 1 = some logs, 2 = debug.
    """

    # ==== Harmonics ====
    M: int
    N: int

    # ==== Fourier coefficients ====
    closed_form: bool = True
    Factorization: bool = False
    fourier_rule: Literal['TF', 'EF', 'NONE'] = 'TF'
        # TF  = Total Factorization (Li)
        # EF  = Even Factorization
        # NONE = no factorization
        
    subpixel: bool = False
        # Use anisotropic subpixel smoothing for material interfaces

    # ==== Solver options ====
    solver: Literal['eigh', 'eig', 'svd', 'custom'] = 'eigh'
        # eigh  – Hermitian solver (preferred, fastest, most stable)
        # eig   – generic solver (if matrix is not Hermitian)
        # svd   – use SVD-based mode extraction (rare)
        # custom – user-supplied backend

    # ==== Stabilization tricks ====
    stabilization: bool = False
    kato_shift: float = 1e-6

    # ==== Debug & logging ====
    verbosity: int = 0

    # Derived fields
    @property
    def Nh(self) -> int:
        """Total number of harmonics (M*N)."""
        return self.M * self.N


@dataclass
class Config:
    pass