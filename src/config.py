
from dataclasses import dataclass


from dataclasses import dataclass, field
from typing import Literal, Optional, Tuple

@dataclass
class LayerConfig:
    """
    Configuration parameters for the RCWA LayerSolver.
    
    Parameters
    ----------
    M, N : int
        Truncation order along x and y directions.
    closed_form : bool
        Whether to use analytic Fourier coefficients for simple shapes.
    factorization : {Jones, Pol, Normal, Jones_direct, None}
        Type of Li's factorization method to use.
    tvf_optimizer : {'LBFGS'}, optional
        Optimizer to use for Tangent Vector Fields (TVF) computation.
    """

    # ==== Harmonics ====
    M: int
    N: int

    # ==== Fourier coefficients ====
    closed_form: bool = True
    circ_truncation: bool = False  # Circular truncation if True, else rectangular
    inverse_regularization: float = 1e-8  # Regularization for inverse Fourier coefficients
    
    # === Matrix Inversion ====
    inverse_matrix_method: Literal['solve', 'inv', 'pinv'] = 'solve'
    
    # ==== Factorization ====
    factorization: Literal['Jones', 'Pol', 'Normal', 'Jones_direct', 'None'] = 'Jones'
        # Jones, Pol, Normal, Jones_direct, None
        
    # ==== TVF optimizer ====
    tvf_optimizer: Optional[Literal['LBFGS']] = 'LBFGS'
    tvf_alpha: float = 1.0
    tvf_beta: float = 1.0e-6
    tvf_gamma: float = 0.0
    tvf_steps: int = 1
    
    # ==== Modes solver ====
    modes_solver: Literal['eig', 'eigh', 'svd', 'qr'] = 'eig'

    # Derived fields
    @property
    def Nh(self) -> int:
        """Total number of harmonics (M*N)."""
        return self.M * self.N


@dataclass
class Config:
    pass