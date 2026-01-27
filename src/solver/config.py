#src/solver/config.py
# Configuration parameters for the RCWA Solver.

from dataclasses import dataclass, field
from typing import Literal, Optional, Tuple

@dataclass
class Config:
    """
    Configuration parameters for the RCWA Solver.
    
    Parameters
    ----------
    solver : {'Python', 'cpp'}
        Solver engine to use.
    use_grads : bool
        Whether to use gradients for optimization.
    closed_form : bool
        Whether to use analytic Fourier coefficients for simple shapes.
    circ_truncation : bool
        Whether to use circular truncation of Fourier orders.
    inverse_regularization : float
        Regularization parameter for inverse Fourier coefficients.
    inverse_matrix_method : {'solve', 'inv', 'pinv'}
        Method for matrix inversion.
    factorization : {Jones, Pol, Normal, Jones_direct, None}
        Type of Li's factorization method to use.
    tvf_optimizer : {'LBFGS'}, optional
        Optimizer to use for Tangent Vector Fields (TVF) computation.
    tvf_alpha : float
        TVF optimization parameter alpha.
    tvf_beta : float
        TVF optimization parameter beta.
    tvf_gamma : float
        TVF optimization parameter gamma.
    tvf_steps : int
        Number of optimization steps for TVF.
    modes_solver : {'eig', 'eigh'}
        Eigenvalue solver to use for mode computation. 
        **Important**: 'eigh' requires Hermitian matrices.
    hsimplify : bool
        Whether to simplify computation for homogeneous layers.
    smlayer : {'Analytic', 'Numerical'}
        Method for S-matrix computation for a layer. If Analytical, uses closed-form expressions.
        If Numerical, computes S-matrix from interface and propagation S-matrices.
    """
    
    # ==== Solver engine ====
    solver: Literal['python', 'cpp'] = 'python'
    use_grads: bool = False  # Whether to use gradients (for optimization)

    # ==== Fourier coefficients ====
    closed_form: bool = True
    circ_truncation: bool = False  # Circular truncation if True, else rectangular
    inverse_regularization: float = 1e-8  # Regularization for inverse Fourier coefficients
    
    # === Matrix Inversion ====
    inverse_matrix_method: Literal['solve', 'inv', 'pinv'] = 'solve'
    
    # ==== Factorization ====
    factorization: Literal['Jones', 'Pol', 'Normal', 'Jones_direct', 'None'] = 'Jones'
        
    # ==== TVF optimizer ====
    tvf_optimizer: Optional[Literal['LBFGS']] = 'LBFGS'
    tvf_alpha: float = 1.0
    tvf_beta: float = 1.0e-6
    tvf_gamma: float = 0.0
    tvf_steps: int = 1
    
    # ==== Modes solver ====
    modes_solver: Literal['eig', 'eigh'] = 'eig' 
    
    # ==== Homogeneous layer ====
    hsimplify: bool = True
    
    # ==== S-matrix computation for a layer ====
    smlayer: Literal['Analytic', 'Numerical'] = 'Numerical'