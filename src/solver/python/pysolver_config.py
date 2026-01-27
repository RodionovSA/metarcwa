#src/solver/python/pysolver_config.py
# Configuration parameters for the Python RCWA Solver.

from dataclasses import dataclass
from typing import Literal

@dataclass(frozen=True)
class PySolverConfig:
    inverse_matrix_method: Literal['solve', 'inv', 'pinv'] 
    factorization: Literal['Jones', 'Pol', 'Normal', 'Jones_direct', 'None'] 
    modes_solver: Literal['eig', 'eigh'] 
    hsimplify: bool 
    smlayer: Literal['Analytic', 'Numerical'] 