# metarcwa/solver/modesolver.py
# Description

from src.metarcwa.model.layer import HomogeneousLayer, PatternedLayer
from src.metarcwa.model.medium import MediumSpec
from .blockmatrix import Block2x2

class ModeSolver:
    """Description"""
    def __init__(self, config, kx, ky):
        pass
    
    def solve(element: HomogeneousLayer|PatternedLayer|MediumSpec) -> Block2x2:
        """Outputs S matrix as Block2x2. Need to branch here depending on element type 
        and medium to use different solvers"""
        pass
    

