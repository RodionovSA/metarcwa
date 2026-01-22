#src/model/geometry/lattice.py
# Lattice base class for geometric objects.

from typing import Tuple

class Lattice:
    """
    Lattice base class for geometric objects.
    """
    def __init__(self, 
                 period: Tuple[float, float],
                 grid: Tuple[int, int],
                 M: int,
                 N: int):
        '''
        Parameters
        ----------
        period : tuple of float
            (Lx, Ly) period of the lattice. Length units.
        grid : tuple of int
            (Nx, Ny) grid size for sampling the material functions.
        M, N : int
            Truncation order along x and y directions.
        '''
        Lattice._init_validation(period, grid, M, N)
        
        self.period = period
        self.grid = grid
        self.M = M
        self.N = N
    
    @property 
    def delta(self) -> Tuple[float, float]:
        '''
        Grid spacing (dx, dy).
        Returns
        -------
        delta : tuple of float
            (dx, dy) grid spacing. Length units.
        '''
        Lx, Ly = self.period
        Nx, Ny = self.grid
        
        dx = Lx / Nx
        dy = Ly / Ny
        
        return (dx, dy)
    
    @staticmethod
    def _init_validation(period, grid, M, N) -> None:
        if len(period) != 2:
            raise ValueError(f"period must be tuple of 2 floats, got {period}")
        if len(grid) != 2:
            raise ValueError(f"grid must be tuple of 2 ints, got {grid}")
        if not all(isinstance(x, int) for x in grid):
            raise ValueError(f"grid values must be integers, got {grid}")
        if not all(isinstance(x, float) or isinstance(x, int) for x in period):
            raise ValueError(f"period values must be floats, got {period}")
        if any(x <= 0 for x in period):
            raise ValueError(f"period values must be positive, got {period}")
        if any(x <= 0 for x in grid):
            raise ValueError(f"grid values must be positive, got {grid}")
        if not isinstance(M, int) or not isinstance(N, int):
            raise ValueError("M and N must be ints")
        if M <= 0 or N <= 0:
            raise ValueError('M and N must be positive') 