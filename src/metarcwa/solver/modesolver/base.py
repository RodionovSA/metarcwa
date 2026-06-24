# metarcwa/solver/modesolver/base.py
# Description
import torch
from typing import Tuple

from metarcwa.model.layer import HomogeneousLayer, PatternedLayer
from metarcwa.model.medium import MediumSpec, IsotropicMediumSpec
from metarcwa.solver.blockmatrix import Block2x2
from metarcwa.solver.smatrix import S_layer
from metarcwa.solver.modesolver.homogeneous import homogeneous_modes
from metarcwa.solver.modesolver.isotropic import compute_isotropic
from metarcwa.solver.modesolver.eigsolver import eigsolver

class ModeSolver:
    """Description"""
    def __init__(self, config, wvl, kx, ky, m_flat, n_flat, tvf):
        self.config = config
        self.wvl = wvl
        self.kx = kx
        self.ky = ky
        self.m_flat = m_flat
        self.n_flat = n_flat
        self.tvf = tvf
        self.W0, self.V0 = self._prepare_vacuum()
        
    def _prepare_vacuum(self) -> Tuple[Block2x2, Block2x2]:
        """ Computes W0 and V0 for vacuum"""
        eps = torch.ones_like(self.kx[:, 0, 0])
        _, V0 = homogeneous_modes(eps, self.kx, self.ky)
        return V0.eye_like(), V0
    
    def solve(self, element: HomogeneousLayer|PatternedLayer|MediumSpec) -> Block2x2:
        """Outputs S matrix as Block2x2. Need to branch here depending on element type
        and medium to use different solvers"""
        if isinstance(element, HomogeneousLayer):
            return self._homogeneous(element)
        elif isinstance(element, PatternedLayer):
            pass
        elif isinstance(element, MediumSpec):
            pass
        else:
            raise TypeError(f"Input element must in {HomogeneousLayer|PatternedLayer|MediumSpec}, but got {type(element)}")
        
    def _homogeneous(self, layer: HomogeneousLayer):
        medium = layer.medium
        d = layer.thickness
        if isinstance(medium, IsotropicMediumSpec):
            lam, V = homogeneous_modes(medium.eps, self.kx, self.ky)
            W = V.eye_like()
        else:
            raise NotImplementedError("")
        S = S_layer(self.W0, self.V0, W, V, lam, d, self.wvl)
        return S
    
    def _patterned(self, layer: PatternedLayer):
        medium_solid = layer.medium_solid
        medium_void = layer.medium_void
        d = layer.thickness
        pattern = layer.pattern
        
        if isinstance(medium_solid, IsotropicMediumSpec) and isinstance(medium_void, IsotropicMediumSpec):
            eps_solid = medium_solid.eps #[N_wvl]
            eps_void = medium_void.eps #[N_wvl]
            eps_grid = eps_solid[:, None, None]*pattern[None, ...] + (1 - pattern[None, ...])*eps_void[:, None, None] #[N_wvl, Ny, Nx]
            
            P, Q = compute_isotropic(eps_grid, self.m_flat, self.n_flat, self.kx, self.ky, self.tvf)
            if self.config.solver.name is "eigsolver":
                lam, W, V = eigsolver(P, Q, self.config.solver.stable)
            else: 
                raise NotImplementedError
            S = S_layer(self.W0, self.V0, W, V, lam, d, self.wvl)
        else:
            raise NotImplementedError("")
        
        
    def _medium(self, medium: MediumSpec):
        pass
    