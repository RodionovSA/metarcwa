# src/model/geometry/base.py

from typing import Tuple, Any
from abc import ABC, abstractmethod

from src.model.geometry.lattice import Lattice
from src.model.material import BaseMaterial
from src.backend import Backend

class BaseObject(ABC):
    """
    Base class for all geometry objects.
    """
    
    @abstractmethod
    def bitmap(self, backend: "Backend", lattice: "Lattice"):
        '''
        Compute bitmap representation of the object.
        '''
        pass
    
    @abstractmethod
    def matmap_fourier(self, 
                       backend: "Backend", 
                       lattice: "Lattice",
                       matval: complex,
                       matbg: complex,
                       closed_form: bool):
        '''
        Computes Fourier material map.
        '''
        pass
    
    @abstractmethod
    def epsilon_xy(self, 
                   backend: "Backend", 
                   material_bg: "BaseMaterial"):
        '''
        Compute the real-space permittivity distribution.
        '''
        pass
    
    @abstractmethod
    def mu_xy(self, 
              backend: "Backend", 
              material_bg: "BaseMaterial"):
        '''
        Compute the real-space permeability distribution.
        '''
        pass
    
    @abstractmethod
    def epsilon_mn(self,
                   backend: "Backend", 
                   lattice: "Lattice",
                   material_bg: "BaseMaterial",
                   closed_form: bool = True,
                   inverse: bool = False,
                   regularized: bool = False,
                   regularization: float = 1e-8):
        '''
        Compute the Fourier coefficients of the permittivity distribution.
        '''
        
    @abstractmethod
    def mu_mn(self,
              backend: "Backend", 
              lattice: "Lattice",
              material_bg: "BaseMaterial",
              closed_form: bool = True,
              inverse: bool = False,
              regularized: bool = False,
              regularization: float = 1e-8):
        '''
        Compute the Fourier coefficients of the permeability distribution.
        '''
    
    @property
    def epsilon(self) -> Any:
        return self.material.epsilon_tensor
    
    @property
    def mu(self) -> Any:
        return self.material.mu_tensor
    