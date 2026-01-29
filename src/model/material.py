# src/model/material.py
# Material objects for different type of materials 

from abc import ABC, abstractmethod
from typing import Any, Sequence
from src.backend import Backend

class BaseMaterial(ABC):
    """Abstract base class for all material types."""
    
    @property
    @abstractmethod
    def type(self) -> str:
        """ Anisotropy type"""
        pass
    
    @property
    @abstractmethod
    def is_magnetic(self) -> bool:
        pass
    
    @abstractmethod
    def epsilon_tensor(self, backend: "Backend") -> Any:
        """Return wvlx3×3 permittivity tensor."""
        pass

    @abstractmethod
    def mu_tensor(self, backend: "Backend") -> Any:
        """Return wvlx3×3 permeability tensor."""
        pass
    
class Material(BaseMaterial):
    """Isotropic and non-magnetic material."""

    def __init__(self, epsilon: Any):
        '''
        Parameters
        ----------
        epsilon : Any
            Electric permittivity.
        '''
        self.epsilon = Material._init_validation(epsilon)
        
    @property
    def type(self) -> str:
        return "isotropic"
    
    @property
    def is_magnetic(self) -> bool:
        return False
    
    @property
    def mu(self) -> Any:
        return 1.0

    def epsilon_tensor(self, backend: "Backend") -> Any:
        """Return wvlx1×1 permittivity tensor."""
        epsilon = backend.asarray(self.epsilon, complex=True)
        if len(epsilon.shape) == 0:
            # Scalar case
            epsilon = backend.reshape(epsilon, (1,))
        return backend.reshape(epsilon, (len(epsilon), 1, 1))
    
    def mu_tensor(self, backend: "Backend") -> Any:
        """Return wvlx1×1 permeability tensor."""
        mu = backend.ones_like(self.epsilon_tensor(backend))
        return backend.asarray(mu, complex=True)
    
    @staticmethod
    def _init_validation(epsilon):
        # Python scalar → make 1D later in backend
        if isinstance(epsilon, (int, float)):
            return epsilon

        # Tensor / array-like
        if hasattr(epsilon, "shape"):
            if len(epsilon.shape) == 0:
                # 0-D tensor → scalar, OK
                return epsilon
            if len(epsilon.shape) == 1:
                if len(epsilon) == 0:
                    raise ValueError("epsilon must have non-zero length")
                return epsilon
            raise ValueError("epsilon must be scalar or 1D")

        raise TypeError(
            "epsilon must be int, float, or array/tensor"
        )
    
class MagneticMaterial(BaseMaterial):
    """Isotropic and magnetic material.
    *TODO*: implement magnetic materials.
    """
    pass

class AnisotropicMaterial(BaseMaterial):
    """Anisotropic material.
    *TODO*: implement anisotropic materials.
    """
    pass