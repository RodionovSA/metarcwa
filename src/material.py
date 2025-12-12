from abc import ABC, abstractmethod
from typing import Any
from src.backend import Backend

class BaseMaterial(ABC):
    """Abstract base class for all material types."""
    
    @property
    @abstractmethod
    def epsilon_tensor(self) -> Any:
        """Return wvlx3×3 permittivity tensor."""
        pass

    @property
    @abstractmethod
    def mu_tensor(self) -> Any:
        """Return wvlx3×3 permeability tensor."""
        pass
    
class Material(BaseMaterial):
    """Isotropic and non-magnetic material."""

    def __init__(self, epsilon: Any, backend: Backend):
        '''
        Parameters
        ----------
        backend : Backend
            Computational backend.
        epsilon : Any
            Electric permittivity.
        '''
        self._epsilon = Material._init_validation(backend, epsilon)
        self._backend = backend
        
    @property
    def epsilon(self) -> Any:
        return self._epsilon
    
    @property
    def mu(self) -> Any:
        return self.backend.ones_like(self._epsilon)
    
    @property
    def backend(self) -> Backend:
        return self._backend

    @property
    def epsilon_tensor(self) -> Any:
        """Return wvlx3×3 permittivity tensor."""
        return self.backend.xp.reshape(self._epsilon, (len(self._epsilon), 1, 1))*self.backend.eye(3)
    
    @property
    def mu_tensor(self) -> Any:
        """Return wvlx3×3 permeability tensor."""
        return self.backend.xp.reshape(self.backend.ones_like(self._epsilon), (len(self._epsilon), 1, 1))*self.backend.eye(3)
    
    @staticmethod
    def _init_validation(backend: Backend, 
                        epsilon: Any) -> None:
        
        if not isinstance(backend, Backend):
            raise TypeError("backend must be a Backend instance")
        
        # Convert to backend tensor
        eps_t = backend.asarray(epsilon, complex=True)
        
        if len(eps_t.shape) > 1:
            raise ValueError("epsilon must be a scalar or 1D tensor")
        if eps_t.shape == ():
            eps_t = backend.xp.reshape(eps_t, (1,))
        
        return eps_t
    
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