# src/model/layer.py
# Layer objects that stores information about material properties and thickness

from typing import Union, Any
from src.model.geometry.base import BaseObject
from src.model.geometry.lattice import Lattice
from src.model.material import BaseMaterial
from src.backend import Backend

    
class Layer:
    """
    Layer in RCWA structure.
    """
    def __init__(self, 
                 objects: BaseObject,
                 thickness: Any,
                 material_bg: BaseMaterial):
        '''
        Parameters
        ----------
        objects : Union[VectorObject, Bitmap, VectorGroup]
            Geometric object(s) defining the material distribution in the layer.
        thickness : Any
            Thickness of the layer. if None, the layer is considered semi-infinite.
        material_bg : BaseMaterial
            Background material of the layer.
        '''
        self._init_validation(objects, 
                              thickness, 
                              material_bg)
        
        self.thickness = thickness
        self.material_bg = material_bg
        self.objects = objects 
        
    @property
    def type(self) -> str:
        if self.material.type == "isotropic" and self.material_bg.type == "isotropic":
            return "isotropic"
        raise NotImplementedError("Only isotropic layers are currently supported.")
    
    @property
    def is_magnetic(self) -> bool:
        if self.material.is_magnetic or self.material_bg.is_magnetic:
            return True
        return False
    
    def is_homogeneous(self, backend: "Backend", lattice: "Lattice") -> bool:
        epsilon_xy = backend.detach(self.epsilon_xy(backend, lattice))
        mu_xy = backend.detach(self.mu_xy(backend, lattice))
        
        epsilon_check = self.homogeneous_check(backend, epsilon_xy)
        mu_check = self.homogeneous_check(backend, mu_xy)
        
        return bool(epsilon_check) and bool(mu_check)
    
    @property
    def is_semi_infinite(self) -> bool:
        return self.thickness is None     
    
    @property
    def material(self):
        return self.objects.material 
    
    def bitmap(self, backend: "Backend", lattice: "Lattice") -> Any:
        """
        Get bitmap representation of the layer.
        """
        return self.objects.bitmap(backend, lattice)
    
    def epsilon_xy(self, backend: "Backend", lattice: "Lattice") -> Any:
        """
        Get permittivity distribution epsilon(x,y) in the layer.
        """
        return self.objects.epsilon_xy(backend, lattice, self.material_bg)
    
    def mu_xy(self, backend: "Backend", lattice: "Lattice") -> Any:
        """
        Get permeability distribution mu(x,y) in the layer.
        """
        return self.objects.mu_xy(backend, lattice, self.material_bg)
    
    def epsilon_mn(self, backend: "Backend", lattice: "Lattice",
                   closed_form: bool = True,
                   inverse: bool = False, regularized: bool = False,
                   regularization: float = 1e-8):
        """
        Get Fourier coefficients epsilon_{m,n} of permittivity in the layer.

        Parameters
        ----------
        backend : Backend
            Backend to use for computations.
        lattice : Lattice
            Lattice defining the periodicity and Fourier orders.
        closed_form : bool, optional
            Whether to use closed-form expressions for Fourier coefficients. Default is True.
            Works only if objects is Vector.
            Not compatible with subpixel averaging techniques. 
        inverse : bool, optional
            Whether to compute Fourier coefficients for the inverse permittivity. Default is False.
        regularized : bool, optional
            Whether to apply regularization in inverse case. Default is False.
        regularization : float, optional
            Regularization parameter. Default is 1e-8.
        Returns
        -------
        epsilon_mn : backend tensor
            Fourier coefficients epsilon_{m,n}, shape (wvl, 3, 3, 2M+1, 2N+1), complex.
        """
        epsilon_mn = self.objects.epsilon_mn(
            backend, lattice, self.material_bg, closed_form=closed_form,
            inverse=inverse, regularized=regularized, regularization=regularization
        )
        return epsilon_mn
    
    def mu_mn(self, backend: "Backend", lattice: "Lattice",
               closed_form: bool = True,
               inverse: bool = False, regularized: bool = False,
               regularization: float = 1e-8):
        """
        Get Fourier coefficients mu_{m,n} of permeability in the layer.

        Parameters
        ----------
        backend : Backend
            Backend to use for computations.
        lattice : Lattice
            Lattice defining the periodicity and Fourier orders.
        closed_form : bool, optional
            Whether to use closed-form expressions for Fourier coefficients. Default is True.
            Works only if objects is VectorObject or VectorGroup.
            Not compatible with subpixel averaging techniques. 
        inverse : bool, optional
            Whether to compute Fourier coefficients for the inverse permeability. Default is False.
        regularized : bool, optional
            Whether to apply regularization in inverse case. Default is False.
        regularization : float, optional
            Regularization parameter. Default is 1e-8.
        Returns
        -------
        mu_mn : backend tensor
            Fourier coefficients mu_{m,n}, shape (wvl, 3, 3, 2M+1, 2N+1), complex.
        """
        mu_mn = self.objects.mu_mn(
            backend, lattice, self.material_bg, closed_form=closed_form,
            inverse=inverse, regularized=regularized, regularization=regularization
        )
        return mu_mn
    
    """ Static helper methods """
    @staticmethod
    def _init_validation(objects, thickness, material_bg):
        if not isinstance(objects, BaseObject):
            raise ValueError(f"objects must be BaseObject instance, got {type(objects)}")
        if thickness is not None and thickness <= 0:
            raise ValueError(f"thickness must be positive, got {thickness}")
        
        if not isinstance(material_bg, BaseMaterial):
            raise TypeError("material_bg must be a BaseMaterial instance")
        
        
    @staticmethod
    def homogeneous_check(backend: Backend, tensor: Any) -> bool:
        tensor_ref = tensor[...,0,0]
        
        # Broadcast
        tensor_ref_broadcast = backend.reshape(tensor_ref, [*tensor_ref.shape, 1, 1])
        tensor_ref_broadcast = backend.expand(tensor_ref_broadcast, tensor.shape)
        
        #Difference
        diff = backend.abs(tensor_ref_broadcast - tensor)
        
        #Error
        err = backend.max(diff, dim=-1).values
        err = backend.max(err,  dim=-1).values
        
        return backend.all(err < 1e-12)
        