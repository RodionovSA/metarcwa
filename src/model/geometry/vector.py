# src/model/geometry/vector.py

from typing import Tuple
from src.model.geometry.base import BaseObject
from src.model.geometry.lattice import Lattice
from src.model.material import BaseMaterial
from src.backend import Backend
from src.model.geometry.sampling import matmap


class VectorObject(BaseObject):
    """
    Geometric vector object base class for defining material distributions.
    """
    def __init__(self,
                 center: Tuple[float, float], 
                 material: "BaseMaterial",
                 angle: float = 0.0,
                 soft_mask: bool = False,
                 smoothness: float = 0.05):
        '''
        Parameters
        ----------
        center : tuple of float
            (x,y) coordinates of the object's center in length units. (0, 0) is the center.
        material : BaseMaterial
            Material object defining the electromagnetic properties.
        angle : float
            Rotation angle in radians.
        soft_mask : bool
            Whether the object should use a soft mask for differentiable operations. Default is False.
            If True the bitmap representation will use smooth sigmoid approximation.
            *Important*: Fourier coefficients will still be computed analytically for sharp boundaries,
            so soft_mask only affects real-space distributions.
        smoothness : float
            Smoothness parameter for sigmoid. Default is 0.05.
        '''
        VectorObject._init_validation(center, 
                                    angle, 
                                    material,
                                    soft_mask,
                                    smoothness)
        
        
        self.material = material
        self.center = center
        self.angle = angle

        # Differentiability settings
        self.soft_mask = soft_mask
        self.smoothness = smoothness
    
    def bitmap(self, 
               backend: "Backend", 
               lattice: "Lattice"):
        pass
    
    def matmap_fourier(self, 
                       backend: "Backend", 
                       lattice: "Lattice",
                       matval: complex,
                       matbg: complex,
                       closed_form: bool):
        pass
    
    def epsilon_xy(self, 
                   backend: "Backend", 
                   lattice: "Lattice",
                   material_bg: "BaseMaterial"):
        '''
        Compute the real-space permittivity distribution.
        Parameters
        ----------
        backend : Backend
            Computational backend.
        lattice : Lattice
            Lattice object defining the simulation domain.
        material_bg : BaseMaterial
            Background material.
        Returns
        -------
        epsilon_xy : backend tensor
            Permittivity distribution tensor in real space, shape (wvl, 3, 3, Nx, Ny), complex dtype.
        '''
        return matmap(backend, 
                      self.bitmap(backend, lattice),
                      self.epsilon,
                      material_bg.epsilon_tensor)
    
    def mu_xy(self, 
              backend: "Backend",
              lattice: "Lattice", 
              material_bg: "BaseMaterial"):
        '''
        Compute the real-space permittivity distribution.
        Parameters
        ----------
        backend : Backend
            Computational backend.
        lattice : Lattice
            Lattice object defining the simulation domain.
        material_bg : BaseMaterial
            Background material.
        Returns
        -------
        mu_xy : backend tensor
            Permeability distribution in real space, shape (wvl, 3, 3, Nx, Ny), complex dtype.
        '''
        return matmap(backend, 
                      self.bitmap(backend, lattice),
                      self.mu,
                      material_bg.mu_tensor)
    
    def epsilon_mn(self,
                   backend: "Backend", 
                   lattice: "Lattice",
                   mat_bg: "BaseMaterial",
                   closed_form: bool = True,
                   inverse: bool = False,
                   regularized: bool = False,
                   regularization: float = 1e-8):
        '''
        Compute the Fourier coefficients of the permittivity distribution in the closed form.
        Parameters
        ----------
        backend : Backend
            Computational backend.
        lattice : Lattice
            Lattice object defining the simulation domain.
        material_bg : BaseMaterial
            Background material.
        inverse : bool
            Whether to compute Fourier coefficients for the inverse permittivity. 
            Default is False.
        regularized : bool
            Whether to apply regularization in inverse case. Default is False.
        regularization : float
            Regularization parameter. Default is 1e-8.
        Returns
        -------
        epsilon_mn : backend tensor
            Fourier coefficients epsilon_{m,n}, shape (wvl, 3, 3, 2M+1, 2N+1), complex.
        '''
        epsilon_tensor = self.epsilon      # (wvl, 3, 3)
        epsilonbg_tensor = mat_bg.epsilon_tensor  # (wvl, 3, 3)
        
        if inverse:
            if regularized:
                epsilon_eff = 1.0 / (epsilon_tensor + regularization)
                epsilon_eff_bg = 1.0 / (epsilonbg_tensor + regularization)
            else:
                epsilon_eff = 1.0 / epsilon_tensor
                epsilon_eff_bg = 1.0 / epsilonbg_tensor
            return self.matmap_fourier(backend, 
                                       lattice, 
                                       epsilon_eff, 
                                       epsilon_eff_bg,
                                       closed_form)
        
        return self.matmap_fourier(backend, 
                                   lattice, 
                                   epsilon_tensor, 
                                   epsilonbg_tensor,
                                   closed_form)
    
    def mu_mn(self,
              backend: "Backend", 
              lattice: "Lattice",
              mat_bg: "BaseMaterial",
              closed_form: bool = True,
              inverse: bool = False,
              regularized: bool = False,
              regularization: float = 1e-8):
        '''
        Compute the Fourier coefficients of the permeability distribution in the closed form.
        Parameters
        ----------
        mat_bg : BaseMaterial
            Background material.
        inverse : bool
            Whether to compute Fourier coefficients for the inverse permeability. 
            Default is False.
        regularized : bool
            Whether to apply regularization in inverse case. Default is False.
        regularization : float
            Regularization parameter. Default is 1e-8.
        Returns
        -------
        mu_mn : backend tensor
            Fourier coefficients mu_{m,n}, shape (wvl, 3, 3, 2M+1, 2N+1), complex.
        '''
        mu_tensor = self.mu      # (wvl, 3, 3)
        mubg_tensor = mat_bg.mu_tensor  # (wvl, 3, 3)
        
        if inverse:
            if regularized:
                mu_eff = 1.0 / (mu_tensor + regularization)
                mu_eff_bg = 1.0 / (mubg_tensor + regularization)
            else:
                mu_eff = 1.0 / mu_tensor
                mu_eff_bg = 1.0 / mubg_tensor
            return self.matmap_fourier(backend, 
                                       lattice, 
                                       mu_eff, 
                                       mu_eff_bg,
                                       closed_form)
        
        return self.matmap_fourier(backend, 
                                   lattice, 
                                   mu_tensor, 
                                   mubg_tensor,
                                   closed_form)
        
   
    @staticmethod
    def _init_validation(center: Tuple[float, float],
                        angle: float,
                        material: BaseMaterial,
                        soft_mask: bool,
                        smoothness: float) -> None:
        
        if len(center) != 2:
            raise ValueError(f"center must be tuple of 2 floats, got {center}")
        if not isinstance(angle, float) and not isinstance(angle, int):
            raise ValueError(f"angle must be a scalar float")
        if not isinstance(soft_mask, bool):
            raise TypeError(f"soft_mask must be bool, got {type(soft_mask)}")
        if not isinstance(smoothness, float) and not isinstance(smoothness, int):
            raise TypeError(f"smoothness must be float, got {type(smoothness)}")
        if smoothness <= 0:
            raise ValueError(f"smoothness must be positive, got {smoothness}")
        if not isinstance(material, BaseMaterial):
            raise TypeError("material must be a BaseMaterial instance")