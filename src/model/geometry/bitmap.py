# src/model/geometry/bitmap.py
# Bitmap-based geometry objects.

from typing import Any
from src.model.geometry.base import BaseObject
from src.model.geometry.lattice import Lattice
from src.model.material import BaseMaterial
from src.backend import Backend
from src.model.geometry.sampling import matmap
from src.model.geometry.fourier import fft_matmap

class Bitmap(BaseObject):
    
    """Material distribution defined from 2D bitmap masks for a given material."""
    
    def __init__(self, bitmap: Any, material: "BaseMaterial"):
        """
        Parameters
        ----------
        bitmap : array-like or backend tensor
            2D array representing the material distribution.
            Values must lie in the range [0, 1].

            - bitmap == 0 corresponds to the background material
            (epsilon_bg, mu_bg).
            - bitmap == 1 corresponds to the foreground material
            (epsilon, mu).
            - 0 < bitmap < 1 represents a continuous mixture of
            background and foreground materials, interpreted
            via linear interpolation.
            
            **Important**: If bitmap grid does not match the lattice grid,
            it will be automatically resampled to fit during simulation.
            
        material : BaseMaterial
            Material object defining epsilon and mu inside pixels where bitmap == 1.
        """
        Bitmap._init_validation(bitmap, material)
        
        self._bitmap = bitmap  # (Nx, Ny), real, values ∈ {0,1}
        self.material = material
    
    def bitmap(self, backend: "Backend", lattice: "Lattice") -> Any:
        '''
        Align the provided bitmap with the specified backend and lattice.
        
        Parameters
        ----------
        backend : Backend
            Computational backend.
        lattice : Lattice
            Lattice object defining the simulation domain.
            
        Returns
        -------
        bitmap : backend tensor
            Aligned bitmap tensor of shape (Nx, Ny), real dtype.
        '''
        bitmap = backend.asarray(self._bitmap, complex=False)
        bitmap_new = backend.resample(bitmap, lattice.grid)
        
        return bitmap_new
    
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
                      self.material.epsilon_tensor(backend),
                      material_bg.epsilon_tensor(backend))
    
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
                      self.material.mu_tensor(backend),
                      material_bg.mu_tensor(backend))
    
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
        epsilon_tensor = self.material.epsilon_tensor(backend)      # (wvl, 3, 3)
        epsilonbg_tensor = mat_bg.epsilon_tensor(backend)  # (wvl, 3, 3)
        
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
        mu_tensor = self.material.mu_tensor(backend)      # (wvl, 3, 3)
        mubg_tensor = mat_bg.mu_tensor(backend)  # (wvl, 3, 3)
        
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
        
    def matmap_fourier(self, 
                       backend: "Backend", 
                       lattice: "Lattice",
                       matval: complex,
                       matbg: complex,
                       closed_form: bool):
        """
        Computes Fourier material map from the real-space distribution

        Parameters
        ----------
        backend : Backend
            Computational backend.
        lattice : Lattice
            Lattice object defining the simulation domain.
        matval : complex
            Material value tensor inside the rectangle (B, 3, 3).
        matbg : complex
            Background material value tensor (B, 3, 3).
        closed_form : bool
            Does not affect bitmap geometries, included for compatibility.

        Returns
        -------
        mat_mn : backend tensor
            Fourier coefficients mat_{m,n}, shape (B, 3, 3, 2M+1, 2N+1), complex.
            Indices correspond to m ∈ [-M..M], n ∈ [-N..N].
        """
        
        mat_xy = matmap(backend, 
                        self.bitmap(backend, lattice),
                        matval,
                        matbg)
        
        mat_mn = fft_matmap(backend,
                            mat_xy,
                            lattice.M,
                            lattice.N)

        return mat_mn
        
    """ Static helper methods """
    @staticmethod
    def _init_validation(bitmap: Any, material: "BaseMaterial") -> None:
        
        if not isinstance(material, BaseMaterial):
            raise TypeError("material must be a BaseMaterial instance")
        
        if not hasattr(bitmap, "shape"):
            raise TypeError("bitmap must be an array/tensor")

        if len(bitmap.shape) != 2:
            raise ValueError("bitmap must be a 2D array/tensor")
            
        # values must lie in [0, 1]
        vmin = bitmap.min()
        vmax = bitmap.max()

        if vmin < 0 or vmax > 1:
            raise ValueError("bitmap values must be in [0, 1]")