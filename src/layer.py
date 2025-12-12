from typing import Union, List, Tuple, Any
from src.geometry import VectorObject, Bitmap, VectorGroup
from src.compute import fft_matfunc
from src.material import BaseMaterial

    
class Layer:
    """
    Layer in RCWA structure.
    """
    def __init__(self, 
                 objects: Union[VectorObject, Bitmap, VectorGroup],
                 thickness: Any,
                 material_bg: BaseMaterial):
        '''
        Parameters
        ----------
        objects : Union[VectorObject, Bitmap, VectorGroup]
            Geometric object(s) defining the material distribution in the layer.
        thickness : Any
            Thickness of the layer.
        material_bg : BaseMaterial
            Background material of the layer.
        '''
        self._init_validation(objects, 
                              thickness, 
                              material_bg)
        self._objects = objects
        self._thickness = thickness
        self._material_bg = material_bg
    
    """ Simulation properties """
    @property
    def backend(self):
        return self._objects.backend
    @property
    def lattice(self):
        return self._objects.lattice
    @property
    def grid(self):
        return self._objects.lattice.grid
    @property
    def period(self):
        return self._objects.lattice.period
        
    """ Geometric properties """
    @property
    def objects(self):
        return self._objects
    @property
    def thickness(self):
        return self._thickness
    @property
    def bitmap(self):
        return self._objects.bitmap
    """ Material properties """
    @property
    def material(self):
        return self._objects.material
    @property
    def material_bg(self):
        return self._material_bg
    @property
    def epsilon(self):
        return self.material.epsilon_tensor
    @property
    def mu(self):
        return self.material.mu_tensor
    @property
    def epsilon_bg(self):
        return self.material_bg.epsilon_tensor
    @property
    def mu_bg(self):
        return self.material_bg.mu_tensor
    
    """ Get material distributions """
    def epsilon_xy(self):
        """
        Get permittivity distribution epsilon(x,y) in the layer.
        """
        return self._objects.epsilon_xy(self.material_bg)
    
    def mu_xy(self):
        """
        Get permeability distribution mu(x,y) in the layer.
        """
        return self._objects.mu_xy(self.material_bg)
    
    def epsilon_mn(self, M: int, N: int, use_closed_form: bool = True,
                   inverse: bool = False, regularized: bool = False,
                   regularization: float = 1e-8):
        """
        Get Fourier coefficients epsilon_{m,n} of permittivity in the layer.

        Parameters
        ----------
        M, N : int
            Number of harmonics along x and y.
        use_closed_form : bool, optional
            Whether to use closed-form expressions for Fourier coefficients. Default is True.
            Works only if objects is VectorObject or VectorGroup.
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
        if use_closed_form and isinstance(self._objects, (VectorObject, VectorGroup)):
            epsilon_mn = self._objects.epsilon_mn(
                M, N, self.material_bg, inverse=inverse, 
                regularized=regularized, regularization=regularization
            )
        else:
            epsilon_xy = self.epsilon_xy()
            if inverse:
                if regularized:
                    epsilon_xy = 1.0 / (epsilon_xy + regularization)
                else:
                    epsilon_xy = 1.0 / epsilon_xy
            epsilon_mn = fft_matfunc(self.backend, epsilon_xy, M, N)
        return epsilon_mn
    
    def mu_mn(self, M: int, N: int, use_closed_form: bool = True,
               inverse: bool = False, regularized: bool = False,
               regularization: float = 1e-8):
        """
        Get Fourier coefficients mu_{m,n} of permeability in the layer.

        Parameters
        ----------
        M, N : int
            Number of harmonics along x and y.
        use_closed_form : bool, optional
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
        if use_closed_form and isinstance(self._objects, (VectorObject, VectorGroup)):
            mu_mn = self._objects.mu_mn(
                M, N, self.material_bg, inverse=inverse, 
                regularized=regularized, regularization=regularization
            )
        else:
            mu_xy = self.mu_xy()
            if inverse:
                if regularized:
                    mu_xy = 1.0 / (mu_xy + regularization)
                else:
                    mu_xy = 1.0 / mu_xy
            mu_mn = fft_matfunc(self.backend, mu_xy, M, N)
        return mu_mn
    
    """ Static helper methods """
    @staticmethod
    def _init_validation(objects, thickness, material_bg):
        if not isinstance(objects, (VectorObject, Bitmap, VectorGroup)):
            raise ValueError(f"objects must be VectorObject, Bitmap, or VectorGroup, got {type(objects)}")
        if thickness <= 0:
            raise ValueError(f"thickness must be positive, got {thickness}")
        
        if not isinstance(material_bg, BaseMaterial):
            raise TypeError("material_bg must be a BaseMaterial instance")
        