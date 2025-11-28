from typing import Union, List, Tuple, Any
from src.geometry import VectorObject, Bitmap, VectorGroup
from src.material_func_fourier import fft_matfunc

    
class Layer:
    """
    Layer in RCWA structure.
    """
    def __init__(self, 
                 objects: Union[VectorObject, Bitmap, VectorGroup],
                 thickness: Any,
                 epsilon_bg: Any = 1.0,
                 mu_bg: Any = 1.0):
        '''
        Parameters
        ----------
        objects : Union[VectorObject, Bitmap, VectorGroup]
            Geometric object(s) defining the material distribution in the layer.
        thickness : Any
            Thickness of the layer.
        epsilon_bg : Any
            Background permittivity. Default is 1.0.
        mu_bg : Any
            Background permeability. Default is 1.0.
        '''
        self._epsilon_bg, self._mu_bg = self._init_validation(objects, 
                                                              thickness, 
                                                              epsilon_bg, 
                                                              mu_bg)
        self._objects = objects
        self._thickness = thickness
    
    """ Simulation properties """
    @property
    def backend(self):
        return self._objects.backend
    @property
    def canvas(self):
        return self._objects.canvas
    @property
    def grid(self):
        return self._objects.canvas.grid
    @property
    def period(self):
        return self._objects.canvas.period
        
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
    def epsilon(self):
        return self._objects.epsilon
    @property
    def mu(self):
        return self._objects.mu
    @property
    def epsilon_bg(self):
        return self._epsilon_bg
    @property
    def mu_bg(self):
        return self._mu_bg
    
    """ Get material distributions """
    def epsilon_xy(self):
        """
        Get permittivity distribution epsilon(x,y) in the layer.
        """
        return self._objects.epsilon_xy(self._epsilon_bg)
    
    def mu_xy(self):
        """
        Get permeability distribution mu(x,y) in the layer.
        """
        return self._objects.mu_xy(self._mu_bg)
    
    def epsilon_mn(self, M: int, N: int, use_closed_form: bool = True):
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
        """
        if use_closed_form and isinstance(self._objects, (VectorObject, VectorGroup)):
            epsilon_mn = self._objects.epsilon_mn(
                M, N, self._epsilon_bg
            )
        else:
            epsilon_xy = self.epsilon_xy()
            epsilon_mn = fft_matfunc(self.backend, epsilon_xy, M, N)
        return epsilon_mn
    
    def mu_mn(self, M: int, N: int, use_closed_form: bool = True):
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
        """
        if use_closed_form and isinstance(self._objects, (VectorObject, VectorGroup)):
            mu_mn = self._objects.mu_mn(
                M, N, self._mu_bg
            )
        else:
            mu_xy = self.mu_xy()
            mu_mn = fft_matfunc(self.backend, mu_xy, M, N)
        return mu_mn
    
    """ Static helper methods """
    @staticmethod
    def _init_validation(objects, thickness, epsilon_bg, mu_bg):
        if not isinstance(objects, (VectorObject, Bitmap, VectorGroup)):
            raise ValueError(f"objects must be VectorObject, Bitmap, or VectorGroup, got {type(objects)}")
        if thickness <= 0:
            raise ValueError(f"thickness must be positive, got {thickness}")
        
        epsilon_bg_t,_ = VectorObject.adjustshapes(objects.backend, 
                                  objects.epsilon, 
                                  epsilon_bg)
        
        mu_bg_t,_ = VectorObject.adjustshapes(objects.backend, 
                                  objects.mu, 
                                  mu_bg)
        
        return epsilon_bg_t[:,0,0], mu_bg_t[:,0,0]
        