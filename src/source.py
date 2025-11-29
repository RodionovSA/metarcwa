from src.backend import Backend
from src.geometry import Canvas
from src.kvector import compute_k0xy, compute_Kxy
from typing import Any

class Source:
    def __init__(self, backend: Backend, wavelength: Any,
                 theta: float, phi: float):
        """
        Initialize the Source object with given parameters.
        Parameters
        ----------
        backend : Backend
            Computational backend to use.
        wavelength : Any
            Wavelength(s) of the source. Length units.
        theta : float
            Incident angle theta in radians.
        phi : float
            Incident angle phi in radians.
        """
        self._wavelength, self._theta, self._phi = self._init_validation(backend, 
                                                                         wavelength, 
                                                                         theta, phi)
        self._backend = backend
        
    @property
    def backend(self):
        return self._backend
    
    @property
    def wavelength(self):
        return self._wavelength
    
    @property
    def theta(self):
        return self._theta
    
    @property
    def phi(self):
        return self._phi
    
    ''' Methods to compute k0x, k0y, Kx, Ky'''
    def k0xy(self, n_inc: Any, reduced: bool = False):
        """
        Compute k0x, k0y for the source parameters.
        
        Parameters
        ----------
        n_inc : Any
            Refractive index of the incident medium (real). 
            Scalar or array-like broadcastable to wavelength.
        reduced : bool, optional
            If True, compute reduced wavevectors (divided by k0). Default is False.
        
        Returns
        -------
        k0x, k0y : torch.Tensor
            Tensors of k0x and k0y values. 
            Shape (Nw, Nt, Np) where Nw is number of wavelengths.
        """
        return compute_k0xy(self._backend,
                            self._wavelength,
                            self._theta,
                            self._phi,
                            n_inc,
                            reduced)
    
    def Kxy(self, n_inc: Any, M: int, N: int, canvas: Canvas):
        """
        Compute Kx, Ky matrices for the source parameters.
        
        Parameters
        ----------
        n_inc : Any
            Refractive index of the incident medium (real). 
            Scalar or array-like broadcastable to wavelength.
        M, N : int
            Number of harmonics along x and y.
        
        Returns
        -------
        Kx, Ky : torch.Tensor
            Tensors of shape (Nw, Nt, Np, 2M+1, 2N+1) where Nw is number of wavelengths.
        """
        k0x, k0y = self.k0xy(n_inc, reduced=True)
        Lx, Ly = canvas.period
        
        Kx, Ky = compute_Kxy(self._backend, k0x, k0y, Lx, Ly, M, N)
        return Kx, Ky
    
    ''' Static helper methods '''
    @staticmethod
    def _init_validation(backend: Backend, wavelength: Any,
                         theta: float, phi: float) -> None:
        '''
        Validate and initialize source parameters.
        '''
        if not isinstance(backend, Backend):
            raise TypeError("backend must be an instance of Backend")
        
        wavelength = backend.asarray(wavelength, complex=False)
        wvl_shape = backend.shape(wavelength)
        if len(wvl_shape) != 0 and len(wvl_shape) != 1:
            raise ValueError("wavelength must be a scalar or 1D array")
        if backend.any(wavelength <= 0):
            raise ValueError("wavelength values must be positive")
        if len(wvl_shape) == 0:
            wavelength = backend.reshape(wavelength, (-1,))  # ensure 1D
            
        theta = backend.asarray(theta, complex=False)
        theta_shape = backend.shape(theta)
        if len(theta_shape) != 0 and len(theta_shape) != 1:
            raise ValueError("theta must be a scalar or 1D array")
        if len(theta_shape) == 0:
            theta = backend.reshape(theta, (-1,))  # ensure 1D
            
        phi = backend.asarray(phi, complex=False)
        phi_shape = backend.shape(phi)
        if len(phi_shape) != 0 and len(phi_shape) != 1:
            raise ValueError("phi must be a scalar or 1D array")
        if len(phi_shape) == 0:
            phi = backend.reshape(phi, (-1,))  # ensure 1D
        
        return wavelength, theta, phi