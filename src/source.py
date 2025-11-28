from src.backend import Backend
from typing import Any

class Source:
    def __init__(self, backend: Backend, wavelength: Any,
                 theta_inc: float, phi_inc: float,
                 polarization: str = 'TE'):
        pass
    
    @staticmethod
    def _init_validation(backend: Backend, wavelength: Any,
                         theta_inc: float, phi_inc: float,
                         polarization: str) -> None:
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
        
        
        
        if polarization not in ['TE', 'TM']:
            raise ValueError("polarization must be 'TE' or 'TM'")
        return wavelength, theta_inc, phi_inc, polarization