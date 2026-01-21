# src/model/source.py
# Source object that defines illumination conditions

from src.backend import Backend
from src.model.geometry.geometry import Lattice
from typing import Any, Tuple

class Source:
    def __init__(self, backend: Backend, wavelength: Any,
                 theta: Any, phi: Any):
        """
        Initialize the Source object with given parameters.
        Parameters
        ----------
        backend : Backend
            Computational backend to use.
        wavelength : Any
            Wavelength(s) of the source. Length units.
        theta : Any
            Incident angle theta in radians.
        phi : Any
            Incident angle phi in radians.
        """
        self.wavelength, self.theta, self.phi = self._init_validation(backend, 
                                                                         wavelength, 
                                                                         theta, phi)
        self.backend = backend
    
    @property
    def k0(self):
        return (2.0 * self.backend.pi) / self.wavelength
    
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
            Shape (Nw, Nt, Np) where Nw is number of wavelengths, Nt number of thetas, Np number of phis.
        """
        return compute_k0xy(self._backend,
                            self._wavelength,
                            self._theta,
                            self._phi,
                            n_inc,
                            reduced)
    
    def Kxy(self, n_inc: Any, M: int, N: int, lattice: Lattice):
        """
        Compute Kx, Ky matrices for the source parameters.
        
        Parameters
        ----------
        n_inc : Any
            Refractive index of the incident medium (real). 
            Scalar or array-like broadcastable to wavelength.
        M, N : int
            Number of harmonics along x and y.
        lattice : Lattice
            Lattice object defining the periodicity.
        
        Returns
        -------
        Kx, Ky : torch.Tensor
            Tensors of shape (Nw, Nt, Np, 2M+1, 2N+1) where Nw is number of wavelengths.
        """
        k0x, k0y = self.k0xy(n_inc, reduced=True)
        Lx, Ly = lattice.period
        
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
        wvl_shape = wavelength.shape
        if len(wvl_shape) != 0 and len(wvl_shape) != 1:
            raise ValueError("wavelength must be a scalar or 1D array")
        if backend.any(wavelength <= 0):
            raise ValueError("wavelength values must be positive")
        if len(wvl_shape) == 0:
            wavelength = backend.reshape(wavelength, (-1,))  # ensure 1D
            
        theta = backend.asarray(theta, complex=False)
        theta_shape = theta.shape
        if len(theta_shape) != 0 and len(theta_shape) != 1:
            raise ValueError("theta must be a scalar or 1D array")
        if len(theta_shape) == 0:
            theta = backend.reshape(theta, (-1,))  # ensure 1D
            
        phi = backend.asarray(phi, complex=False)
        phi_shape = phi.shape
        if len(phi_shape) != 0 and len(phi_shape) != 1:
            raise ValueError("phi must be a scalar or 1D array")
        if len(phi_shape) == 0:
            phi = backend.reshape(phi, (-1,))  # ensure 1D
        
        return wavelength, theta, phi

""" Helper functions""" 
def compute_k0xy(
    backend: Backend,          
    wavelengths,      
    theta,            
    phi,
    n_inc,              
    reduced: bool = False 
    ) -> Tuple[Any, Any]:
    """
    Compute k0x, k0y for a batch of wavelengths and given incidence angles.

    Parameters
    ----------
    backend : Backend
        Computational backend (e.g., TorchBackend instance).

    wavelengths : array-like
        Wavelengths in the same units you want k0 in (e.g. micrometers).
        Shape can be (n_lambda,) or any broadcastable shape.

    theta : array-like or scalar
        Polar angle from +z (normal incidence = 0).
        In radians.
        
    phi : array-like or scalar
        Azimuthal angle in the xy-plane (0 along +x).
        In radians.
        
    n_inc : array-like or scalar
        Refractive index of the incident medium.
        Can be scalar or broadcastable to `wavelengths`.
        Imag part is ignored.
        
    reduced : bool, optional
        If True, compute reduced wavevectors (divided by k0). Default is False.

    Returns
    -------
    k0x, k0y : torch.Tensor
        Tensors of shape (Nw, Nt, Np) where Nw is number of wavelengths,
        real-valued, on backend.device, with backend.dtype.
    """
    if isinstance(backend, Backend) is False:
        raise TypeError("backend must be an instance of Backend.")
    
    # Convert all inputs to backend tensors
    lam   = backend.asarray(wavelengths, complex=False)
    th    = backend.asarray(theta,       complex=False)
    ph    = backend.asarray(phi,         complex=False)
    n_inc = backend.asarray(n_inc,       complex=False)
    
    lam_shape = lam.shape
    n_shape   = n_inc.shape
    th_shape  = th.shape
    ph_shape  = ph.shape
    # --- Minimal sanity: wavelengths, n_inc, th, phi must be scalar or 1D ---
    if len(lam_shape) > 1:
        raise ValueError(f"wavelengths must be scalar or 1D, got shape={lam_shape}")
    if len(n_shape) > 1:
        raise ValueError(f"n_inc must be scalar or 1D, got shape={n_shape}")
    if len(th_shape) > 1:
        raise ValueError(f"theta must be scalar or 1D, got shape={th_shape}")
    if len(ph_shape) > 1:
        raise ValueError(f"phi must be scalar or 1D, got shape={ph_shape}")

    # If both lam and n_inc are 1D, enforce same length
    if len(lam_shape) == 1 and len(n_shape) == 1 and lam_shape[0] != n_shape[0]:
        raise ValueError(
            f"wavelengths and n_inc must have same length when both are 1D, "
            f"got {lam_shape[0]} and {n_shape[0]}"
        )
    
    # Force them into [Nw,1,1], [1,Nt,1], [1,1,Np] so that
    # broadcasting gives [Nw,Nt,Np] for everything.
    lam = backend.reshape(lam, (-1, 1, 1))   # [Nw,1,1]
    th = backend.reshape(th, (1, -1, 1))     # [1,Nt,1]
    ph = backend.reshape(ph, (1, 1, -1))     # [1,1,Np]
    
    # Number of wavelengths after reshaping
    Nw = lam.shape[0]

    if len(n_shape) == 0:
        n_inc = backend.ones_like(lam) * n_inc
    elif len(n_shape) == 1:
        if n_shape[0] not in (1, Nw):
            raise ValueError(f"n_inc length must be 1 or {Nw}")
        n_inc = backend.reshape(n_inc, (-1, 1, 1))
        n_inc = backend.expand(n_inc, (Nw, 1, 1))

    # Direction cosines
    sin_th = backend.sin(th)
    cos_ph = backend.cos(ph)
    sin_ph = backend.sin(ph)

    # These are ALWAYS real-valued
    kx_dir = n_inc * sin_th * cos_ph
    ky_dir = n_inc * sin_th * sin_ph

    if reduced:
        # reduced wavevectors = direction cosines × refractive index
        return kx_dir, ky_dir

    # Free-space wavenumber
    k0 = (2.0 * backend.pi) / lam

    k0x = k0 * kx_dir
    k0y = k0 * ky_dir
        
    return k0x, k0y

def compute_Kxy(
    backend: Backend,
    kx0: Any,
    ky0: Any,
    Lx: Any,
    Ly: Any,
    M: int,
    N: int,
) -> Tuple[Any, Any]:
    """
    Construct Kx, Ky grids for RCWA harmonics:
        Kx = kx0 + 2* pi* m / Lx
        Ky = ky0 + 2* pi* n / Ly
    with m ∈ [-M..M], n ∈ [-N..N].

    Parameters
    ----------
    backend : Backend
        Computational backend (e.g., TorchBackend).

    kx0, ky0 : torch.Tensor
        Incident wavevector components along x and y in the superstrate,
        typically shape (n_lambda,). Must be real-valued.

    Lx, Ly : array-like or scalar
        Fundamental lattice periods along x and y. Usually scalars.
        Will be converted to backend real arrays and broadcast if needed.

    M, N : int
        Truncation orders along x and y. Total numbers are (2M+1) and (2N+1).

    Returns
    -------
    Kx, Ky : torch.Tensor
        Harmonic wavevector components with shape (Nw, Nt, Np, 2M+1, 2N+1),
        real-valued, on backend.device, with backend.dtype.
    """
    # Ensure base k components are backend-compatible and 1D
    kx0 = backend.asarray(kx0, complex=False)  # (B,)
    ky0 = backend.asarray(ky0, complex=False)  # (B,)

    if kx0.shape != ky0.shape:
        raise ValueError(f"kx0 and ky0 must have the same shape, got {ky0.shape} and {ky0.shape}")
    
    if len(kx0.shape) != 3:
        raise ValueError(
            f"kx0 and ky0 must have shape (Nw, Nt, Np); "
            f"got ndim={len(kx0.shape)}, shape={kx0.shape}"
        )

    Nw, Nt, Np = kx0.shape

    # Periods → reciprocal lattice vectors
    Lx_t = backend.asarray(Lx, complex=False)
    Ly_t = backend.asarray(Ly, complex=False)

    Gx = (2.0 * backend.pi / Lx_t)
    Gy = (2.0 * backend.pi / Ly_t)

    # Harmonic indices m ∈ [-M..M], n ∈ [-N..N]
    m = backend.arange(-M, M+1)
    n = backend.arange(-N, N+1)

    # Reshape for broadcasting:
    kx0 = backend.reshape(kx0, (Nw, Nt, Np, 1, 1))
    ky0 = backend.reshape(ky0, (Nw, Nt, Np, 1, 1))

    # m : (1, 1, 1, 2M+1, 1)
    m = backend.reshape(m, (1, 1, 1, 2 * M + 1, 1))
    # n : (1, 1, 1, 1, 2N+1)
    n = backend.reshape(n, (1, 1, 1, 1, 2 * N + 1))

    # Gx, Gy : (1, 1, 1, 1, 1) so they broadcast with everything
    Gx = backend.reshape(Gx, (1, 1, 1, 1, 1))
    Gy = backend.reshape(Gy, (1, 1, 1, 1, 1))

    # Compute "lines":
    Kx_line = kx0 + m * Gx       # (*batch_shape, 2M+1, 1)
    Ky_line = ky0 + n * Gy       # (*batch_shape, 1, 2N+1)

    # (*batch_shape, 2M+1, 2N+1)
    full_shape = (Nw, Nt, Np, 2 * M + 1, 2 * N + 1)
    Kx = backend.expand(Kx_line, full_shape)
    Ky = backend.expand(Ky_line, full_shape)

    return Kx, Ky