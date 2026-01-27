# src/model/source.py
# Source object that defines illumination conditions

from src.backend import Backend
from src.model.geometry.lattice import Lattice
from src.model.geometry.harmonics import build_harmonic_grid, elliptical_truncation_mask, flatten_Kxy
from typing import Any, Tuple

class Source:
    def __init__(self, wavelength: Any, theta: Any, phi: Any):
        """
        Initialize the Source object with given parameters.
        Parameters
        ----------
        wavelength : Any
            Wavelength(s) of the source. Length units.
        theta : Any
            Incident angle theta in radians.
        phi : Any
            Incident angle phi in radians.
        """
        self.wavelength = self._init_validation_wvl(wavelength)
        self.theta = self._init_validation_angle(theta)
        self.phi = self._init_validation_angle(phi)
        
    def plane_wave_field(self, 
                         backend: "Backend",
                         lattice: "Lattice",
                         n_inc: Any,
                         mn: Tuple[int, int],
                         s: complex,
                         p: complex) -> Tuple[Any, Any, Any, Any]:
        '''
        Output Fourier coefficients of the incident plane wave field.
        
        Parameters
        ----------
        backend : Backend
            Computational backend.
        lattice : Lattice
            Lattice object defining the periodicity.
        mn : Tuple[int, int]
            Incident wave order (m, n). If normal incidence, (0, 0).
        n_inc : Any
            Refractive index of the incident medium (real).
        s : complex
            S-polarized amplitude.
        p : complex
            P-polarized amplitude.
        Returns
        -------
        Ex, Ey, Hx, Hy: Any
            Fourier maps of the incident plane wave fields.
        '''
        k0x, k0y = self.k0xy(backend, n_inc, reduced=True)
        Kx, Ky = self.Kxy(backend, lattice, n_inc)  
        
        Ex, Ey, Hx, Hy = compute_pw_fields(backend,
                                           n_inc,
                                           k0x,k0y,
                                           s,p)
    
        Ex_map, Ey_map, Hx_map, Hy_map = compute_pw_field_maps(backend,
                                                               mn,
                                                               Ex,Ey,Hx,Hy,
                                                               Kx,Ky)
        return Ex_map, Ey_map, Hx_map, Hy_map
    
    def psi_vector_inc(self, backend: "Backend",
                        lattice: "Lattice",
                        n_inc: Any,
                        mn: Tuple[int, int],
                        s: complex,
                        p: complex,
                        circ_truncation: bool = False) -> Any:
        '''
        Compute the incident field state vector Psi_inc for RCWA.
        
        Parameters
        ----------
        backend : Backend
            Computational backend.
        lattice : Lattice
            Lattice object defining the periodicity.
        mn : Tuple[int, int]
            Incident wave order (m, n). If normal incidence, (0, 0).
        n_inc : Any
            Refractive index of the incident medium (real).
        s : complex
            S-polarized amplitude.
        p : complex
            P-polarized amplitude.
        circ_truncation : bool, optional
            Whether to use circular truncation of harmonics. Default is False.
        
        Returns
        -------
        psi_inc : Any
            Incident field state vector.
        '''
        # Generate maps
        Ex_map, Ey_map, Hx_map, Hy_map = self.plane_wave_field(backend,
                                                               lattice,
                                                               n_inc,
                                                               mn,
                                                               s,p)
        
        # Flatten the maps
        Ex_map_f, Ey_map_f, _ = flatten_Kxy(backend, Ex_map, Ey_map)
        Hx_map_f, Hy_map_f, _ = flatten_Kxy(backend, Hx_map, Hy_map)
        
        # Apply circular truncation if needed
        M = lattice.M
        N = lattice.N
        if circ_truncation:
            mx, ny = build_harmonic_grid(backend, M, N)
            mask = elliptical_truncation_mask(mx, ny, M, N)
        else:
            mask = backend.astype(backend.ones((2*M+1)*(2*N+1)), backend.bool)
            
        # Apply truncation mask
        Ex_map_t = Ex_map_f[..., mask]
        Ey_map_t = Ey_map_f[..., mask]
        Hx_map_t = Hx_map_f[..., mask]
        Hy_map_t = Hy_map_f[..., mask]
        
        # Construct Psi_inc vector
        psi_inc = backend.cat([Ex_map_t, Ey_map_t, Hx_map_t, Hy_map_t], dim=-1)
        return psi_inc
        
    def k0(self, backend: "Backend") -> Any:
        wavelength = backend.asarray(self.wavelength, complex=False)
        return (2.0 * backend.pi) / wavelength
    
    ''' Methods to compute k0x, k0y, Kx, Ky'''
    def k0xy(self, backend: "Backend", n_inc: Any, reduced: bool = False) -> Tuple[Any, Any]:
        """
        Compute k0x, k0y for the source parameters.
        
        Parameters
        ----------
        backend : Backend
            Computational backend.
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
        return compute_k0xy(backend,
                            self.wavelength,
                            self.theta,
                            self.phi,
                            n_inc,
                            reduced)
    
    def Kxy(self, backend: "Backend", lattice: Lattice, n_inc: Any) -> Tuple[Any, Any]:
        """
        Compute Kx, Ky matrices for the source parameters.
        
        Parameters
        ----------
        backend : Backend
            Computational backend.
        lattice : Lattice
            Lattice object defining the periodicity.
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
        k0x, k0y = self.k0xy(backend, n_inc, reduced=True)
        Lx, Ly = lattice.period
        
        Kx, Ky = compute_Kxy(backend, k0x, k0y, Lx, Ly, lattice.M, lattice.N)
        return Kx, Ky
    
    ''' Static helper methods '''
    @staticmethod
    def _init_validation_wvl(wavelength: Any) -> Any:
        '''
        Validate and initialize source parameters.
        '''
        # Python scalar
        if isinstance(wavelength, (int, float)):
            if wavelength <= 0:
                raise ValueError("wavelength must be positive")
            return wavelength

        # Array / tensor
        if hasattr(wavelength, "shape"):
            if len(wavelength.shape) > 1:
                raise ValueError("wavelength must be scalar or 1D")

            # value check 
            if (wavelength <= 0).any():
                raise ValueError("wavelength values must be positive")

            return wavelength

        raise TypeError(
            "wavelength must be int, float, or array/tensor"
        )
        
    @staticmethod
    def _init_validation_angle(angle: Any) -> Any:
        '''
        Validate and initialize source parameters.
        '''
        # Python scalar
        if isinstance(angle, (int, float)):
            return angle

        # Array / tensor
        if hasattr(angle, "shape"):
            if len(angle.shape) > 1:
                raise ValueError("angle must be scalar or 1D")
            return angle

        raise TypeError(
            "angle must be int, float, or array/tensor"
        )

""" Helper functions""" 
def compute_k0xy(
    backend: "Backend",          
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

def compute_pw_fields(backend: "Backend",
                      n_inc: Any,
                      k0x: Any,
                      k0y: Any,
                      s: complex,
                      p: complex) -> Tuple[Any, Any, Any, Any]:
    '''
    Compute plane wave field components for given k0x, k0y, s, p.

    Parameters
    ----------
    backend : Backend
        Computational backend.
    n_inc : Any
        Refractive index of the incident medium (real). Shape [wvl].
    k0x : Any
        x-component of the incident wavevector. Divided by k0. Shape [wvl, theta, phi].
    k0y : Any
        y-component of the incident wavevector. Divided by k0. SHape [wvl, theta, phi].
    s : complex
        S-polarized amplitude.
    p : complex
        P-polarized amplitude.
    Returns
    -------
    Ex, Ey, Hx, Hy : Any
        Field components of the incident plane wave. Shape [wvl, theta, phi].
    '''
    s = backend.asarray(s, complex=True)
    p = backend.asarray(p, complex=True)
    n_inc = backend.asarray(n_inc, complex=False)
    
    # Reshape n_inc to [wvl,1,1] 
    n_shape = n_inc.shape
    if len(n_shape) > 1:
        raise ValueError(f"n_inc must be scalar or 1D, got shape={n_shape}")
    
    if n_shape == ():
        n_inc = backend.reshape(n_inc, (1,))
        
    n_inc = backend.reshape(n_inc, (-1, 1, 1))  # [wvl,1,1]
    n_shape = n_inc.shape # updated shape after reshape
    
    # Validation
    s_shape = s.shape  
    p_shape = p.shape  
    k0x_shape = k0x.shape  
    k0y_shape = k0y.shape
    if len(s_shape) > 1:
        raise ValueError(f"s must be scalar or 1D, got shape={s_shape}")
    if len(p_shape) > 1:
        raise ValueError(f"p must be scalar or 1D, got shape={p_shape}")
    if len(k0x_shape) != 3:
        raise ValueError(f"k0x must be 3D, got shape={k0x_shape}")
    if len(k0y_shape) != 3:
        raise ValueError(f"k0y must be 3D, got shape={k0y_shape}")
    if k0x_shape[0] != n_shape[0]:
        raise ValueError(f"First dimension of k0x must match length of n_inc, got {k0x_shape[0]} and {n_shape[0]}")
    if k0y_shape[0] != n_shape[0]:
        raise ValueError(f"First dimension of k0y must match length of n_inc, got {k0y_shape[0]} and {n_shape[0]}")
    
    # Calculate unit vectors for s and p polarizations
    kt2 = k0x**2 + k0y**2
    kt  = backend.sqrt(kt2)
    
    kz2 = n_inc**2 - kt2
    kz = backend.sqrt(kz2) # longitudinal wavevector component
    
    normal = kt2 < backend.asarray(1e-9, complex=False)
    
    # s
    sx = -k0y / kt
    sy =  k0x / kt
    
    # p
    px = k0x * kz / (n_inc * kt)
    py = k0y * kz / (n_inc * kt)
    
    # Normal-incidence convention: s->y, p->x 
    ones  = backend.ones_like(k0x)
    zeros = backend.zeros_like(k0x)
    sx_ni, sy_ni = zeros, ones
    px_ni, py_ni = ones, zeros
    
    # Select correct basis
    sx = backend.where(normal, sx_ni, sx)
    sy = backend.where(normal, sy_ni, sy)

    px = backend.where(normal, px_ni, px)
    py = backend.where(normal, py_ni, py)
    
    # Field components
    Ex = s * sx + p * px # Projection from s,p to x,y
    Ey = s * sy + p * py
    Ez = -(p*kt / n_inc)
    
    Hx = -1j*(k0y*Ez - kz*Ey)
    Hy = -1j*(kz*Ex - k0x*Ez)
    
    return Ex, Ey, Hx, Hy

def compute_pw_field_maps(backend: "Backend",
                          mn: Tuple[int, int],
                          Ex: Any, Ey: Any, Hx: Any, Hy: Any,
                          Kx: Any, Ky: Any):
    '''
    Compute fourier maps of the plane wave fields over the harmonic grid.
    
    Parameters
    ----------
    backend : Backend
        Computational backend.
    mn : Tuple[int, int]
        Incident wave order (m, n). If normal incidence, (0, 0).
    Ex, Ey, Hx, Hy : Any
        Field components. Shape [wvl, theta, phi].
    Kx, Ky : Any
        Kx and Ky harmonic grids. Shape [wvl, theta, phi, 2M+1, 2N+1].
        
    Returns
    -------
    Ex_map, Ey_map, Hx_map, Hy_map : Any
        Fourier maps of the plane wave fields. Shape [wvl, theta, phi, 2M+1, 2N+1].
    '''
    m_inc, n_inc = mn
    M = (Kx.shape[-2] - 1) // 2
    N = (Kx.shape[-1] - 1) // 2
    
    if not (-M <= m_inc <= M):
        raise ValueError(f"m_inc={m_inc} out of bounds for M={M}")
    if not (-N <= n_inc <= N):
        raise ValueError(f"n_inc={n_inc} out of bounds for N={N}")
    if Kx.shape != Ky.shape:
        raise ValueError(f"Kx and Ky must have the same shape, got {Kx.shape} and {Ky.shape}")
    if Kx.shape[:-2] != Ex.shape:
        raise ValueError(f"Leading shape of Kx {Kx.shape[:-2]} must match shape of Ex {Ex.shape}")
    if Kx.shape[:-2] != Ey.shape:
        raise ValueError(f"Leading shape of Kx {Kx.shape[:-2]} must match shape of Ey {Ey.shape}")
    if Kx.shape[:-2] != Hx.shape:
        raise ValueError(f"Leading shape of Kx {Kx.shape[:-2]} must match shape of Hx {Hx.shape}")
    if Kx.shape[:-2] != Hy.shape:
        raise ValueError(f"Leading shape of Kx {Kx.shape[:-2]} must match shape of Hy {Hy.shape}")
    
    # Indices of the incident wave in the harmonic grid
    m_idx = m_inc + M
    n_idx = n_inc + N
    
    # Initialize maps with zeros
    shape = Kx.shape
    Ex_map = backend.asarray(backend.zeros(shape), complex=True)
    Ey_map = backend.asarray(backend.zeros(shape), complex=True)
    Hx_map = backend.asarray(backend.zeros(shape), complex=True)
    Hy_map = backend.asarray(backend.zeros(shape), complex=True)
    
    # Assign the incident wave components to the correct harmonic
    Ex_map[..., m_idx, n_idx] = Ex
    Ey_map[..., m_idx, n_idx] = Ey
    Hx_map[..., m_idx, n_idx] = Hx
    Hy_map[..., m_idx, n_idx] = Hy
    
    return Ex_map, Ey_map, Hx_map, Hy_map
    