from typing import Tuple, Any

from src.backend import Backend

class Canvas:
    """
    Canvas base class for geometric objects.
    """
    def __init__(self, period: Tuple[float, float],
                 grid: Tuple[int, int]):
        '''
        Parameters
        ----------
        period : tuple of float
            (Lx, Ly) period of the unit cell.
        grid : tuple of int
            (Nx, Ny) grid size for sampling the material functions.
        '''
        Canvas._init_validation(period, grid)
        
        self._period = period
        self._grid = grid
        
    @property
    def period(self) -> Tuple[float, float]:
        return self._period

    @property
    def grid(self) -> Tuple[int, int]:
        return self._grid
    
    @staticmethod
    def _init_validation(period, grid) -> None:
        '''
        Validate period and grid.
        Parameters
        ----------
        period : tuple of float
            (Lx, Ly) period of the unit cell.
        grid : tuple of int
            (Nx, Ny) grid size for sampling the material functions.
        '''
        if len(period) != 2:
            raise ValueError(f"period must be tuple of 2 floats, got {period}")
        if len(grid) != 2:
            raise ValueError(f"grid must be tuple of 2 ints, got {grid}")
        if not all(isinstance(x, int) for x in grid):
            raise ValueError(f"grid values must be integers, got {grid}")
        if not all(isinstance(x, float) or isinstance(x, int) for x in period):
            raise ValueError(f"period values must be floats, got {period}")
        if any(x <= 0 for x in period):
            raise ValueError(f"period values must be positive, got {period}")
        if any(x <= 0 for x in grid):
            raise ValueError(f"grid values must be positive, got {grid}")

class Object:
    '''
    Geometric object base class for defining material distributions.
    '''
    def __init__(self,
                 backend: Backend,
                 canvas: Canvas,
                 center: Tuple[float, float], 
                 size: Tuple[float, float],
                 epsilon: Any,
                 mu: Any = 1.0,
                 angle: float = 0.0):
        '''
        Parameters
        ----------
        backend : Backend
            Computational backend.
        canvas : Canvas
            Canvas object defining the simulation domain.
        center : tuple of float
            (x,y) coordinates of the object's center (in the range [-Lx/2, Lx/2] x [-Ly/2, Ly/2]).
            If center is outside this range, it will be wrapped into the principal cell.
        size : tuple of float
            (width, height) size of the object.
        epsilon : Any
            Electric permittivity.
        mu : Any
            Magnetic permeability. Default is 1.0.
        angle : float
            Rotation angle in radians.
        '''
        Object._init_validation(backend, canvas, center, size, angle)
        
        self._backend = backend
        self._canvas = canvas
        
        self._center = Object._wrap_center(center[0], center[1], 
                                           self.canvas.period[0], 
                                           self.canvas.period[1])
        self._center = backend.asarray(self._center, complex=False)
        self._size = backend.asarray(size, complex=False)
        self._angle = backend.asarray(angle, complex=False)
        
        self.epsilon = backend.asarray(epsilon, complex=True)
        self.mu = backend.asarray(mu, complex=True)
        
    @property
    def backend(self) -> Backend:
        return self._backend

    @property
    def canvas(self) -> Canvas:
        return self._canvas
    
    @property
    def center(self) -> Tuple[float, float]:
        return self._center
    
    @center.setter
    def center(self, value: Tuple[float, float]) -> None:
        if len(value) != 2:
            raise ValueError(f"center must be tuple of 2 floats, got {value}")
        if not all(isinstance(x, float) or isinstance(x, int) for x in value):
            raise ValueError(f"center values must be floats, got {value}")
        
        self._center = Object._wrap_center(value[0], value[1], 
                                           self.canvas.period[0], 
                                           self.canvas.period[1])
        self._center = self.backend.asarray(self._center, complex=False)

    @property
    def size(self) -> Tuple[float, float]:
        return self._size
    
    @size.setter
    def size(self, value: Tuple[float, float]) -> None:
        if len(value) != 2:
            raise ValueError(f"size must be tuple of 2 floats, got {value}")
        if not all(isinstance(x, float) or isinstance(x, int) for x in value):
            raise ValueError(f"size values must be floats, got {value}")
        if any(s <= 0 for s in value):
            raise ValueError(f"size values must be positive, got {value}")
        
        self._size = self.backend.asarray(value, complex=False)

    @property
    def angle(self) -> float:
        return self._angle
    
    @angle.setter
    def angle(self, value: float) -> None:
        if not isinstance(value, float) and not isinstance(value, int):
            raise ValueError(f"angle must be a float, got {value}")
        self._angle = self.backend.asarray(value, complex=False)
    
    def epsilon_xy(self, epsilonbg: Any = 1.0):
        '''
        Compute the real-space permittivity distribution.
        Parameters
        ----------
        epsilonbg : Any
            Background permittivity. Default is 1.0.
        Returns
        -------
        epsilon_xy : backend tensor
            Permittivity distribution in real space, shape (Nx, Ny), complex dtype.
        '''
        return self._matdist_real(self.epsilon, 
                                  epsilonbg)
    
    def mu_xy(self, mubg: Any = 1.0):
        '''
        Compute the real-space permittivity distribution.
        Parameters
        ----------
        mubg : Any
            Background permeability. Default is 1.0.
        Returns
        -------
        mu_xy : backend tensor
            Permeability distribution in real space, shape (Nx, Ny), complex dtype.
        '''
        return self._matdist_real(self.mu, 
                                  mubg)
    
    def epsilon_mn(self,
                   M: int,
                   N: int,
                   epsilonbg: Any = 1.0):
        '''
        Compute the Fourier coefficients of the permittivity distribution analytically.
        Parameters
        ----------
        M, N : int
            Number of harmonics along x and y.
        epsilonbg : Any
            Background permittivity. Default is 1.0.
        Returns
        -------
        epsilon_mn : backend tensor
            Fourier coefficients epsilon_{m,n}, shape (2M+1, 2N+1), complex.
        '''
        return self._matdist_fourier(M, N, 
                                     self.epsilon, epsilonbg)
    
    def mu_mn(self,
               M: int,
               N: int,
               mubg: Any = 1.0):
        '''
        Compute the Fourier coefficients of the permeability distribution analytically.
        Parameters
        ----------
        M, N : int
            Number of harmonics along x and y.
        mubg : Any
            Background permeability. Default is 1.0.
        Returns
        -------
        mu_mn : backend tensor
            Fourier coefficients mu_{m,n}, shape (2M+1, 2N+1), complex.
        '''
        return self._matdist_fourier(M, N, 
                                     self.mu, mubg)
        
    def _matdist_real(self, 
                    matval: Any,
                    matbg: Any):
        '''
        Compute the real-space material distribution for the object.

        Parameters
        ----------
        matval : Any
            Material value inside the object.
        matbg : Any
            Background material value.

        Returns
        -------
        matdist_xy : backend tensor
            Material distribution in real space, shape (Nx, Ny), complex dtype.
        '''
        Nx, Ny = self.canvas.grid

        mask_c = self.bitmap          # (Nx, Ny), bool
        mask_c = self.backend.reshape(mask_c, (1, Nx, Ny))  # (1, Nx, Ny)
        
        bg_b, val_b = self.adjustshapes(self.backend, matval, matbg)
        matdist_xy = self.backend.expand(bg_b, (bg_b.shape[0], Nx, Ny))

        # Δmat per batch
        delta_mat_b = val_b - bg_b                          # (B, 1, 1)

        # mat = bg + Δmat * mask
        matdist_xy = matdist_xy + delta_mat_b * mask_c      # (B, Nx, Ny)

        return matdist_xy
        
    @staticmethod
    def adjustshapes(backend: Backend, matval: Any, matbg: Any):
        '''
        Adjust shapes of matval and matbg to be compatible for broadcasting.
        Parameters
        ----------
        backend : Backend
            Computational backend.
        matval : Any
            Material value inside the object.
        matbg : Any
            Background material value.
        Returns
        -------
        bg_b : backend tensor
            Reshaped background material value, shape (B, 1, 1).
        val_b : backend tensor
            Reshaped material value inside the object, shape (B, 1, 1).
        '''
        bg = backend.asarray(matbg, complex=True)   # shape () or (B,)
        val = backend.asarray(matval, complex=True) # shape () or (B,)
        
        # Make sure shapes are compatible
        bg_shape = backend.shape(bg)
        val_shape = backend.shape(val)

        if len(bg_shape) == 0 and len(val_shape) == 0:
            # both scalars → B = 1
            B = 1
            bg_b = backend.reshape(bg, (B, 1, 1))     # (1,1,1)
            val_b = backend.reshape(val, (B, 1, 1))   # (1,1,1)
        elif len(bg_shape) == 1 and len(val_shape) == 1:
            if bg_shape[0] != val_shape[0]:
                raise ValueError(f"matbg and matval batch sizes differ: {bg_shape[0]} vs {val_shape[0]}")
            B = bg_shape[0]
            bg_b = backend.reshape(bg, (B, 1, 1))     # (B,1,1)
            val_b = backend.reshape(val, (B, 1, 1))   # (B,1,1)
        else:
            # one is scalar, one is 1D
            if len(bg_shape) == 0 and len(val_shape) == 1:
                B = val_shape[0]
                bg_b = backend.reshape(bg, (1, 1, 1))        # scalar
                bg_b = bg_b  # will broadcast over B
                val_b = backend.reshape(val, (B, 1, 1))
            elif len(bg_shape) == 1 and len(val_shape) == 0:
                B = bg_shape[0]
                val_b = backend.reshape(val, (1, 1, 1))
                val_b = val_b  # broadcast over B
                bg_b = backend.reshape(bg, (B, 1, 1))
            else:
                raise ValueError(
                    f"Unsupported shapes for matbg {bg_shape} and matval {val_shape}"
                )
                
        return bg_b, val_b
    
    @staticmethod
    def _init_validation(backend: Backend, 
                        canvas: Canvas,
                        center: Tuple[float, float],
                        size: Tuple[float, float],
                        angle: float) -> None:
        
        if not isinstance(backend, Backend):
            raise TypeError("backend must be a Backend instance")
        if not isinstance(canvas, Canvas):
            raise TypeError("canvas must be a Canvas instance")
        if len(center) != 2:
            raise ValueError(f"center must be tuple of 2 floats, got {center}")
        if not all(isinstance(c, (int, float)) for c in center):
            raise TypeError(f"center values must be floats, got {center}")
        if len(size) != 2:
            raise ValueError(f"size must be tuple of 2 floats, got {size}")
        if not all(isinstance(s, (int, float)) for s in size):
            raise TypeError(f"size values must be floats, got {size}")
        if any(s <= 0 for s in size):
            raise ValueError(f"size values must be positive, got {size}")
        if not isinstance(angle, (int, float)):
            raise TypeError(f"angle must be a float, got {type(angle)}")
        
    @staticmethod
    def _wrap_center(cx, cy, Lx, Ly):
        """
        Map center (cx, cy) to the principal cell [-Lx/2, Lx/2] x [-Ly/2, Ly/2]
        using periodicity.
        """

        cx_wrapped = ((cx + Lx / 2.0) % Lx) - Lx / 2.0
        cy_wrapped = ((cy + Ly / 2.0) % Ly) - Ly / 2.0
        return cx_wrapped, cy_wrapped 
        
        
class Rectangle(Object):
    """
    Rectangle geometric object.
    """
    @property 
    def bitmap(self) -> Any:
        '''
        Convert the rectangle to a bitmap representation on the specified grid.
        
        Parameters
        ----------
        Returns
        -------
        bitmap : backend tensor
            Bitmap representation of the rectangle, shape (Nx, Ny), dtype bool.
        '''
        
        Lx, Ly = self.canvas.period
        Nx, Ny = self.canvas.grid
        
        # Coordinate grid 
        ix = self.backend.arange(0, Nx)
        iy = self.backend.arange(0, Ny)
        x = ix * (Lx / Nx)
        y = iy * (Ly / Ny)
        X, Y = self.backend.meshgrid(x, y, indexing='ij')

        w, h = self.size
        cx, cy = self.center
        
        # Shift center to [0, Lx] x [0, Ly] coordinates
        cx = cx + Lx / 2.0
        cy = cy + Ly / 2.0

        half_w = w / 2.0
        half_h = h / 2.0
        
        # ---------- ROTATED RECTANGLE MASK ----------
        # angle in radians (scalar → backend tensor)
        theta = self.backend.asarray(self.angle, complex=False)
        cos_t = self.backend.cos(theta)
        sin_t = self.backend.sin(theta)

        # coordinates relative to center
        dx = X - cx    # (Nx, Ny)
        dy = Y - cy    # (Nx, Ny)
        
        Lx_t = self.backend.asarray(Lx, complex=False)
        Ly_t = self.backend.asarray(Ly, complex=False)  
        
        dx_p = self.backend.mod(dx + Lx_t/2, Lx_t) - Lx_t/2
        dy_p = self.backend.mod(dy + Ly_t/2, Ly_t) - Ly_t/2

        # rotate grid into rectangle's local frame
        # (so rectangle is axis-aligned in (xr, yr))
        xr = dx_p * cos_t + dy_p * sin_t
        yr = -dx_p * sin_t + dy_p * cos_t

        # mask in local coordinates
        ax = self.backend.abs(xr)
        ay = self.backend.abs(yr)
        mask = (ax <= half_w) & (ay <= half_h)            # (Nx, Ny), bool
        # --------------------------------------------

        return self.backend.asarray(mask, complex=False)

    def _matdist_fourier(self,
                        M: int,
                        N: int,
                        matval: complex,
                        matbg: complex):
        """
        Analytic Fourier coefficients matfunc_{m,n} for a single rectangle
        in a periodic cell.

        Uses the standard formula:
            Δmat_{mn} = Δmat * (w*h / (Lx*Ly))
                         * sinc(G_m w/2) * sinc(G_n h/2)
                         * exp(-j(G_m x_c + G_n y_c))

        and adds the background term matbg * δ_{m0} δ_{n0}.

        Parameters
        ----------
        M, N : int
            Number of harmonics along x and y.
        matval : complex
            Material value inside the rectangle.
        matbg : complex
            Background material value.

        Returns
        -------
        mat_mn : backend tensor
            Fourier coefficients mat_{m,n}, shape (2M+1, 2N+1), complex.
            Indices correspond to m ∈ [-M..M], n ∈ [-N..N].
        """
        Lx, Ly = self.canvas.period
        w, h = self.size
        cx, cy = self.center
        
        # Shift center to [0, Lx] x [0, Ly] coordinates
        cx = cx + Lx / 2.0
        cy = cy + Ly / 2.0
        
        bg_b, val_b = self.adjustshapes(self.backend,matval, matbg)

        # Material contrast
        delta_mat = val_b - bg_b   # (B, 1, 1)

        # Harmonic indices m, n
        m = self.backend.arange(-M, M + 1)   # (2M+1,)
        n = self.backend.arange(-N, N + 1)   # (2N+1,)
        # Reciprocal lattice vectors
        # G_m = 2π m / Lx, G_n = 2π n / Ly
        two_pi = 2.0 * self.backend.pi

        Gm = (two_pi / Lx) * m   # (2M+1,)
        Gn = (two_pi / Ly) * n   # (2N+1,)

        # 2D grids for Gm, Gn
        Gm_grid, Gn_grid = self.backend.meshgrid(Gm, Gn, indexing='ij')  # (2M+1, 2N+1)

        # Helper: sinc(z) = sin(z)/z with safe handling of z=0
        def sinc(z):
            """
            Unnormalized sinc: sin(z) / z, with sinc(0) = 1.
            """
            z_abs = self.backend.abs(z)
            one = self.backend.ones_like(z)
            # Avoid division by zero
            z_safe = self.backend.where(z_abs < 1e-14, one, z)
            s = self.backend.sin(z) / z_safe
            # Enforce limit sinc(0) = 1
            s = self.backend.where(z_abs < 1e-14, one, s)
            return s

        # --- Rotation: project (Gx, Gy) into local rectangle axes (u,v) ---
        # angle in radians
        theta = self.backend.asarray(self.angle, complex=False)
        cos_t = self.backend.cos(theta)
        sin_t = self.backend.sin(theta)

        # k_u, k_v in local frame
        ku = Gm_grid * cos_t + Gn_grid * sin_t          # (2M+1, 2N+1)
        kv = -Gm_grid * sin_t + Gn_grid * cos_t         # (2M+1, 2N+1)

        # Sinc factors in local coordinates
        zx = ku * (w / 2.0)
        zy = kv * (h / 2.0)

        Sx = sinc(zx)
        Sy = sinc(zy)

        # Phase factor exp(-j(Gm x_c + Gn y_c))
        phase_arg = Gm_grid * cx + Gn_grid * cy
        phase = self.backend.exp(-1j * phase_arg)

        # Contrast contribution
        area_factor = (w * h) / (Lx * Ly)

        core = area_factor * Sx * Sy * phase          # (2M+1, 2N+1)
        core = self.backend.reshape(core, (1, 2*M + 1, 2*N + 1))  # (1, 2M+1, 2N+1)
        
        delta_mat_mn = delta_mat * area_factor * Sx * Sy * phase  # (2M+1, 2N+1)

        # Initialize with contrast term
        mat_mn = delta_mat_mn

        # --- Add background at (m=0,n=0), i.e. index (M,N) ---

        # bg_b: (B,1,1) or (1,1,1); broadcast to (B,)
        # We add bg_b to mat_mn[:, M, N]
        # For Torch/NumPy this indexing is fine:
        if hasattr(mat_mn, "__setitem__"):
            # mat_{00} = matbg + Δmat * fill_fraction (already in mat_mn[:, M, N])
            mat_mn[:, M, N] = mat_mn[:, M, N] + self.backend.reshape(bg_b, (-1,))
        else:
            # if it is a different backend without in-place assignment,
            # should implement a functional update here.
            raise NotImplementedError("In-place assignment not supported for this backend.")

        return mat_mn
        
        
class Bitmap:
    """Material distribution defined from 2D bitmap masks for epsilon and mu."""
    def __init__(self,
                  backend: Backend,
                  canvas: Canvas,
                  bitmap: Any,
                  epsilon_value: Any,
                  epsilon_bg: Any = 1.0, 
                  mu_value: Any = 1.0,
                  mu_bg: Any = 1.0):
        """
        Parameters
        ----------
        backend : Backend
            Computational backend.

        canvas : Canvas
            Canvas object defining the simulation domain.

        bitmap : array-like or backend tensor
            2D array representing the bitmap of material distribution.
            Must be 0/1 or False/True. 
            0 → epsilon_bg, mu_bg, 1 → epsilon_value, mu_value.

        epsilon_value : Any
            Permittivity inside pixels where bitmap == 1.

        epsilon_bg : Any, optional
            Background permittivity (default is 1.0).

        mu_value : Any, optional
            Permeability inside pixels where bitmap == 1 (default is 1.0).

        mu_bg : Any, optional
            Background permeability (default is 1.0).
        """
        Bitmap._init_validation(backend, canvas)
        
        self._backend = backend

        # --- Convert bitmap to backend real tensors ---
        bm = backend.asarray(bitmap, complex=False)

        # --- Basic shape check (2D) ---
        shape = backend.shape(bm)
        if len(shape) != 2:
            raise ValueError(f"bitmap must be 2D, got shape {shape}")
        
        # Setup canvas
        canvas._grid = shape  # enforce grid from bitmap shape
        self._canvas = canvas

        # --- Force it to be strict 0/1 mask ---
        # Treat anything != 0 as 1 (so True also maps to 1).
        zero = backend.zeros_like(bm)
        one = backend.ones_like(bm)
        bm_bin = backend.where(bm != zero, one, zero)

        self._bitmap = bm_bin   # (Nx, Ny), real, values ∈ {0,1}
        
        # Store material values
        self.epsilon_value = backend.asarray(epsilon_value, complex=True)
        self.epsilon_bg = backend.asarray(epsilon_bg, complex=True)

        self.mu_value = backend.asarray(mu_value, complex=True)
        self.mu_bg = backend.asarray(mu_bg, complex=True)
    
    @property
    def backend(self) -> Backend:
        return self._backend
    @property
    def canvas(self) -> Canvas:
        return self._canvas
    @property    
    def period(self) -> Tuple[float, float]:
        return self._canvas.period
    @property
    def grid(self) -> Tuple[int, int]:
        return self._bitmap.shape
    @property
    def bitmap(self) -> Any:
        return self._bitmap
    
    @property
    def epsilon_xy(self) -> Any:
        """
        Compute the real-space permittivity distribution.
        
        Returns
        -------
        epsilon_xy : backend tensor
            Permittivity distribution in real space, shape (Nx, Ny), complex dtype.   
        """
        bg_b, val_b = Object.adjustshapes(self.backend, 
                                        self.epsilon_value, 
                                        self.epsilon_bg)
        
        epsilon_xy = self.bitmap * (val_b - bg_b) + bg_b
        epsilon_xy = self.backend.asarray(epsilon_xy, complex=True)
        return epsilon_xy
    
    @property
    def mu_xy(self) -> Any:
        """
        Compute the real-space permeability distribution.
        
        Returns
        -------
        mu_xy : backend tensor
            Permeability distribution in real space, shape (Nx, Ny), complex dtype.   
        """
        bg_b, val_b = Object.adjustshapes(self.backend, 
                                        self.mu_value, 
                                        self.mu_bg)
        
        mu_xy = self.bitmap * (val_b - bg_b) + bg_b
        mu_xy = self.backend.asarray(mu_xy, complex=True)
        return mu_xy
    
    @staticmethod
    def _init_validation(backend: Backend, 
                        canvas: Canvas) -> None:
        if not isinstance(backend, Backend):
            raise TypeError("backend must be a Backend instance")
        if not isinstance(canvas, Canvas):
            raise TypeError("canvas must be a Canvas instance")