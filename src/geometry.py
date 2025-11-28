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

class VectorObject:
    """
    Geometric vector object base class for defining material distributions.
    """
    def __init__(self,
                 backend: Backend,
                 canvas: Canvas,
                 center: Tuple[float, float], 
                 epsilon: Any,
                 mu: Any = 1.0,
                 angle: float = 0.0,
                 soft_mask: bool = False,
                 smoothness: float = 0.05):
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
        epsilon : Any
            Electric permittivity.
        mu : Any
            Magnetic permeability. Default is 1.0.
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
        mu_t = VectorObject._init_validation(backend, 
                                       canvas, 
                                       center, 
                                       angle, 
                                       epsilon, 
                                       mu,
                                       soft_mask,
                                       smoothness)
        
        self._backend = backend
        self._canvas = canvas
        
        self._center = VectorObject._wrap_center(backend,
                                           center[0], center[1], 
                                           self.canvas.period[0], 
                                           self.canvas.period[1])
        self._angle = backend.asarray(angle, complex=False)

        # Differentiability settings
        self._soft_mask = soft_mask
        self._smoothness = smoothness
        
        self._epsilon = backend.asarray(epsilon, complex=True)
        self._mu = mu_t

    """ Simulation properties """
    @property
    def backend(self) -> Backend:
        return self._backend

    @property
    def canvas(self) -> Canvas:
        return self._canvas

    """ Autograd properties """
    @property
    def soft_mask(self) -> bool:
        return self._soft_mask

    @property
    def smoothness(self) -> float:
        return self._smoothness

    """ Geometric properties """
    @property
    def center(self) -> Tuple[float, float]:
        return self._center
    
    @center.setter
    def center(self, value: Tuple[float, float]) -> None:
        if len(value) != 2:
            raise ValueError(f"center must be tuple of 2 floats, got {value}")
        
        self._center = VectorObject._wrap_center(self.backend, 
                                           value[0], value[1], 
                                           self.canvas.period[0], 
                                           self.canvas.period[1])
        self._center = self.backend.asarray(self._center, complex=False)

    @property
    def angle(self) -> float:
        return self._angle
    
    @angle.setter
    def angle(self, value: float) -> None:
        if not isinstance(value, float) and not isinstance(value, int):
            raise ValueError(f"angle must be a float, got {value}")
        self._angle = self.backend.asarray(value, complex=False)
    
    """ Material properties """
    @property
    def epsilon(self) -> Any:
        return self._epsilon
    
    @property
    def mu(self) -> Any:
        return self._mu
    
    """ Calculate material distributions """
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
        Compute the Fourier coefficients of the permittivity distribution in the closed form.
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
        Compute the Fourier coefficients of the permeability distribution in the closed form.
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
    
    """ Static helper methods """
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
                bg_b = backend.expand(bg_b, (B, 1, 1))  # broadcast over B
                val_b = backend.reshape(val, (B, 1, 1))
            elif len(bg_shape) == 1 and len(val_shape) == 0:
                B = bg_shape[0]
                val_b = backend.reshape(val, (1, 1, 1))
                val_b = backend.expand(val_b, (B, 1, 1))  # broadcast over B
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
                        angle: float,
                        epsilon: Any,
                        mu: Any,
                        soft_mask: bool,
                        smoothness: float) -> None:
        
        if not isinstance(backend, Backend):
            raise TypeError("backend must be a Backend instance")
        if not isinstance(canvas, Canvas):
            raise TypeError("canvas must be a Canvas instance")
        if len(center) != 2:
            raise ValueError(f"center must be tuple of 2 floats, got {center}")
        if not isinstance(soft_mask, bool):
            raise TypeError(f"soft_mask must be bool, got {type(soft_mask)}")
        if not isinstance(smoothness, float) and not isinstance(smoothness, int):
            raise TypeError(f"smoothness must be float, got {type(smoothness)}")
        if smoothness <= 0:
            raise ValueError(f"smoothness must be positive, got {smoothness}")
        
        # Convert to backend tensors 
        eps_t = backend.asarray(epsilon, complex=True)
        eps_shape = backend.shape(eps_t)
        
        is_scalar_number = isinstance(mu, (int, float, complex))
        if is_scalar_number and (mu == 1 or mu == 1.0 or mu == 1+0j):
        # If epsilon is scalar → scalar mu is fine
        # If epsilon is batch → create matching batch of ones
            if len(eps_shape) == 0:
                mu_t = backend.asarray(1.0, complex=True)
            else:
                B = eps_shape[0]
                mu_t = backend.asarray(backend.ones((B,)), complex=True)
                
        else:
            mu_t = backend.asarray(mu, complex=True)

        mu_shape  = backend.shape(mu_t)

        # Only allow scalar or 1D for both
        if len(eps_shape) > 1:
            raise ValueError(
                f"epsilon must be scalar or 1D (batch), got shape {eps_shape}"
            )
        if len(mu_shape) > 1:
            raise ValueError(
                f"mu must be scalar or 1D (batch), got shape {mu_shape}"
            )

        # If both are 1D, batch sizes must match
        if len(eps_shape) == 1 and len(mu_shape) == 1:
            if eps_shape[0] != mu_shape[0]:
                raise ValueError(
                    f"epsilon and mu batch sizes differ: "
                    f"{eps_shape[0]} vs {mu_shape[0]}"
                )
        return mu_t
        
    @staticmethod
    def _wrap_center(backend, cx, cy, Lx, Ly):
        """
        Map center (cx, cy) to the principal cell [-Lx/2, Lx/2] x [-Ly/2, Ly/2]
        using periodicity.
        """

        cx_t = backend.asarray(cx, complex=False)
        cy_t = backend.asarray(cy, complex=False)

        Lx_t = backend.asarray(Lx, complex=False)
        Ly_t = backend.asarray(Ly, complex=False)

        cx_wrapped = backend.mod(cx_t + Lx_t / 2.0, Lx_t) - Lx_t / 2.0
        cy_wrapped = backend.mod(cy_t + Ly_t / 2.0, Ly_t) - Ly_t / 2.0
        
        return cx_wrapped, cy_wrapped 
        
class Rectangle(VectorObject):
    """
    Rectangle vector object.
    """
    def __init__(self,
                 backend: Backend,
                 canvas: Canvas,
                 center: Tuple[float, float],
                 size: Tuple[float, float], 
                 epsilon: Any,
                 mu: Any = 1.0,
                 angle: float = 0.0,
                 soft_mask: bool = False,
                 smoothness: float = 0.05):
        """
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
            (width, height) of the rectangle.
        epsilon : Any
            Electric permittivity.
        mu : Any
            Magnetic permeability. Default is 1.0.
        angle : float
            Rotation angle in radians.
        soft_mask : bool
            Whether the object should use a soft mask for differentiable operations. Default is False.
            If True the bitmap representation will use smooth sigmoid approximation.
            *Important*: Fourier coefficients will still be computed analytically for sharp boundaries,
            so soft_mask only affects real-space distributions.
        smoothness : float
            Smoothness parameter for sigmoid. Default is 0.05.
        
        """
        
        super().__init__(backend,
                         canvas,
                         center,
                         epsilon,
                         mu,
                         angle,
                         soft_mask,
                         smoothness)
        
        if len(size) != 2:
            raise ValueError(f"size must be tuple of 2 floats, got {size}")
        if any(s <= 0 for s in size):
            raise ValueError(f"size values must be positive, got {size}")
        
        self._size = self.backend.asarray(size, complex=False)
        
    @property
    def size(self) -> Tuple[float, float]:
        return self._size
    
    @size.setter
    def size(self, value: Tuple[float, float]) -> None:
        if len(value) != 2:
            raise ValueError(f"size must be tuple of 2 floats, got {value}")
        if any(s <= 0 for s in value):
            raise ValueError(f"size values must be positive, got {value}")
        
        self._size = self.backend.asarray(value, complex=False)
        
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
        
        if self.soft_mask:
            # Smooth mask
            eps = self.smoothness * min(w, h)  # tuning parameter

            dist_x = ax - half_w
            dist_y = ay - half_h

            ex = -dist_x / eps
            ey = -dist_y / eps

            sx = self.backend.sigmoid(ex)
            sy = self.backend.sigmoid(ey)

            mask = sx * sy        # (Nx, Ny), float, differentiable
        else:
            # Sharp mask
            mask = (ax <= half_w) & (ay <= half_h)            # (Nx, Ny), bool

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
            Fourier coefficients mat_{m,n}, shape (B, 2M+1, 2N+1), complex.
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
                  epsilon: Any,
                  mu: Any = 1.0):
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
            0 → epsilon_bg, mu_bg, 1 → epsilon, mu.
        epsilon : Any
            Permittivity inside pixels where bitmap == 1.
        mu : Any, optional
            Permeability inside pixels where bitmap == 1 (default is 1.0).

        """
        bitmap_new, mu_t = Bitmap._init_validation(backend, canvas, bitmap, 
                                                   epsilon, mu)
        
        self._backend = backend
        self._bitmap = bitmap_new  # (Nx, Ny), real, values ∈ {0,1}
        
        # Setup canvas
        canvas._grid = self._backend.shape(self._bitmap)  # enforce grid from bitmap shape
        self._canvas = canvas
        
        # Store material values
        self._epsilon = backend.asarray(epsilon, complex=True)
        self._mu = mu_t
    
    """ Simulation properties """
    @property
    def backend(self) -> Backend:
        return self._backend
    @property
    def canvas(self) -> Canvas:
        return self._canvas
    
    """ Geometric properties """
    @property    
    def period(self) -> Tuple[float, float]:
        return self._canvas.period
    @property
    def grid(self) -> Tuple[int, int]:
        return self._bitmap.shape
    @property
    def bitmap(self) -> Any:
        return self._bitmap
    
    """ Material properties """
    @property
    def epsilon(self) -> Any:
        return self._epsilon
    @property
    def mu(self) -> Any:
        return self._mu
    
    """ Calculate material distributions """
    def epsilon_xy(self, epsilon_bg: Any) -> Any:
        """
        Compute the real-space permittivity distribution.
        
        Returns
        -------
        epsilon_xy : backend tensor
            Permittivity distribution in real space, shape (Nx, Ny), complex dtype.   
        """
        bg_b, val_b = VectorObject.adjustshapes(self.backend, 
                                        self.epsilon, 
                                        epsilon_bg)
        epsilon_xy = self.bitmap * (val_b - bg_b) + bg_b
        epsilon_xy = self.backend.asarray(epsilon_xy, complex=True)
        return epsilon_xy
    
    def mu_xy(self, mu_bg: Any) -> Any:
        """
        Compute the real-space permeability distribution.
        
        Returns
        -------
        mu_xy : backend tensor
            Permeability distribution in real space, shape (Nx, Ny), complex dtype.   
        """
        bg_b, val_b = VectorObject.adjustshapes(self.backend, 
                                        self.mu, 
                                        mu_bg)
        
        mu_xy = self.bitmap * (val_b - bg_b) + bg_b
        mu_xy = self.backend.asarray(mu_xy, complex=True)
        return mu_xy
    
    """ Static helper methods """
    @staticmethod
    def _init_validation(backend: Backend, 
                        canvas: Canvas,
                        bitmap: Any,
                        epsilon: Any,
                        mu: Any) -> None:
        
        if not isinstance(backend, Backend):
            raise TypeError("backend must be a Backend instance")
        if not isinstance(canvas, Canvas):
            raise TypeError("canvas must be a Canvas instance")
        
        # --- Convert bitmap to backend real tensors ---
        bm = backend.asarray(bitmap, complex=False)

        # --- Basic shape check (2D) ---
        shape = backend.shape(bm)
        if len(shape) != 2:
            raise ValueError(f"bitmap must be 2D, got shape {shape}")
        
        # --- Validate that the bitmap only contains 0/1 ---
        # Compute unique values using backend (works for torch/numpy/jax)
        uniq = backend.unique(backend.clone(bm))

        # Allowed values: 0 and 1
        allowed0 = backend.asarray(0, complex=False)
        allowed1 = backend.asarray(1, complex=False)

        # Boolean mask: True when value is NOT 0 or 1
        is_not_0 = uniq != allowed0
        is_not_1 = uniq != allowed1
        bad_mask = is_not_0 & is_not_1   # True for illegal values

        # If any illegal values exist → issue a Python warning
        # This won't break autograd because only uniq is used
        if backend.any(bad_mask):
            import warnings
            warnings.warn(
                f"Bitmap contains values other than 0 and 1. "
                f"Values found: {uniq}",
                RuntimeWarning
            )
        
         # --- Force it to be strict 0 to 1 mask ---
        bm_bin = backend.clamp(bm, 0, 1)
        
        # Convert to backend tensors 
        eps_t = backend.asarray(epsilon, complex=True)
        eps_shape = backend.shape(eps_t)
        
        is_scalar_number = isinstance(mu, (int, float, complex))
        if is_scalar_number and (mu == 1 or mu == 1.0 or mu == 1+0j):
        # If epsilon is scalar → scalar mu is fine
        # If epsilon is batch → create matching batch of ones
            if len(eps_shape) == 0:
                mu_t = backend.asarray(1.0, complex=True)
            else:
                B = eps_shape[0]
                mu_t = backend.asarray(backend.ones((B,)), complex=True)
                
        else:
            mu_t = backend.asarray(mu, complex=True)

        mu_shape  = backend.shape(mu_t)

        # Only allow scalar or 1D for both
        if len(eps_shape) > 1:
            raise ValueError(
                f"epsilon must be scalar or 1D (batch), got shape {eps_shape}"
            )
        if len(mu_shape) > 1:
            raise ValueError(
                f"mu must be scalar or 1D (batch), got shape {mu_shape}"
            )

        # If both are 1D, batch sizes must match
        if len(eps_shape) == 1 and len(mu_shape) == 1:
            if eps_shape[0] != mu_shape[0]:
                raise ValueError(
                    f"epsilon and mu batch sizes differ: "
                    f"{eps_shape[0]} vs {mu_shape[0]}"
                )
        return bm_bin, mu_t
    
class VectorGroup:
    """
    Group of vector objects.
    **TODO**: to be implemented.
    """
    def __init__(self):
        raise NotImplementedError("VectorGroup is not yet implemented.")
        
    