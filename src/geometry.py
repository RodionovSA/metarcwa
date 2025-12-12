from typing import Tuple, Any

from src.backend import Backend
from src.material import BaseMaterial

class Lattice:
    """
    Lattice base class for geometric objects.
    """
    def __init__(self, period: Tuple[float, float],
                 grid: Tuple[int, int]):
        '''
        Parameters
        ----------
        period : tuple of float
            (Lx, Ly) period of the lattice. Length units.
        grid : tuple of int
            (Nx, Ny) grid size for sampling the material functions.
        '''
        Lattice._init_validation(period, grid)
        
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
            (Lx, Ly) period of the unit cell. Length units.
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
                 lattice: Lattice,
                 center: Tuple[float, float], 
                 material: BaseMaterial,
                 angle: float = 0.0,
                 soft_mask: bool = False,
                 smoothness: float = 0.05):
        '''
        Parameters
        ----------
        backend : Backend
            Computational backend.
        lattice : Lattice
            Lattice object defining the simulation domain.
        center : tuple of float
            (x,y) coordinates of the object's center (in the range [-Lx/2, Lx/2] x [-Ly/2, Ly/2]).
            Length units. If center is outside this range, it will be wrapped into the principal cell.
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
        VectorObject._init_validation(backend, 
                                    lattice, 
                                    center, 
                                    angle, 
                                    material,
                                    soft_mask,
                                    smoothness)
        
        self._backend = backend
        self._lattice = lattice
        self._material = material
        
        self._center = VectorObject._wrap_center(backend,
                                           center[0], center[1], 
                                           self.lattice.period[0], 
                                           self.lattice.period[1])
        self._angle = self.backend.asarray(angle, complex=False)

        # Differentiability settings
        self._soft_mask = soft_mask
        self._smoothness = smoothness

    """ Simulation properties """
    @property
    def backend(self) -> Backend:
        return self._backend

    @property
    def lattice(self) -> Lattice:
        return self._lattice

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
        angle = self.backend.asarray(value, complex=False)
        if len(angle.shape) != 0:
            raise ValueError(f"angle must be a scalar float, got shape {angle.shape}")
        self._angle = angle
    
    """ Material properties """
    @property
    def material(self) -> BaseMaterial:
        return self._material
    
    @property
    def epsilon(self) -> Any:
        return self.material.epsilon_tensor
    
    @property
    def mu(self) -> Any:
        return self.material.mu_tensor
    
    """ Calculate material distributions """
    def epsilon_xy(self, material_bg: BaseMaterial):
        '''
        Compute the real-space permittivity distribution.
        Parameters
        ----------
        material_bg : BaseMaterial
            Background material.
        Returns
        -------
        epsilon_xy : backend tensor
            Permittivity distribution tensor in real space, shape (wvl, 3, 3, Nx, Ny), complex dtype.
        '''
        return self._matdist_real(material_bg, 
                                  mode='epsilon')
    
    def mu_xy(self, material_bg: BaseMaterial):
        '''
        Compute the real-space permittivity distribution.
        Parameters
        ----------
        material_bg : BaseMaterial
            Background material.
        Returns
        -------
        mu_xy : backend tensor
            Permeability distribution in real space, shape (wvl, 3, 3, Nx, Ny), complex dtype.
        '''
        return self._matdist_real(material_bg, 
                                  mode='mu')
    
    def epsilon_mn(self,
                   M: int,
                   N: int,
                   mat_bg: BaseMaterial,
                   inverse: bool = False,
                   regularized: bool = False,
                   regularization: float = 1e-8):
        '''
        Compute the Fourier coefficients of the permittivity distribution in the closed form.
        Parameters
        ----------
        M, N : int
            Number of harmonics along x and y.
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
            return self._matdist_fourier(M, N, 
                                         epsilon_eff, epsilon_eff_bg)
        
        return self._matdist_fourier(M, N, 
                                     epsilon_tensor, epsilonbg_tensor)
    
    def mu_mn(self,
               M: int,
               N: int,
               mat_bg: BaseMaterial,
               inverse: bool = False,
               regularized: bool = False,
               regularization: float = 1e-8):
        '''
        Compute the Fourier coefficients of the permeability distribution in the closed form.
        Parameters
        ----------
        M, N : int
            Number of harmonics along x and y.
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
            return self._matdist_fourier(M, N, 
                                         mu_eff, mu_eff_bg)
        
        return self._matdist_fourier(M, N, 
                                     mu_tensor, mubg_tensor)
        
    def _matdist_real(self, material_bg: BaseMaterial, mode: str):
        '''
        Compute the real-space material distribution for the object.

        Parameters
        ----------
        material_bg : BaseMaterial
            Background material value.
        mode : str
            'epsilon' or 'mu' to specify which material property to compute.
        Returns
        -------
        matdist_xy : backend tensor
            Material distribution in real space, shape (B, 3, 3, Nx, Ny), complex dtype.
        '''
        Nx, Ny = self.lattice.grid

        # Get bitmap mask
        mask_c = self.bitmap          # (Nx, Ny), bool
        mask_c = self.backend.reshape(mask_c, (1, 1, 1, Nx, Ny))  # (1, 1, 1, Nx, Ny)
        
        # Material values
        if mode == 'epsilon':
            mat_tensor = self.epsilon  # (wvl, 3, 3)
            mat_bg_tensor = material_bg.epsilon_tensor  # (wvl, 3, 3)
        elif mode == 'mu':
            mat_tensor = self.mu  # (wvl, 3, 3)
            mat_bg_tensor = material_bg.mu_tensor  # (wvl, 3, 3)
        else:
            raise ValueError(f"Unsupported mode '{mode}', must be 'epsilon' or 'mu'")
        
        if mat_tensor.shape[0] != mat_bg_tensor.shape[0]:
            if mat_bg_tensor.shape[0] == 1:
                # replicate along wavelength dimension
                target_shape = (mat_tensor.shape[0],) + mat_bg_tensor.shape[1:]
                mat_bg_tensor = self.backend.expand(mat_bg_tensor, target_shape)
            else:
                raise ValueError("Material and background material must have the same number of wavelengths")
        
        # Broadcast epsilon and epsilon_bg to (wvl, 3, 3, 1, 1)
        init_shape = mat_tensor.shape      # (wvl, 3, 3)
        mat = self.backend.reshape(mat_tensor, init_shape + (1, 1))# (wvl, 3, 3, 1, 1)
        mat_bg = self.backend.reshape(mat_bg_tensor, init_shape + (1, 1))# (wvl, 3, 3, 1, 1)
        
        # Expand eps_bg to (wvl, 3, 3, Nx, Ny)
        matdist_xy = self.backend.expand(mat_bg,init_shape + (Nx, Ny))# (wvl, 3, 3, Nx, Ny)
        # Δmat per batch
        delta_mat_b = mat - mat_bg                          # (B, 3, 3, 1, 1)

        # mat = bg + Δmat * mask
        matdist_xy = matdist_xy + delta_mat_b * mask_c      # (B, 3, 3, Nx, Ny)

        return matdist_xy
    
    """ Static helper methods """
    @staticmethod
    def _init_validation(backend: Backend, 
                        lattice: Lattice,
                        center: Tuple[float, float],
                        angle: float,
                        material: BaseMaterial,
                        soft_mask: bool,
                        smoothness: float) -> None:
        
        angle = backend.asarray(angle, complex=False)
        if not isinstance(backend, Backend):
            raise TypeError("backend must be a Backend instance")
        if not isinstance(lattice, Lattice):
            raise TypeError("lattice must be a Lattice instance")
        if len(center) != 2:
            raise ValueError(f"center must be tuple of 2 floats, got {center}")
        if len(angle.shape) != 0:
            raise ValueError(f"angle must be a scalar float, got shape {angle.shape}")
        if not isinstance(soft_mask, bool):
            raise TypeError(f"soft_mask must be bool, got {type(soft_mask)}")
        if not isinstance(smoothness, float) and not isinstance(smoothness, int):
            raise TypeError(f"smoothness must be float, got {type(smoothness)}")
        if smoothness <= 0:
            raise ValueError(f"smoothness must be positive, got {smoothness}")
        if not isinstance(material, BaseMaterial):
            raise TypeError("material must be a BaseMaterial instance")
        
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
                 lattice: Lattice,
                 center: Tuple[float, float],
                 size: Tuple[float, float], 
                 material: BaseMaterial,
                 angle: float = 0.0,
                 soft_mask: bool = False,
                 smoothness: float = 0.05):
        """
        Parameters
        ----------
        backend : Backend
            Computational backend.
        lattice : Lattice
            Lattice object defining the simulation domain.
        center : tuple of float
            (x,y) coordinates of the object's center (in the range [-Lx/2, Lx/2] x [-Ly/2, Ly/2]).
            Length units. If center is outside this range, it will be wrapped into the principal cell.
        size : tuple of float
            (width, height) of the rectangle. Length units.
        material : BaseMaterial
            Material of the rectangle.
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
                         lattice,
                         center,
                         material,
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
        
        Lx, Ly = self.lattice.period
        Nx, Ny = self.lattice.grid
        
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

    @property
    def fraction(self) -> Any:
        '''
        Compute the fill fraction of the rectangle in the unit cell.
        
        Returns
        -------
        fraction : backend tensor
            Fill fraction of the rectangle, shape (B,), float dtype.
        '''
        Lx, Ly = self.lattice.period
        w, h = self.size
        
        area_rect = w * h
        area_cell = Lx * Ly
        
        fill_fraction = area_rect / area_cell
        
        return self.backend.asarray(fill_fraction, complex=False)
    
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
            Material value tensor inside the rectangle (B, 3, 3).
        matbg : complex
            Background material value tensor (B, 3, 3).

        Returns
        -------
        mat_mn : backend tensor
            Fourier coefficients mat_{m,n}, shape (B, 3, 3, 2M+1, 2N+1), complex.
            Indices correspond to m ∈ [-M..M], n ∈ [-N..N].
        """
        Lx, Ly = self.lattice.period
        w, h = self.size
        cx, cy = self.center
        
        # Shift center to [0, Lx] x [0, Ly] coordinates
        cx = cx + Lx / 2.0
        cy = cy + Ly / 2.0
        
        # Adjust shapes
        if matval.shape[0] != matbg.shape[0]:
            if matbg.shape[0] == 1:
                # replicate along wavelength dimension
                target_shape = (matval.shape[0],) + matbg.shape[1:]
                matbg = self.backend.expand(matbg, target_shape)
            else:
                raise ValueError("Material and background material must have the same number of wavelengths")
            
        val_b = self.backend.reshape(matval, (matval.shape[0], 3, 3, 1, 1)) # (B, 3, 3, 1, 1)
        bg_b  = self.backend.reshape(matbg,  (matbg.shape[0], 3, 3, 1, 1)) # (B, 3, 3, 1, 1)

        # Material contrast
        delta_mat = val_b - bg_b   # (B, 3, 3, 1, 1)

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

        Sx = sinc(zx) # (2M+1, 2N+1)
        Sy = sinc(zy) # (2M+1, 2N+1)

        # Phase factor exp(-j(Gm x_c + Gn y_c))
        phase_arg = Gm_grid * cx + Gn_grid * cy
        phase = self.backend.exp(-1j * phase_arg) # (2M+1, 2N+1)
        
        #Broadcast to (B, 3, 3, 2M+1, 2N+1)
        Sx = self.backend.reshape(Sx, (1, 1, 1, 2 * M + 1, 2 * N + 1))
        Sy = self.backend.reshape(Sy, (1, 1, 1, 2 * M + 1, 2 * N + 1))
        phase = self.backend.reshape(phase, (1, 1, 1, 2 * M + 1, 2 * N + 1))

        # Contrast contribution
        area_factor = (w * h) / (Lx * Ly)
        
        delta_mat_mn = delta_mat * area_factor * Sx * Sy * phase  # (B, 3, 3, 2M+1, 2N+1)

        # Initialize with contrast term
        mat_mn = delta_mat_mn

        # --- Add background at (m=0,n=0), i.e. index (M,N) ---

        # bg_b: (B, 3, 3, 1, 1)
        # We add bg_b to mat_mn[:, M, N]
        # For Torch/NumPy this indexing is fine:
        if hasattr(mat_mn, "__setitem__"):
            # mat_{00} = matbg + Δmat * fill_fraction (already in mat_mn[:, M, N])
            mat_mn[..., M, N] = mat_mn[..., M, N] + bg_b[..., 0, 0]
        else:
            # if it is a different backend without in-place assignment,
            # should implement a functional update here.
            raise NotImplementedError("In-place assignment not supported for this backend.")

        return mat_mn
          
class Bitmap:
    """Material distribution defined from 2D bitmap masks for a given material."""
    def __init__(self,
                  backend: Backend,
                  lattice: Lattice,
                  bitmap: Any,
                  material: BaseMaterial):
        """
        Parameters
        ----------
        backend : Backend
            Computational backend.
        lattice : Lattice
            Lattice object defining the simulation domain.
        bitmap : array-like or backend tensor
            2D array representing the bitmap of material distribution.
            Must be 0/1 or False/True. 
            0 → epsilon_bg, mu_bg, 1 → epsilon, mu.
        material : BaseMaterial
            Material object defining epsilon and mu inside pixels where bitmap == 1.

        """
        bitmap_new = Bitmap._init_validation(backend, lattice, bitmap, material)
        
        self._backend = backend
        self._bitmap = bitmap_new  # (Nx, Ny), real, values ∈ {0,1}
        self._material = material
        
        # Setup lattice
        lattice._grid = self._backend.shape(self._bitmap)  # enforce grid from bitmap shape
        self._lattice = lattice
    
    """ Simulation properties """
    @property
    def backend(self) -> Backend:
        return self._backend
    @property
    def lattice(self) -> Lattice:
        return self._lattice
    
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
    def material(self) -> BaseMaterial:
        return self._material
    @property
    def epsilon(self) -> Any:
        return self.material.epsilon_tensor
    @property
    def mu(self) -> Any:
        return self.material.mu_tensor
    
    """ Calculate material distributions """
    def epsilon_xy(self, material_bg: BaseMaterial) -> Any:
        """
        Compute the real-space permittivity distribution.
        
        Parameters
        ----------
        material_bg : BaseMaterial
            Background material value.
        
        Returns
        -------
        epsilon_xy : backend tensor
            Permittivity distribution in real space, shape (wvl, 3, 3, Nx, Ny), complex dtype.   
        """
        # Init epsilon tensors
        epsilon_tensor = self.epsilon  # (wvl, 3, 3)
        epsilon_bg_tensor = material_bg.epsilon_tensor  # (wvl, 3, 3)
        
        if epsilon_tensor.shape[0] != epsilon_bg_tensor.shape[0]:
            if epsilon_bg_tensor.shape[0] == 1:
                # replicate along wavelength dimension
                target_shape = (epsilon_tensor.shape[0],) + epsilon_bg_tensor.shape[1:]
                epsilon_bg_tensor = self.backend.expand(epsilon_bg_tensor, target_shape)
            else:
                raise ValueError("Material and background material must have the same number of wavelengths")
            
        # Broadcast epsilon and epsilon_bg to (wvl, 3, 3, 1, 1)
        init_shape = epsilon_tensor.shape      # (wvl, 3, 3)
        eps = self.backend.reshape(epsilon_tensor, init_shape + (1, 1))# (wvl, 3, 3, 1, 1)
        eps_bg = self.backend.reshape(epsilon_bg_tensor, init_shape + (1, 1))# (wvl, 3, 3, 1, 1)
        
        # Broadcast bitmap to (1, 1, 1, Nx, Ny)
        bitmap = self.backend.reshape(self.bitmap, (1, 1, 1) + self.bitmap.shape)  # (1, 1, 1, Nx, Ny)
        
        epsilon_xy = bitmap * (eps - eps_bg) + eps_bg # (wvl, 3, 3, Nx, Ny)
        epsilon_xy = self.backend.asarray(epsilon_xy, complex=True)
        return epsilon_xy
    
    def mu_xy(self, material_bg: Any) -> Any:
        """
        Compute the real-space permeability distribution.
        
        Parameters
        ----------
        material_bg : BaseMaterial
            Background material value.
        
        Returns
        -------
        mu_xy : backend tensor
            Permeability distribution in real space, shape (wvl, 3, 3, Nx, Ny), complex dtype.   
        """
        # Init epsilon tensors
        mu_tensor = self.mu  # (wvl, 3, 3)
        mu_bg_tensor = material_bg.mu_tensor  # (wvl, 3, 3)
        
        if mu_tensor.shape[0] != mu_bg_tensor.shape[0]:
            if mu_bg_tensor.shape[0] == 1:
                # replicate along wavelength dimension
                target_shape = (mu_tensor.shape[0],) + mu_bg_tensor.shape[1:]
                mu_bg_tensor = self.backend.expand(mu_bg_tensor, target_shape)
            else:
                raise ValueError("Material and background material must have the same number of wavelengths")
            
        # Broadcast epsilon and epsilon_bg to (wvl, 3, 3, 1, 1)
        init_shape = mu_tensor.shape      # (wvl, 3, 3)
        mu = self.backend.reshape(mu_tensor, init_shape + (1, 1))# (wvl, 3, 3, 1, 1)
        mu_bg = self.backend.reshape(mu_bg_tensor, init_shape + (1, 1))# (wvl, 3, 3, 1, 1)
        
        # Broadcast bitmap to (1, 1, 1, Nx, Ny)
        bitmap = self.backend.reshape(self.bitmap, (1, 1, 1) + self.bitmap.shape)  # (1, 1, 1, Nx, Ny)
        
        mu_xy = bitmap * (mu - mu_bg) + mu_bg
        mu_xy = self.backend.asarray(mu_xy, complex=True)
        return mu_xy
    
    """ Static helper methods """
    @staticmethod
    def _init_validation(backend: Backend, 
                        lattice: Lattice,
                        bitmap: Any,
                        material: BaseMaterial) -> None:
        
        if not isinstance(backend, Backend):
            raise TypeError("backend must be a Backend instance")
        if not isinstance(lattice, Lattice):
            raise TypeError("lattice must be a Lattice instance")
        if not isinstance(material, BaseMaterial):
            raise TypeError("material must be a BaseMaterial instance")
        
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
        
        return bm_bin
    
class VectorGroup:
    """
    Group of vector objects.
    **TODO**: to be implemented.
    """
    def __init__(self):
        raise NotImplementedError("VectorGroup is not yet implemented.")
        
    