from typing import Union, List, Tuple, Any
from src.geometry import Object, Bitmap


class MaterialMap:
    def __init__(self, 
                 objects: Union[Object, List[Object], Bitmap], 
                 period: Tuple[float, float], 
                 grid: Tuple[int, int],
                 epsilonbg: Any = 1.0,
                 mugbg: Any = 1.0,
                 fourier_analytic: bool = False,
                 subpixel_smoothing: bool = False):
        '''
        Parameters
        ----------
        objects : Object, Bitmap, or list of Object
            Geometric object(s) defining the material distribution.
        period : tuple of float
            (Lx, Ly) period of the unit cell.
        grid : tuple of int
            (Nx, Ny) grid size for sampling the material functions.
            Ignored in the case of Bitmap object.
        epsilonbg : float, optional
            Background permittivity (default is 1.0).
            Ignored in the case of Bitmap object.
        mugbg : float, optional
            Background permeability (default is 1.0).
            Ignored in the case of Bitmap object.
        fourier_analytic : bool, optional
            Whether to use analytic Fourier coefficients (default is False).
            Works only with Object.
        subpixel_smoothing : bool, optional
            Whether to apply subpixel smoothing (default is False).
            Not compatible with fourier_analytic=True.
        '''
        
        backend = self.init_validation(objects)
        
        self._backend = backend
        self._objects = objects
        self._period = period
        self._grid = grid
        self._epsilonbg = epsilonbg
        self._mugbg = mugbg
        
        self.fourier_analytic = fourier_analytic
        self.subpixel_smoothing = subpixel_smoothing
    
    
    @property
    def backend(self):
        return self._backend
    @property
    def objects(self) -> Union[Object, List[Object], Bitmap]:
        return self._objects
    @property    
    def period(self) -> Tuple[float, float]:
        return self._period
    @property
    def grid(self) -> Tuple[int, int]:
        return self._grid
    @property
    def epsilonbg(self) -> Any:
        return self._epsilonbg
    @property
    def mugbg(self) -> Any:
        return self._mugbg
    
    @property
    def epsilon_xy(self):
        '''
        Real-space permittivity distribution epsilon(x,y).
        Shape: (B, Nx, Ny)
        In the case of multiple objects, later objects override earlier ones. 
        '''
        backend = self.backend
        period = self.period
        grid = self.grid

        # Single object case
        if isinstance(self.objects, Object):
            return self.objects.epsilon_xy(period, grid, self.epsilonbg)

        # Single bitmap case
        if isinstance(self.objects, Bitmap):
            return self.objects.epsilon_xy  

        # List of objects case
        for obj in self.objects:
            if not isinstance(obj, Object):
                raise TypeError("All items in objects list must be Object instances")

        Nx, Ny = grid
        
        # Get epsilon from first object
        epsilon_val0 = self.objects[0].epsilon

        # --- Background field, shape (B, Nx, Ny) ---
        bg_b, _ = Object.adjustshapes(backend, 
                                        epsilon_val0, 
                                        self.epsilonbg)  # (B, 1, 1)
        B = backend.shape(bg_b)[0] # batch size
        
        # Make a (1, Nx, Ny) ones field and broadcast bg_b
        ix = backend.arange(0, Nx)
        iy = backend.arange(0, Ny)
        X, _ = backend.meshgrid(ix, iy, indexing="ij")      # (Nx, Ny)
        ones_xy = backend.ones_like(X)                      # (Nx, Ny)
        ones_xy = backend.reshape(ones_xy, (1, Nx, Ny))     # (1, Nx, Ny)

        eps_xy = bg_b * ones_xy                             # (B, Nx, Ny)

        # --- Paint objects one by one; last object wins on overlaps ---
        for obj in self.objects:
            # 0/1 mask for this object, shape (B, Nx, Ny), from epsilon geometry
            mask = obj.to_bitmap(period, grid, epsilon=True)        # (B, Nx, Ny)
            mask_real = backend.asarray(mask, complex=False)        # real 0/1

            # Per-object epsilon value, with the SAME background and batch shape
            bg_obj, val_obj = Object.adjustshapes(
                backend,
                obj.epsilon,      # matval
                self.epsilonbg    # matbg
            )  # both (B, 1, 1)
            
            if backend.shape(bg_obj)[0] != B:
                raise ValueError("All objects must have the same batch size")
            
            delta_obj = val_obj - bg_obj                             # (B, 1, 1)

            # Build full epsilon for this object over the cell: bg + Δ * mask
            mask_c = backend.asarray(mask_real, complex=True)        # (B, Nx, Ny)
            eps_obj = bg_obj + delta_obj * mask_c                    # (B, Nx, Ny)

            # Where mask == 1, override eps_xy with eps_obj
            cond = mask_real > 0.0                                   # (B, Nx, Ny), bool
            eps_xy = backend.where(cond, eps_obj, eps_xy)            # (B, Nx, Ny)

        return eps_xy
    
    def init_validation(self, objects: Union[Object, List[Object], Bitmap]):
        # Validation
        if not isinstance(objects, (Object, Bitmap, list, tuple)):
            raise TypeError("objects must be an Object, Bitmap, or list/tuple of Objects")

        # Check backend consistency
        if isinstance(objects, (Object, Bitmap)):
            # Single object
            backend = objects.backend
        else:
            # List / tuple of objects
            if len(objects) == 0:
                raise ValueError("objects list/tuple cannot be empty")

            first = objects[0]
            if not isinstance(first, (Object, Bitmap)):
                raise TypeError(
                    f"All items in objects list/tuple must be Object or Bitmap instances, got {type(first)}"
                )

            backend = first.backend

            for obj in objects:
                if not isinstance(obj, Object):
                    raise TypeError(
                        f"All items in objects list/tuple must be Object or Bitmap instances, got {type(obj)}"
                    )
                if obj.backend is not backend:
                    raise ValueError("All objects in the list must have the same backend")
        
        return backend
    
    
class Layer:
    def __init__(self, epsilon_xy, mu_xy, thickness):
        '''
        Parameters
        ----------
        epsilon_xy : array-like or backend tensor
            Material function map epsilon(x,y) in real space.
        mu_xy : array-like or backend tensor
            Material function map mu(x,y) in real space.
        thickness : float
            Thickness of the layer.
        '''
        self.epsilon_xy = epsilon_xy  # (Nr, Ntheta) or (B, Nr, Ntheta)
        self.mu_xy = mu_xy            # (Nr, Ntheta) or (B, Nr, Ntheta)
        self.thickness = thickness