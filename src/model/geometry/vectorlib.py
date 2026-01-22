# src/model/geometry/vectorlib.py
# Vector geometry library

from typing import Tuple, Any

from src.model.geometry.vector import VectorObject
from src.model.geometry.lattice import Lattice
from src.model.material import BaseMaterial
from src.backend import Backend
from src.model.geometry.sampling import bitmap_rect, bitmap_ellipse, matmap
from src.model.geometry.fourier import matmap_fourier_rect, matmap_fourier_ellipse, fft_matmap

class Rectangle(VectorObject):
    """
    Rectangle vector object.
    """
    def __init__(self,
                 center: Tuple[float, float],
                 size: Tuple[float, float], 
                 material: "BaseMaterial",
                 angle: float = 0.0,
                 soft_mask: bool = False,
                 smoothness: float = 0.02):
        """
        Parameters
        ----------
        center : tuple of float
            (x,y) coordinates of the rectangle's center in length units. (0, 0) is the center.
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
            Smoothness parameter for sigmoid. Default is 0.02.
        
        """
        
        super().__init__(center,
                         material,
                         angle,
                         soft_mask,
                         smoothness)
        
        if len(size) != 2:
            raise ValueError(f"size must be tuple of 2 floats, got {size}")
        if any(s < 0 for s in size):
            raise ValueError(f"size values must be positive, got {size}")
        
        self.size = size
        
    def bitmap(self, backend: "Backend", lattice: "Lattice") -> Any:
        '''
        Convert the rectangle to a bitmap representation on the specified grid.
        
        Parameters
        ----------
        backend : Backend
            Computational backend.
        lattice : Lattice
            Lattice object defining the simulation domain.
            
        Returns
        -------
        bitmap : backend tensor
            Bitmap representation of the rectangle, shape (Nx, Ny), dtype bool.
        '''
        
        return bitmap_rect(backend,
                           lattice,
                           self.center,
                           self.size,
                           self.angle,
                           self.soft_mask,
                           self.smoothness)

    def fraction(self, backend: "Backend", lattice: "Lattice") -> Any:
        '''
        Compute the fill fraction of the rectangle in the unit cell.
        
        Parameters
        ----------
        backend : Backend
            Computational backend.
        lattice : Lattice
            Lattice object defining the simulation domain.
        
        Returns
        -------
        fraction : backend tensor
            Fill fraction of the rectangle, shape (B,), float dtype.
        '''
        Lx, Ly = lattice.period
        w, h = self.size
        
        area_rect = w * h
        area_cell = Lx * Ly
        
        fill_fraction = area_rect / area_cell
        
        return backend.asarray(fill_fraction, complex=False)
    
    def matmap_fourier(self, 
                       backend: "Backend", 
                       lattice: "Lattice",
                       matval: complex,
                       matbg: complex,
                       closed_form: bool):
        """
        Computes Fourier material map in the closed-form or 
        from the real-space distribution

        Parameters
        ----------
        backend : Backend
            Computational backend.
        lattice : Lattice
            Lattice object defining the simulation domain.
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
        if closed_form:
            mat_mn = matmap_fourier_rect(backend,
                                        self.center,
                                        self.size,
                                        self.angle,
                                        lattice.period,
                                        lattice.M,
                                        lattice.N,
                                        matval, matbg)
        else:
            mat_xy = matmap(backend, 
                            self.bitmap(backend, lattice),
                            matval,
                            matbg)
            mat_mn = fft_matmap(backend,
                                mat_xy,
                                lattice.M,
                                lattice.N)

        return mat_mn
          
class Ellipse(VectorObject):
    """
    Ellipse vector object.
    """
    def __init__(self,
                 center: Tuple[float, float],
                 size: Tuple[float, float], 
                 material: BaseMaterial,
                 angle: float = 0.0,
                 soft_mask: bool = False,
                 smoothness: float = 0.02):
        """
        Parameters
        ----------
        center : tuple of float
            (x,y) coordinates of the ellipse's center in length units. (0, 0) is the center.
        size : tuple of float
            (width, height) of the ellipse. Length units.
        material : BaseMaterial
            Material of the ellipse.
        angle : float
            Rotation angle in radians.
        soft_mask : bool
            Whether the object should use a soft mask for differentiable operations. Default is False.
            If True the bitmap representation will use smooth sigmoid approximation.
            *Important*: Fourier coefficients will still be computed analytically for sharp boundaries,
            so soft_mask only affects real-space distributions.
        smoothness : float
            Smoothness parameter for sigmoid. Default is 0.02.
        
        """
        
        super().__init__(center,
                         material,
                         angle,
                         soft_mask,
                         smoothness)
        
        if len(size) != 2:
            raise ValueError(f"size must be tuple of 2 floats, got {size}")
        if any(s < 0 for s in size):
            raise ValueError(f"size values must be positive, got {size}")
        
        self.size = size
        

    def bitmap(self, backend: "Backend", lattice: "Lattice") -> Any:
        '''
        Convert the ellipse to a bitmap representation on the specified grid.
        
        Parameters
        ----------
        backend : Backend
            Computational backend.
        lattice : Lattice
            Lattice object defining the simulation domain.
            
        Returns
        -------
        bitmap : backend tensor
            Bitmap representation of the ellipse, shape (Nx, Ny), dtype bool.
        '''
        
        return bitmap_ellipse(backend,
                              lattice,
                              self.center,
                              self.size,
                              self.angle,
                              self.soft_mask,
                              self.smoothness)

    def fraction(self, backend: "Backend", lattice: "Lattice") -> Any:
        '''
        Compute the fill fraction of the ellipse in the unit cell.
        
        Parameters
        ----------
        backend : Backend
            Computational backend.
        lattice : Lattice
            Lattice object defining the simulation domain.
        
        Returns
        -------
        fraction : backend tensor
            Fill fraction of the ellipse, shape (B,), float dtype.
        '''
        Lx, Ly = lattice.period
        w, h = self.size
        
        area_ellipse = backend.pi * (w / 2) * (h / 2)
        area_cell = Lx * Ly
        
        fill_fraction = area_ellipse / area_cell
        
        return backend.asarray(fill_fraction, complex=False)
    
    def matmap_fourier(self, 
                       backend: "Backend", 
                       lattice: "Lattice",
                       matval: complex,
                       matbg: complex,
                       closed_form: bool):
        """
        Computes Fourier material map in the closed-form or 
        from the real-space distribution

        Parameters
        ----------
        backend : Backend
            Computational backend.
        lattice : Lattice
            Lattice object defining the simulation domain.
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
        if closed_form:
            mat_mn = matmap_fourier_ellipse(backend,
                                        self.center,
                                        self.size,
                                        self.angle,
                                        lattice.period,
                                        lattice.M,
                                        lattice.N,
                                        matval, matbg)
        else:
            mat_xy = matmap(backend, 
                            self.bitmap(backend, lattice),
                            matval,
                            matbg)
            mat_mn = fft_matmap(backend,
                                mat_xy,
                                lattice.M,
                                lattice.N)

        return mat_mn

class Uniform(Rectangle):
    """
    Uniform vector object that fills the entire unit cell.
    Inherits from Rectangle with size equal to lattice period.
    """
    def __init__(self, material: "BaseMaterial"):
        """
        Parameters
        ----------
        material : BaseMaterial
            Material of the uniform object.
        """
        super().__init__(center=(0.0, 0.0),
                         size=(0.0, 0.0),  
                         material=material,
                         angle=0.0,
                         soft_mask=False,
                         smoothness=0.01)
    
    def bitmap(self, backend: "Backend", lattice: "Lattice") -> Any:
        '''
        Convert the uniform object to a bitmap representation on the specified grid.
        This will be a full True bitmap.

        Parameters
        ----------
        backend : Backend
            Computational backend.
        lattice : Lattice
            Lattice object defining the simulation domain.

        Returns
        -------
        bitmap : backend tensor
            Bitmap representation of the uniform object, shape (Nx, Ny), dtype bool.
        '''
        Nx, Ny = lattice.grid
        return backend.ones((Nx, Ny))