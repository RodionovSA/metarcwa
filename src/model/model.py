# src/model/model.py
# Class for storing the simulation model
from typing import List, Sequence, Any

from src.model.layer import Layer
from src.model.source import Source
from src.model.geometry.lattice import Lattice
from src.backend import Backend

class Model:
    """ RCWA model """
    def __init__(self, 
                 backend: "Backend",
                 lattice: "Lattice",
                 layers: List["Layer"], 
                 source: "Source"):
        """
        Parameters
        ----------
        backend : Backend
            Computational backend
        lattice : Lattice
            Lattice object defining the periodicity
        layers : List[Layer]
            List of layers in the model (from incident to substrate)
        source : Source
            Source object defining the incident field
        """
        self._init_validation(backend, lattice, layers, source)
        
        self.backend = backend 
        self.lattice = lattice
        self.layers = layers
        self.source = source
        
    """ Simulation properties """
    @property
    def n_inc(self) -> Any:
        """Return refractive index of incident medium."""
        epsilon_inc = self.layers[0].material.epsilon_tensor(self.backend)[:,0,0]
        eps = self.backend.asarray(1e-9, complex=False)  
        return self.backend.sqrt(self.backend.real(epsilon_inc) + eps)
    
    @property
    def Kxy(self) -> Any:
        """ Return Kx and Ky matrices. """
        Kx, Ky = self.source.Kxy(self.backend, self.lattice, self.n_inc)
        return Kx, Ky
    
    @property
    def k0(self) -> Any:
        """ Return free space wavenumber array. """
        return self.source.k0(self.backend)
    
    @property
    def layers_properties(self) -> List[dict]:
        """Return list of dictionaries with layer properties."""
        props = []
        for layer in self.layers:
            prop = {
                "thickness": layer.thickness,
                "material_type": layer.material.type,
                "is_homogeneous": layer.is_homogeneous(self.backend, self.lattice),
                "is_magnetic": layer.material.is_magnetic
            }
            props.append(prop)
        return props
    
    
    
    """ Static helper methods """
    @staticmethod
    def _init_validation(backend, lattice, layers, source):
        # --- backend consistency ---
        if not isinstance(backend, Backend):
            raise TypeError("backend must be Backend instance") 
        
        # --- lattice consistency ---
        if not isinstance(lattice, Lattice):
            raise TypeError("lattice must be Lattice instance")
        
        # --- source compatibility ---
        if not isinstance(source, Source):
            raise ValueError(
                "source must be Source instance"
            )
        
        # --- layers container ---
        if not isinstance(layers, Sequence):
            raise TypeError("layers must be a sequence of Layer objects")

        if len(layers) < 2:
            raise ValueError("RCWA model must contain at least 2 layers (incident + substrate)")

        # --- layer type check ---
        for i, layer in enumerate(layers):
            if not isinstance(layer, Layer):
                raise TypeError(f"layers[{i}] is not a Layer instance")

        # --- incident medium ---
        inc = layers[0]
        if not inc.is_homogeneous(backend, lattice):
            raise ValueError("Incident layer must be homogeneous")
        if not inc.is_semi_infinite:
            raise ValueError("Incident layer must be semi-infinite (thickness=None)")

        # --- substrate ---
        sub = layers[-1]
        if not sub.is_homogeneous(backend, lattice):
            raise ValueError("Substrate layer must be homogeneous")
        if not sub.is_semi_infinite:
            raise ValueError("Substrate layer must be semi-infinite (thickness=None)")

        # --- internal layers ---
        for i, layer in enumerate(layers[1:-1], start=1):
            if layer.is_semi_infinite:
                raise ValueError(
                    f"Internal layer {i} cannot be semi-infinite"
                )
                
        # --- wvl dim compatible across layers ---
        wvl_sizes = [layer.epsilon_xy(backend, lattice).shape[0] for layer in layers]
        if len(set(wvl_sizes)) != 1:
            raise ValueError(
                "All layers must have the same wavelength dimension in their material properties"
            )
            
         # --- wvl dim compatible for layers and source ---
        if source.k0(backend).shape[0] != wvl_sizes[0]:
            raise ValueError(
                "Source wavelength dimension must match layers' material properties"
            )
        
            
        