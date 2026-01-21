# src/model/model.py
# Class for storing the simulation model
from typing import List, Sequence

from src.model.layer import Layer
from src.model.source import Source

class Model:
    """ RCWA model """
    def __init__(self, layers: List["Layer"], source: "Source"):
        self._init_validation(layers, source)
        
        self._layers = layers
        self._source = source
        
    """ Simulation properties """
    @property
    def backend(self):
        return self.source.backend
    @property
    def lattice(self):
        return self.layers[0].lattice
    @property
    def layers(self):
        return self._layers
    @property
    def source(self):
        return self._source
    @property
    def n_inc(self):
        return self.backend.asarray(self.layers[0].epsilon, complex=False)
    
    """ Static helper methods """
    @staticmethod
    def _init_validation(layers, source):
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
        if not inc.is_homogeneous():
            raise ValueError("Incident layer must be homogeneous")
        if not inc.is_semi_infinite():
            raise ValueError("Incident layer must be semi-infinite (thickness=None)")

        # --- substrate ---
        sub = layers[-1]
        if not sub.is_homogeneous():
            raise ValueError("Substrate layer must be homogeneous")
        if not sub.is_semi_infinite():
            raise ValueError("Substrate layer must be semi-infinite (thickness=None)")

        # --- internal layers ---
        for i, layer in enumerate(layers[1:-1], start=1):
            if layer.is_semi_infinite():
                raise ValueError(
                    f"Internal layer {i} cannot be semi-infinite"
                )

        # --- source compatibility ---
        if not isinstance(source, Source):
            raise ValueError(
                "source must be Source instance"
            )
            
        # --- backend consistency ---
        backends = {layer.backend for layer in layers}

        if len(backends) != 1:
            raise ValueError(
                "All layers must use the same backend. "
                f"Got backends: {backends}"
            )
            
        if source.backend not in backends:
            raise ValueError(
                "Source backend must match layer backend. "
                f"Source backend: {source.backend}, "
                f"Layer backend(s): {backends}"
            )
        
        # --- lattice consistency ---
        ref = layers[0].lattice

        for layer in layers:
            if layer.lattice is not ref:
                raise ValueError("Lattice object mismatch")