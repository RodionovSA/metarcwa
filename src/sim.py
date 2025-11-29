from src.layer import Layer
from src.source import Source

class Eigensolver:
    def __init__(self, layer: Layer, source: Source):
        """
        Initialize the Eigensolver object with given layer and source.
        
        Parameters
        ----------
        layer : Layer
            The layer object defining the RCWA structure.
        source : Source
            The source object defining the incident wave parameters.
        """
        self._init_validation(layer, source)
        self._layer = layer
        self._source = source
        
    @staticmethod
    def _init_validation(layer: Layer, source: Source) -> None:
        if not isinstance(layer, Layer):
            raise TypeError("layer must be an instance of Layer.")
        if not isinstance(source, Source):
            raise TypeError("source must be an instance of Source.")
        if layer.backend != source.backend:
            raise ValueError("layer and source must use the same backend.")
        
        if len(source.wavelength) not in (len(layer.epsilon), len(layer.mu), len(layer.epsilon_bg), len(layer.mu_bg)):
            raise ValueError(
                "Length of source.wavelength must match length of layer.epsilon, layer.mu, layer.epsilon_bg, and layer.mu_bg."
            )
        