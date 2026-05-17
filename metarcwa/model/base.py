# metarcwa/model/base.py
# DESCRIPTION

import torch
import torch.nn as nn

from .stack import Stack
from .source import Source
from .spec import ModelSpec

class Model(nn.Module):
    """
    Simulation model for RCWA.  
    """
    def __init__(self, stack: Stack, source: Source):
        super().__init__()
        self.stack = stack
        self.source = source
        
    def spec(self) -> ModelSpec:
        """Build the complete Model -> Solver description."""
        wavelength = self.source.wavelength

        stack_spec = self.stack.spec(wavelength)
        source_spec = self.source.spec(stack_spec.eps_incidence)

        return ModelSpec(stack=stack_spec, source=source_spec)
    