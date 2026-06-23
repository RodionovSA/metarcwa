# metarcwa/model/stack.py
# DESCRIPTION

from typing import Sequence
import torch
import torch.nn as nn
from dataclasses import dataclass

from .layer import Layer, HomogeneousLayer, PatternedLayer
from .lattice import Lattice
from .medium import Medium, MediumSpec

@dataclass(frozen=True)
class StackSpec:
    """Resolved structure crossing the Model -> Solver boundary.

    An immutable snapshot of the stack after geometry has been rasterized
    and materials evaluated at the source wavelength. The Solver reads this
    and cannot reach back into the Model.

    Layers are stored as a heterogeneous tuple of per-layer specs
    (``HomogeneousLayer`` / ``PatternedLayer``), not a single stacked eps
    tensor: layers may differ in spatial kind (uniform vs patterned) and
    cannot be forced into one uniform array. The incidence and transmission
    media are bare ``MediumSpec`` values — semi-infinite.

    Attributes
    ----------
    layers : tuple[HomogeneousLayer | PatternedLayer, ...]
        Ordered finite layers, incidence side first. Each entry already
        carries its own resolved permittivity (gridded for patterned layers,
        bulk for homogeneous) and thickness. Per-layer eps is **always
        complex** (promoted in ``Medium.spec()``).
    incidence : MediumSpec
        Material spec for the semi-infinite incidence medium (uniform).
        Stored as resolved; losslessness is not enforced here. Consumers
        that need a real refractive index take the real part themselves.
    transmission : MediumSpec
        Material spec for the semi-infinite transmission medium (uniform).
        May be lossy.
    a1 : Tensor | nn.Parameter
        First lattice vector, shape [2]. Needed for reciprocal-lattice
        construction.
    a2 : Tensor | nn.Parameter
        Second lattice vector, shape [2]. Same convention as ``a1``.
    """

    layers: tuple["HomogeneousLayer | PatternedLayer", ...]
    incidence: "MediumSpec"
    transmission: "MediumSpec"
    a1: torch.Tensor
    a2: torch.Tensor

class Stack(nn.Module):
    """An ordered stack of finite layers between two semi-infinite media.

    Owns the system-wide invariant shared by all layers: the lattice. Grid
    resolution is supplied per spec() call.

    The incidence and transmission media are semi-infinite half-spaces: they
    have permittivity but no thickness and no pattern, so they are stored
    separately from the finite layers, as bare Medium instances.

    Parameters
    ----------
    incidence : Medium
        Semi-infinite incidence medium. Should be lossless for
        well-defined mode classification, but this is **not** checked or
        enforced — its imaginary part is simply ignored by the consumers.
    layers : Sequence[Layer]
        Ordered finite layers, incidence side first.
    transmission : Medium
        Semi-infinite transmission medium. May be lossy.
    lattice : Lattice
        In-plane periodicity, shared by all layers.
    """

    def __init__(
        self,
        incidence: Medium,
        layers: Sequence[Layer],
        transmission: Medium,
        lattice: Lattice,
    ):
        super().__init__()
        if not isinstance(incidence, Medium):
            raise TypeError("incidence must be a Medium.")
        if not isinstance(transmission, Medium):
            raise TypeError("transmission must be a Medium.")

        self.incidence = incidence
        self.layers = nn.ModuleList(layers)
        self.transmission = transmission
        self.lattice = lattice

    def spec(
        self,
        wavelength: torch.Tensor,
        nx: int,
        ny: int,
    ) -> StackSpec:
        """Resolve the stack at a given wavelength and grid resolution.

        Parameters
        ----------
        wavelength : Tensor
            Wavelength batch, forwarded to every Medium's eps_fn.
        nx : int
            Grid resolution along x, for any patterned layer's shape_fn.
            Ignored by homogeneous layers.
        ny : int
            Grid resolution along y, same caveat as nx.
        """
        layer_specs = [
            layer.spec(wavelength, self.lattice, nx, ny) for layer in self.layers
        ]
        incidence_spec = self.incidence.spec(wavelength)
        transmission_spec = self.transmission.spec(wavelength)

        return StackSpec(
            layers=layer_specs,
            incidence=incidence_spec,
            transmission=transmission_spec,
            a1=self.lattice.a1,
            a2=self.lattice.a2,
        )


