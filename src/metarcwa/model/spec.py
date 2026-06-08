# metarcwa/model/spec.py
# DESCRIPTION

from dataclasses import dataclass
import torch

from .lattice import Lattice


@dataclass(frozen=True)
class StackSpec:
    """Resolved structure crossing the Model -> Solver boundary.

    An immutable snapshot of the stack after geometry has been rasterized
    and materials evaluated at the source wavelength. The Solver reads this
    and cannot reach back into the Model.

    Attributes
    ----------
    layer_eps : Tensor
        Per-layer permittivity on the grid, shape [N_layers, ..., Nx, Ny].
        Leading dims after N_layers may carry a geometry/wavelength batch.
    layer_thickness : Tensor
        Thickness of each finite layer, shape [N_layers] (broadcastable
        against any batch).
    eps_incidence : Tensor
        Permittivity of the semi-infinite incidence medium (uniform).
    eps_transmission : Tensor
        Permittivity of the semi-infinite transmission medium (uniform).
    lattice : Lattice
        In-plane periodicity, needed for reciprocal-lattice construction.
    """

    layer_eps: torch.Tensor
    layer_thickness: torch.Tensor
    eps_incidence: torch.Tensor
    eps_transmission: torch.Tensor
    lattice: Lattice


@dataclass(frozen=True)
class SourceSpec:
    """Abstract base for the solver-facing illumination description.

    A SourceSpec carries the *physical*, pre-expansion description of the
    illumination. Expansion onto Fourier harmonics is the Solver's job, so
    nothing solver-side (e.g. the harmonic count) appears here. Concrete
    variants — PlaneWaveSpec, and beam specs in future — subclass this.
    """

    wavelength: torch.Tensor


@dataclass(frozen=True)
class PlaneWaveSpec(SourceSpec):
    """Plane-wave illumination.

    Attributes
    ----------
    wavelength : Tensor
        Free-space wavelength (inherited). May be batched.
    kx0, ky0 : Tensor
        In-plane wavevector components of the incident wave, already
        resolved using the incidence-medium index.
    s, p : Tensor
        Complex s- and p-polarization amplitudes.
    """

    kx0: torch.Tensor
    ky0: torch.Tensor
    s: torch.Tensor
    p: torch.Tensor
    
@dataclass(frozen=True)
class ModelSpec:
    """Complete Model output: structure + illumination, for the Solver."""
    stack: StackSpec
    source: SourceSpec