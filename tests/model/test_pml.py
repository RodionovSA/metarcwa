# tests/model/test_pml.py
# Tests for metarcwa.model.pml (PML, PMLSpec) and its integration in Stack /
# Model.spec: construction checks, stretch profiles, edge padding.

import pytest
import torch
from torch.testing import assert_close

from metarcwa import Model, Source
from metarcwa.model.lattice import Lattice
from metarcwa.model.layer import Layer, HomogeneousLayer, PatternedLayer
from metarcwa.model.medium import IsotropicMedium
from metarcwa.model.nn_helpers import CallableModule
from metarcwa.model.pml import PML, PMLSpec, PolynomialProfile, StretchAbsorbProfile
from metarcwa.model.stack import Stack


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _medium(value):
    val = torch.as_tensor(value, dtype=torch.complex64)
    return IsotropicMedium(CallableModule(lambda wl: val.expand(wl.shape)))


def _stack(pml=None, lattice=None, mask=None):
    """Stack with one homogeneous and one patterned layer."""
    lattice = lattice if lattice is not None else Lattice.rectangular(1.0, 2.0)
    mask = mask if mask is not None else torch.rand(16, 8, generator=torch.Generator().manual_seed(0))
    layers = [
        Layer(medium_solid=_medium(2.25), thickness=0.1),
        Layer(medium_solid=_medium(12.0), medium_void=_medium(1.0),
              shape_fn=CallableModule(lambda lat, nx, ny: mask), thickness=0.2),
    ]
    return Stack(incidence=_medium(1.0), layers=layers, transmission=_medium(2.25),
                 lattice=lattice, pml=pml)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

class TestConstruction:

    def test_scalar_width_applies_to_both_axes(self):
        assert_close(PML(width=0.1).width, torch.tensor([0.1, 0.1]))

    def test_tuple_width(self):
        assert_close(PML(width=(0.1, 0.0)).width, torch.tensor([0.1, 0.0]))

    @pytest.mark.parametrize("width", [
        torch.nn.Parameter(torch.tensor(0.1)),
        torch.tensor(0.1, requires_grad=True),
        torch.nn.Parameter(torch.tensor([0.1])),
    ])
    def test_scalar_grad_width_raises(self, width):
        with pytest.raises(TypeError, match="plain scalar"):
            PML(width=width)

    def test_two_axis_parameter_width_is_optimizable(self):
        width = torch.nn.Parameter(torch.tensor([0.1, 0.2]))
        pml = PML(width=width)
        assert pml.width is width
        assert any(p is width for p in pml.parameters())

    def test_default_profile(self):
        assert isinstance(PML(width=0.1).profile, StretchAbsorbProfile)

    @pytest.mark.parametrize("kwargs", [
        dict(width=-0.1),
        dict(width=0.0),
        dict(width=(0.1, 0.2, 0.3)),
        dict(width=0.1, method="upml"),
    ])
    def test_invalid_arguments_raise(self, kwargs):
        with pytest.raises(ValueError):
            PML(**kwargs)

    def test_profile_must_be_module(self):
        with pytest.raises(TypeError, match="nn.Module"):
            PML(width=0.1, profile=lambda d: 1 + 1j * d)

    @pytest.mark.parametrize("make", [
        lambda: StretchAbsorbProfile(a_max=-1.0),
        lambda: StretchAbsorbProfile(p=0.0),
        lambda: StretchAbsorbProfile(sigma_max=0.0),
        lambda: PolynomialProfile(sigma_max=0.0),
        lambda: PolynomialProfile(p=0.0),
    ])
    def test_invalid_profile_arguments_raise(self, make):
        with pytest.raises(ValueError):
            make()

    def test_non_rectangular_lattice_raises(self):
        with pytest.raises(ValueError, match="rectangular"):
            _stack(pml=PML(width=0.1), lattice=Lattice.hexagonal(1.0))

    def test_width_too_large_raises(self):
        # cell is 1.0 x 2.0: 2 * 0.5 >= 1.0 leaves no physical region along x
        with pytest.raises(ValueError, match="no physical region"):
            _stack(pml=PML(width=(0.5, 0.1)))


# ---------------------------------------------------------------------------
# Stretch profiles
# ---------------------------------------------------------------------------

class TestProfile:

    def _spec(self, width=(0.2, 0.3), nx=64, ny=96, profile=None):
        return PML(width=width, profile=profile).spec(Lattice.rectangular(1.0, 2.0), nx, ny)

    def test_shapes_and_dtype(self):
        spec = self._spec()
        assert isinstance(spec, PMLSpec)
        assert spec.sx.shape == (64,) and spec.sy.shape == (96,)
        assert spec.sx.is_complex() and spec.sy.is_complex()

    def test_unit_in_physical_region(self):
        """s == 1 exactly where |x| <= P/2 - w (pixel centres)."""
        nx = 64
        spec = self._spec(nx=nx)
        x = (torch.arange(nx) + 0.5) / nx - 0.5
        physical = x.abs() <= 0.5 - 0.2
        assert_close(spec.sx[physical], torch.ones_like(spec.sx[physical]))
        assert (spec.sx[~physical] != 1).all()

    def test_absorbing_and_stretching_inside(self):
        spec = self._spec()
        for s in (spec.sx, spec.sy):
            inside = s != 1
            assert (s.imag[inside] > 0).all()
            assert (s.real[inside] >= 1).all()

    def test_symmetric_about_cell_centre(self):
        """Symmetric profile: the two PMLs meeting at the cell edge are continuous."""
        spec = self._spec()
        assert_close(spec.sx, spec.sx.flip(0))
        assert_close(spec.sy, spec.sy.flip(0))

    def test_continuous_profile(self):
        """Largest jump between neighbouring pixels shrinks with resolution."""
        jumps = [self._spec(nx=n).sx.diff().abs().max() for n in (64, 128, 256)]
        assert jumps[1] < 0.6 * jumps[0] and jumps[2] < 0.6 * jumps[1]

    def test_disabled_axis_is_none(self):
        spec = self._spec(width=(0.2, 0.0))
        assert spec.sx is not None and spec.sy is None

    @staticmethod
    def _d(nx=64, w=0.2):
        x = (torch.arange(nx) + 0.5) / nx - 0.5
        return ((x.abs() - (0.5 - w)) / w).clamp(0, 1)

    def test_stretch_absorb_matches_formula(self):
        """(1 + a d^p)(1 + j sigma sin^2(pi d/2)) at every pixel."""
        spec = self._spec(profile=StretchAbsorbProfile(a_max=3.0, p=2.0, sigma_max=1.5))
        d = self._d()
        expected = (1 + 3.0 * d**2) * (1 + 1j * 1.5 * torch.sin(torch.pi / 2 * d) ** 2)
        assert_close(spec.sx, expected.to(spec.sx.dtype))

    def test_polynomial_matches_formula(self):
        """1 + j sigma d^p at every pixel (no real stretch)."""
        spec = self._spec(profile=PolynomialProfile(sigma_max=2.0, p=3.0))
        d = self._d()
        assert_close(spec.sx, (1 + 1j * 2.0 * d**3).to(spec.sx.dtype))
        assert_close(spec.sx.real, torch.ones_like(spec.sx.real))

    def test_user_defined_profile(self):
        a = torch.nn.Parameter(torch.tensor(0.5))
        profile = CallableModule(lambda d: 1 + (1 + 1j) * a * d, a)
        spec = self._spec(profile=profile)
        assert_close(spec.sx, (1 + (1 + 1j) * 0.5 * self._d()).to(spec.sx.dtype))

    @pytest.mark.parametrize("fn", [
        lambda d: 2 + 1j * d,            # s(0) != 1
        lambda d: 1 - 1j * d,            # Im s < 0 (amplifying)
        lambda d: 1 - 2 * d + 0j,        # Re s <= 0
    ])
    def test_invalid_user_profile_raises(self, fn):
        with pytest.raises(ValueError, match="PML profile"):
            self._spec(profile=CallableModule(fn))

    def test_gradient_wrt_profile_parameter(self):
        sigma = torch.nn.Parameter(torch.tensor(1.0))
        pml = PML(width=0.2, profile=StretchAbsorbProfile(sigma_max=sigma))
        pml.spec(Lattice.rectangular(1.0, 1.0), 32, 32).sx.imag.sum().backward()
        assert sigma.grad is not None and sigma.grad > 0


# ---------------------------------------------------------------------------
# Edge padding
# ---------------------------------------------------------------------------

class TestPad:

    def _pml_lattice(self, width=(0.25, 0.25)):
        return PML(width=width), Lattice.rectangular(1.0, 1.0)

    def test_physical_region_unchanged_and_pml_padded(self):
        pml, lat = self._pml_lattice()
        n = 16                                   # w = P/4 -> 4 PML pixels per side
        pattern = torch.rand(n, n, generator=torch.Generator().manual_seed(1))
        out = pml.pad(pattern, lat)
        assert_close(out[4:12, 4:12], pattern[4:12, 4:12])
        # columns left/right of the physical region copy its edge column
        for j in range(4):
            assert_close(out[4:12, j], pattern[4:12, 4])
            assert_close(out[4:12, n - 1 - j], pattern[4:12, 11])
        # rows above/below, and corners from the physical corner pixel
        assert_close(out[0, 4:12], pattern[4, 4:12])
        assert_close(out[n - 1, 4:12], pattern[11, 4:12])
        assert out[0, 0] == pattern[4, 4] and out[-1, -1] == pattern[11, 11]

    def test_pattern_constant_along_normal_inside_pml(self):
        pml, lat = self._pml_lattice()
        out = pml.pad(torch.rand(16, 16), lat)
        assert (out[:, :4] == out[:, 4:5]).all()
        assert (out[:4, :] == out[4:5, :]).all()

    def test_disabled_axis_not_padded(self):
        pml, lat = self._pml_lattice(width=(0.25, 0.0))
        pattern = torch.rand(16, 16)
        out = pml.pad(pattern, lat)
        assert_close(out[:, 4:12], pattern[:, 4:12])

    def test_batched_pattern(self):
        pml, lat = self._pml_lattice()
        pattern = torch.rand(3, 16, 16)
        out = pml.pad(pattern, lat)
        assert out.shape == pattern.shape
        for b in range(3):
            assert_close(out[b], pml.pad(pattern[b], lat))

    def test_gradient_flows_through_padding(self):
        pml, lat = self._pml_lattice()
        pattern = torch.rand(16, 16, requires_grad=True)
        pml.pad(pattern, lat).sum().backward()
        # an edge pixel of the physical region is copied 4 times + itself
        assert pattern.grad[8, 4] == 5
        assert pattern.grad[0, 0] == 0          # PML pixels do not contribute


# ---------------------------------------------------------------------------
# Stack / Model integration
# ---------------------------------------------------------------------------

class TestStackIntegration:

    def test_no_pml_keeps_spec_unchanged(self):
        spec = _stack().spec(torch.tensor([1.0]), 8, 16)
        assert spec.pml is None

    def test_spec_pads_patterned_layers_only(self):
        mask = torch.rand(16, 8, generator=torch.Generator().manual_seed(2))
        pml = PML(width=(0.25, 0.5))
        stack = _stack(pml=pml, mask=mask)
        spec = stack.spec(torch.tensor([1.0]), 8, 16)

        assert isinstance(spec.pml, PMLSpec)
        assert spec.pml.sx.shape == (8,) and spec.pml.sy.shape == (16,)
        assert isinstance(spec.layers[0], HomogeneousLayer)
        assert isinstance(spec.layers[1], PatternedLayer)
        assert_close(spec.layers[1].pattern, pml.pad(mask, stack.lattice))

    def test_model_spec_carries_pml(self):
        model = Model(_stack(pml=PML(width=0.25)), Source(wavelength=1.0))
        assert isinstance(model.spec(8, 16).pml, PMLSpec)

    def test_model_to_float64(self):
        model = Model(_stack(pml=PML(width=0.25)), Source(wavelength=1.0)).to(torch.float64)
        assert model.stack.pml.width.dtype == torch.float64
        assert model.stack.pml.profile.sigma_max.dtype == torch.float64
        assert model.spec(8, 16).pml.sx.dtype == torch.complex128

    def test_pml_in_parameters_when_optimizable(self):
        sigma = torch.nn.Parameter(torch.tensor(1.0))
        pml = PML(width=0.25, profile=StretchAbsorbProfile(sigma_max=sigma))
        model = Model(_stack(pml=pml), Source(wavelength=1.0))
        assert any(p is sigma for p in model.parameters())
