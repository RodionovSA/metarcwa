# tests/solver/test_pml_stretch.py
# Tests for the stretched wavevector operators of a lateral PML
# (metarcwa.solver.pml.stretched_k) and their use for patterned layers
# (build_layersolver / LayerSolver._patterned). See docs/PML.md.

import pytest
import torch
from torch.testing import assert_close

from metarcwa import Config, IsotropicMedium, Lattice, Layer, Model, PML, Source, Stack
from metarcwa.model.nn_helpers import CallableModule
from metarcwa.model.pml import PMLSpec, StretchAbsorbProfile
from metarcwa.solver.base import build_layersolver
from metarcwa.solver.blockmatrix import Block
from metarcwa.solver.convolution import convolution_matrix
from metarcwa.solver.harmonics import harmonic_index_map
from metarcwa.solver.pml import stretched_k


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _unit_profile():
    """s(d) = 1: a PML that does nothing."""
    return CallableModule(lambda d: torch.ones_like(d) + 0j)


def _spec(width=(0.25, 0.25), nx=64, ny=64, profile=None) -> PMLSpec:
    pml = PML(width=width, profile=profile).to(torch.float64)
    return pml.spec(Lattice.rectangular(1.0, 1.0).to(torch.float64), nx, ny)


def _medium(v):
    return IsotropicMedium(CallableModule(lambda wl: torch.full_like(wl, v, dtype=torch.complex128)))


def _disk(lat, nx, ny):
    """Centred disk of radius 0.2 in a unit cell: stays clear of a 0.25 PML."""
    x = (torch.arange(nx, dtype=torch.float64) + 0.5) / nx - 0.5
    y = (torch.arange(ny, dtype=torch.float64) + 0.5) / ny - 0.5
    return ((x[None, :] ** 2 + y[:, None] ** 2) < 0.2 ** 2).double()


def _config(**kw):
    base = dict(dtype=torch.float64, nx=64, ny=64, m=6, n=6,
                truncation="rectangular", factorization=None)
    base.update(kw)
    return Config(**base)


def _layersolver(pml, config, theta=0.0, phi=0.0):
    """LayerSolver + the patterned-layer spec of a one-layer disk stack."""
    layer = Layer(medium_solid=_medium(4.0), medium_void=_medium(1.0),
                  shape_fn=CallableModule(_disk), thickness=0.3)
    stack = Stack(incidence=_medium(1.0), layers=[layer], transmission=_medium(2.25),
                  lattice=Lattice.rectangular(1.0, 1.0), pml=pml)
    model = Model(stack, Source(wavelength=0.8, theta=theta, phi=phi))
    model = model.to(dtype=config.dtype, device=config.device)
    spec = model.spec(config.nx, config.ny)
    return build_layersolver(spec, config), spec


def _dense(K: Block, n: int) -> torch.Tensor:
    return K.to(Block.DENSE, n).data


# ---------------------------------------------------------------------------
# stretched_k
# ---------------------------------------------------------------------------

class TestStretchedK:

    def _k(self, M=4, N=4, circular=False):
        m, n = harmonic_index_map(M, N, circular, "cpu")
        g = torch.Generator().manual_seed(0)
        kx = 0.3 * torch.randn(m.shape[0], dtype=torch.float64, generator=g)
        ky = 0.3 * torch.randn(m.shape[0], dtype=torch.float64, generator=g)
        return kx, ky, m, n

    def test_disabled_axis_unchanged(self):
        kx, ky, m, n = self._k()
        Kx, Ky = stretched_k(kx, ky, _spec(width=(0.25, 0.0)), m, n)
        assert Kx.kind == Block.DENSE and Ky.kind == Block.DIAG
        assert_close(Ky.data, ky)

    @pytest.mark.parametrize("factorization", ["laurent", "inverse"])
    def test_unit_profile_gives_unstretched_k(self, factorization):
        kx, ky, m, n = self._k()
        Kx, Ky = stretched_k(kx, ky, _spec(profile=_unit_profile()), m, n, factorization)
        assert_close(Kx.data, torch.diag(kx).to(Kx.data.dtype), atol=1e-14, rtol=0)
        assert_close(Ky.data, torch.diag(ky).to(Ky.data.dtype), atol=1e-14, rtol=0)

    def test_laurent_matches_2d_convolution_matrix(self):
        """[[1/s_x]] equals the existing 2-D convolution matrix of a grid
        that varies only along x."""
        kx, ky, m, n = self._k()
        spec = _spec()
        Kx, Ky = stretched_k(kx, ky, spec, m, n)
        grid_x = (1 / spec.sx)[None, :].expand(64, 64)
        grid_y = (1 / spec.sy)[:, None].expand(64, 64)
        assert_close(Kx.data, convolution_matrix(grid_x, m, n) * kx[None, :])
        assert_close(Ky.data, convolution_matrix(grid_y, m, n) * ky[None, :])

    def test_couples_own_index_only(self):
        kx, ky, m, n = self._k()
        Kx, Ky = stretched_k(kx, ky, _spec(), m, n)
        assert (Kx.data[n[:, None] != n[None, :]] == 0).all()
        assert (Ky.data[m[:, None] != m[None, :]] == 0).all()

    def test_zeroth_harmonic_is_unaffected_at_normal_incidence(self):
        """kx0 = 0: K e0 = 0, hence K̃ e0 = 0 (the stretch does nothing)."""
        m, n = harmonic_index_map(4, 4, False, "cpu")
        kx, ky = m.double(), n.double()               # kx0 = ky0 = 0, unit periods (k0-scaled)
        Kx, Ky = stretched_k(kx, ky, _spec(), m, n)
        e0 = ((m == 0) & (n == 0)).nonzero().item()
        assert Kx.data[:, e0].abs().max() == 0
        assert Ky.data[:, e0].abs().max() == 0

    def test_commute_for_rectangular_truncation(self):
        """Rectangular lattice: kx depends on m only, ky on n only."""
        m, n = harmonic_index_map(4, 4, False, "cpu")
        kx, ky = 0.1 + 1.3 * m.double(), -0.2 + 0.7 * n.double()
        Kx, Ky = stretched_k(kx, ky, _spec(), m, n)
        assert_close(Kx.data @ Ky.data, Ky.data @ Kx.data, atol=1e-13, rtol=0)

    def test_laurent_and_inverse_converge(self):
        """For a continuous profile [[1/s]] and [[s]]^-1 agree in the limit."""
        diffs = []
        for M in (8, 32):
            m, n = harmonic_index_map(M, 0, False, "cpu")
            one = torch.ones(m.shape[0], dtype=torch.float64)
            spec = _spec(width=(0.25, 0.0), nx=8 * (2 * M + 1), ny=8)
            KL, _ = stretched_k(one, one, spec, m, n, "laurent")
            KI, _ = stretched_k(one, one, spec, m, n, "inverse")
            c = m.abs() <= 2
            diffs.append((KL.data[c][:, c] - KI.data[c][:, c]).abs().max())
        assert diffs[1] < 0.05 * diffs[0]

    def test_batched_wavevectors(self):
        kx, ky, m, n = self._k()
        kxb, kyb = torch.stack([kx, 2 * kx]), torch.stack([ky, 2 * ky])
        Kx, Ky = stretched_k(kxb, kyb, _spec(), m, n)
        assert Kx.data.shape == (2, m.shape[0], m.shape[0])
        assert_close(Kx.data[1], 2 * Kx.data[0])


# ---------------------------------------------------------------------------
# build_layersolver
# ---------------------------------------------------------------------------

class TestBuildLayerSolver:

    def test_oblique_along_pml_axis_raises(self):
        with pytest.raises(NotImplementedError, match="Oblique incidence along the PML axis x"):
            _layersolver(PML(width=0.25), _config(), theta=0.3, phi=0.0)

    def test_tilt_along_non_pml_axis(self):
        """x-only PML, plane of incidence along y: kx0 (~1e-17 from cos(pi/2))
        is zeroed, ky0 kept; Kx stretched, Ky diagonal."""
        cfg = _config()
        ls, _ = _layersolver(PML(width=(0.25, 0.0)), cfg, theta=0.3, phi=torch.pi / 2)
        ref, _ = _layersolver(None, cfg, theta=0.3, phi=torch.pi / 2)
        assert ls.Kx.kind == Block.DENSE and ls.Ky.kind == Block.DIAG
        assert_close(ls.ky, ref.ky)
        assert ls.kx[..., (ls.m_flat == 0)].abs().max() == 0

    def test_homogeneous_elements_not_supported_yet(self):
        ls, spec = _layersolver(PML(width=0.25), _config())
        with pytest.raises(NotImplementedError, match="PML"):
            ls.prepare(spec.incidence)

    def test_no_pml_keeps_diagonal_k(self):
        ls, _ = _layersolver(None, _config())
        assert not ls.stretched
        assert ls.Kx.kind == Block.DIAG and ls.Ky.kind == Block.DIAG


# ---------------------------------------------------------------------------
# Patterned layer under PML
# ---------------------------------------------------------------------------

class TestPatternedLayer:

    @pytest.mark.parametrize("modesolver", ["eig", "matexp"])
    def test_unit_profile_matches_periodic(self, modesolver):
        """A PML with s = 1 (and a pattern clear of the PML, so padding is a
        no-op) reproduces the periodic S-matrix."""
        cfg = _config(modesolver=modesolver)
        ls0, spec0 = _layersolver(None, cfg)
        ls1, spec1 = _layersolver(PML(width=0.25, profile=_unit_profile()), cfg)
        nh = ls0.m_flat.shape[0]
        S0 = ls0.run(spec0.layers[0]).to_dense(nh)
        S1 = ls1.run(spec1.layers[0]).to_dense(nh)
        assert_close(S1, S0, atol=1e-12, rtol=0)

    @pytest.mark.parametrize("truncation", ["rectangular", "circular"])
    def test_pml_makes_all_modes_lossy(self, truncation):
        """With the PML every mode decays (Re lam < 0), clearly more than the
        regularization-level loss of the periodic layer."""
        cfg = _config(truncation=truncation)
        ls0, spec0 = _layersolver(None, cfg)
        ls1, spec1 = _layersolver(PML(width=0.25), cfg)
        lam0 = ls0.prepare(spec0.layers[0]).lam
        lam1 = ls1.prepare(spec1.layers[0]).lam
        assert lam1.real.max() < 0
        assert lam1.real.max() < 100 * lam0.real.max()

    def test_eig_and_matexp_agree(self):
        S = {}
        for modesolver in ("eig", "matexp"):
            ls, spec = _layersolver(PML(width=0.25), _config(modesolver=modesolver))
            S[modesolver] = ls.run(spec.layers[0]).to_dense(ls.m_flat.shape[0])
        assert_close(S["matexp"], S["eig"], atol=1e-10, rtol=0)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_runs_on_cuda(self):
        ls, spec = _layersolver(PML(width=0.25), _config(device="cuda", m=4, n=4, nx=32, ny=32))
        lam = ls.prepare(spec.layers[0]).lam
        assert ls.Kx.data.device.type == "cuda" and lam.device.type == "cuda"
        assert lam.real.max() < 0

    def test_gradient_wrt_pml_parameter(self):
        sigma = torch.nn.Parameter(torch.tensor(1.0, dtype=torch.float64))
        pml = PML(width=0.25, profile=StretchAbsorbProfile(sigma_max=sigma))
        ls, spec = _layersolver(pml, _config(m=3, n=3, nx=32, ny=32))
        S = ls.run(spec.layers[0]).to_dense(ls.m_flat.shape[0])
        S.abs().sum().backward()
        assert sigma.grad is not None and torch.isfinite(sigma.grad) and sigma.grad != 0


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

class TestConfig:

    def test_default_is_laurent(self):
        assert Config().pml_factorization == "laurent"

    def test_invalid_value_raises(self):
        with pytest.raises(ValueError, match="pml_factorization"):
            Config(pml_factorization="direct")

    def test_dict_round_trip(self):
        cfg = Config(pml_factorization="inverse")
        assert Config.from_dict(cfg.to_dict()).pml_factorization == "inverse"
