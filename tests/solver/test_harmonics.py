import torch
from torch.testing import assert_close
import pytest

from metarcwa.solver.harmonics import (
    compute_kxy,
    harmonic_index_map,
    harmonic_wavevectors,
    reciprocal_index_map,
    reciprocal_lattice_vectors,
)

TWO_PI = 2 * torch.pi


# ── fixtures / helpers ────────────────────────────────────────────────────────

def square_lattice():
    return torch.tensor([1.0, 0.0]), torch.tensor([0.0, 1.0])


def rhombic_lattice(angle_deg=60.0):
    theta = torch.tensor(angle_deg * torch.pi / 180.0)
    a1 = torch.tensor([1.0, 0.0])
    a2 = torch.stack([torch.cos(theta), torch.sin(theta)])
    return a1, a2


# ── compute_kxy ───────────────────────────────────────────────────────────────

class TestComputeKxy:

    def test_output_shape_scalar_batch(self):
        a1, a2 = square_lattice()
        m_flat, n_flat = harmonic_index_map(1, 1)
        kx, ky = compute_kxy(
            torch.tensor(0.0), torch.tensor(0.0), a1, a2, m_flat, n_flat
        )
        Nh = m_flat.shape[0]
        assert kx.shape == (Nh,)
        assert ky.shape == (Nh,)

    def test_output_shape_1d_batch(self):
        a1, a2 = square_lattice()
        m_flat, n_flat = harmonic_index_map(2, 2)
        kx0, ky0 = torch.zeros(5), torch.zeros(5)
        kx, ky = compute_kxy(kx0, ky0, a1, a2, m_flat, n_flat)
        assert kx.shape == (5, m_flat.shape[0])
        assert ky.shape == (5, m_flat.shape[0])

    def test_output_shape_2d_batch(self):
        a1, a2 = square_lattice()
        m_flat, n_flat = harmonic_index_map(1, 1)
        kx0, ky0 = torch.zeros(3, 4), torch.zeros(3, 4)
        kx, ky = compute_kxy(kx0, ky0, a1, a2, m_flat, n_flat)
        assert kx.shape == (3, 4, m_flat.shape[0])
        assert ky.shape == (3, 4, m_flat.shape[0])

    def test_square_lattice_normal_incidence(self):
        """kx_mn = m·2π, ky_mn = n·2π for a unit square lattice at normal incidence."""
        a1, a2 = square_lattice()
        m_flat, n_flat = harmonic_index_map(2, 2)
        kx, ky = compute_kxy(
            torch.tensor(0.0), torch.tensor(0.0), a1, a2, m_flat, n_flat
        )
        assert_close(kx, m_flat.float() * TWO_PI)
        assert_close(ky, n_flat.float() * TWO_PI)

    def test_oblique_incidence_shifts_all_harmonics_uniformly(self):
        """An offset kx0 must shift every harmonic by exactly kx0."""
        a1, a2 = square_lattice()
        m_flat, n_flat = harmonic_index_map(1, 1)
        kx_normal, _ = compute_kxy(
            torch.tensor(0.0), torch.tensor(0.0), a1, a2, m_flat, n_flat
        )
        kx_oblique, _ = compute_kxy(
            torch.tensor(0.3), torch.tensor(0.0), a1, a2, m_flat, n_flat
        )
        assert_close(kx_oblique - kx_normal, torch.full_like(kx_normal, 0.3))

    def test_agrees_with_manual_pipeline(self):
        """compute_kxy must equal the explicit three-step computation."""
        a1, a2 = square_lattice()
        m_flat, n_flat = harmonic_index_map(2, 2)
        kx0 = torch.tensor([0.1, 0.2])
        ky0 = torch.tensor([0.0, 0.05])

        kx_combined, ky_combined = compute_kxy(kx0, ky0, a1, a2, m_flat, n_flat)

        b1, b2 = reciprocal_lattice_vectors(a1, a2)
        Gx, Gy = reciprocal_index_map(m_flat, n_flat, b1, b2)
        kx_manual, ky_manual = harmonic_wavevectors(kx0, ky0, Gx, Gy)

        assert_close(kx_combined, kx_manual)
        assert_close(ky_combined, ky_manual)


class TestReciprocalLatticeVectors:

    @pytest.mark.parametrize("lattice_fn", [square_lattice, rhombic_lattice])
    def test_orthogonality_relations(self, lattice_fn):
        """b_i · a_j = 2π δ_{ij} must hold for any lattice."""
        a1, a2 = lattice_fn()
        b1, b2 = reciprocal_lattice_vectors(a1, a2)
        assert_close(torch.dot(b1, a1), torch.tensor(TWO_PI), atol=1e-5, rtol=1e-5)
        assert_close(torch.dot(b1, a2), torch.tensor(0.0),    atol=1e-5, rtol=1e-5)
        assert_close(torch.dot(b2, a1), torch.tensor(0.0),    atol=1e-5, rtol=1e-5)
        assert_close(torch.dot(b2, a2), torch.tensor(TWO_PI), atol=1e-5, rtol=1e-5)


class TestHarmonicIndexMap:

    def test_rectangular_count(self):
        m_flat, n_flat = harmonic_index_map(2, 3)
        assert m_flat.shape[0] == 5 * 7

    def test_circular_fewer_than_rectangular(self):
        m_flat_rect, _ = harmonic_index_map(3, 3, circular=False)
        m_flat_circ, _ = harmonic_index_map(3, 3, circular=True)
        assert m_flat_circ.shape[0] < m_flat_rect.shape[0]

    def test_circular_all_within_ellipse(self):
        m_max, n_max = 3, 2
        m_flat, n_flat = harmonic_index_map(m_max, n_max, circular=True)
        r2 = (m_flat.float() / m_max) ** 2 + (n_flat.float() / n_max) ** 2
        assert (r2 <= 1.0).all()
