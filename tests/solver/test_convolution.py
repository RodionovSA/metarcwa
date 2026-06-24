import torch
from torch.testing import assert_close

from metarcwa.solver.convolution import convolution_matrix
from metarcwa.solver.harmonics import harmonic_index_map


# ── helpers ───────────────────────────────────────────────────────────────────

def rect_indices(m_max, n_max):
    return harmonic_index_map(m_max, n_max, circular=False)


# ── convolution_matrix ────────────────────────────────────────────────────────

class TestConvolutionMatrix:

    def test_output_shape_unbatched(self):
        m_flat, n_flat = rect_indices(2, 2)
        C = convolution_matrix(torch.ones(16, 16), m_flat, n_flat)
        assert C.shape == (m_flat.shape[0], m_flat.shape[0])

    def test_output_shape_batched(self):
        m_flat, n_flat = rect_indices(1, 1)
        B = 7
        C = convolution_matrix(torch.ones(B, 8, 8), m_flat, n_flat)
        assert C.shape == (B, m_flat.shape[0], m_flat.shape[0])

    def test_uniform_eps_gives_scaled_identity(self):
        """Constant eps has only a DC Fourier coefficient → C = eps · I."""
        eps_val = 3.7
        m_flat, n_flat = rect_indices(2, 2)
        Nh = m_flat.shape[0]
        C = convolution_matrix(torch.full((16, 16), eps_val), m_flat, n_flat)
        expected = (torch.eye(Nh) * eps_val).to(dtype=C.dtype)
        assert_close(C, expected, atol=1e-5, rtol=1e-5)

    def test_hermitian_for_real_eps(self):
        """For real-valued eps, the convolution matrix must be Hermitian: C = C†."""
        torch.manual_seed(0)
        eps = torch.rand(16, 16)
        m_flat, n_flat = rect_indices(2, 2)
        C = convolution_matrix(eps, m_flat, n_flat)
        assert_close(C, C.conj().mT, atol=1e-5, rtol=1e-5)

    def test_1d_sinusoidal_grating_known_coefficients(self):
        """
        eps(x) = 1 + 0.5·cos(2π x / Nx) has Fourier series coefficients:
          eps_hat[0]  = 1.0
          eps_hat[±1] = 0.25
        The convolution matrix diagonal must be 1.0 and adjacent entries 0.25.
        """
        Nx = 64
        x = torch.arange(Nx, dtype=torch.float)
        eps_1d = 1.0 + 0.5 * torch.cos(2 * torch.pi * x / Nx)
        # Ny = 1: only x-variation, dn = 0 for all harmonic pairs
        eps = eps_1d.unsqueeze(0)

        m_flat = torch.tensor([-1, 0, 1])
        n_flat = torch.zeros(3, dtype=torch.long)

        C = convolution_matrix(eps, m_flat, n_flat)

        coeff = torch.tensor(0.25, dtype=C.real.dtype)
        dc    = torch.tensor(1.0,  dtype=C.real.dtype)

        assert_close(C.diagonal().real, dc.expand(3), atol=1e-5, rtol=1e-5)
        assert_close(C[0, 1].real, coeff, atol=1e-5, rtol=1e-5)
        assert_close(C[1, 0].real, coeff, atol=1e-5, rtol=1e-5)
        assert_close(C[1, 2].real, coeff, atol=1e-5, rtol=1e-5)
        assert_close(C[2, 1].real, coeff, atol=1e-5, rtol=1e-5)
        # m=-1 and m=+1 are two orders apart — no coupling
        assert_close(C[0, 2].abs(), torch.tensor(0.0), atol=1e-5, rtol=0.0)

    def test_x_only_variation_decouples_n_harmonics(self):
        """
        eps varying only in x must produce zero coupling between harmonics
        with different n indices. This validates that dn indexes the y axis
        and dm indexes the x axis (not swapped).
        """
        Nx, Ny = 32, 32
        x = torch.arange(Nx, dtype=torch.float)
        eps_1d = 1.0 + 0.5 * torch.cos(2 * torch.pi * x / Nx)
        eps = eps_1d.unsqueeze(0).expand(Ny, Nx)

        m_flat, n_flat = rect_indices(1, 1)
        C = convolution_matrix(eps, m_flat, n_flat)

        # Build a mask for off-n pairs
        different_n = n_flat[:, None] != n_flat[None, :]
        off_n_entries = C[different_n]
        assert_close(off_n_entries.abs(), torch.zeros_like(off_n_entries.abs()),
                     atol=1e-5, rtol=0.0)
