from src.backend import TorchBackend
from src.kvector import compute_k0xy, compute_Kxy
import torch

''' Tests for k0xy computation '''

def test_k0xy_normal_incidence():
    backend = TorchBackend(device="cpu", dtype=torch.float32, use_compile=False)

    wavelengths = backend.asarray(torch.tensor([0.5, 0.8, 1.0]), complex=False)
    theta = 0.0  # normal incidence
    phi = 0.0

    k0x, k0y = compute_k0xy(backend, wavelengths, theta, phi, n_inc=1.0)

    assert torch.allclose(k0x, torch.zeros_like(k0x), atol=1e-7)
    assert torch.allclose(k0y, torch.zeros_like(k0y), atol=1e-7)
    
def test_k0xy_oblique_phi_zero():
    backend = TorchBackend(device="cpu", dtype=torch.float32, use_compile=False)

    wavelengths = backend.asarray(torch.tensor([1.0]), complex=False)  # λ = 1
    theta = torch.tensor(0.5)  # radians
    phi = 0.0

    k0x, k0y = compute_k0xy(backend, wavelengths, theta, phi, n_inc=1.0)

    k0 = 2.0 * torch.pi / wavelengths
    expected_k0x = k0 * torch.sin(theta)
    expected_k0y = torch.zeros_like(expected_k0x)

    assert torch.allclose(k0x, expected_k0x, atol=1e-6)
    assert torch.allclose(k0y, expected_k0y, atol=1e-6)
    
def test_k0xy_reduced_independent_of_lambda():
    backend = TorchBackend(device="cpu", dtype=torch.float32, use_compile=False)

    wavelengths = backend.asarray(torch.tensor([0.5, 0.8, 1.0]), complex=False)
    theta = torch.tensor(0.7)
    phi = torch.tensor(1.1)
    n_inc = backend.asarray(torch.ones_like(wavelengths) * 1.5, complex=False)

    k0x, k0y = compute_k0xy(backend, wavelengths, theta, phi, n_inc=n_inc, reduced=True)

    sin_th = torch.sin(theta)
    cos_ph = torch.cos(phi)
    sin_ph = torch.sin(phi)

    expected_k0x = n_inc * sin_th * cos_ph
    expected_k0y = n_inc * sin_th * sin_ph

    # Broadcasting: expected is scalar, k0x has shape (n_lambda,)
    assert torch.allclose(k0x, expected_k0x.expand_as(k0x), atol=1e-6)
    assert torch.allclose(k0y, expected_k0y.expand_as(k0y), atol=1e-6)
    
def test_k0xy_broadcast_theta_phi():
    backend = TorchBackend(device="cpu", dtype=torch.float32, use_compile=False)

    # 3 wavelengths, 1 theta, 1 phi
    wavelengths = backend.asarray(torch.tensor([0.5, 0.6, 0.7]), complex=False)
    theta = backend.asarray(torch.tensor(0.3), complex=False)
    phi = backend.asarray(torch.tensor(0.9), complex=False)
    n_inc = backend.asarray(torch.tensor(2.5), complex=False)

    k0x, k0y = compute_k0xy(backend, wavelengths, theta, phi, n_inc=n_inc)

    # Manually compute for each wavelength
    k0 = 2.0 * torch.pi / wavelengths
    sin_th = torch.sin(theta)
    cos_ph = torch.cos(phi)
    sin_ph = torch.sin(phi)

    expected_k0x = k0 * n_inc * sin_th * cos_ph
    expected_k0y = k0 * n_inc * sin_th * sin_ph

    assert k0x.shape == wavelengths.shape
    assert k0y.shape == wavelengths.shape
    assert torch.allclose(k0x, expected_k0x, atol=1e-6)
    assert torch.allclose(k0y, expected_k0y, atol=1e-6)
    
def test_k0xy_device_and_dtype():
    backend = TorchBackend(device="cpu", dtype=torch.float32, use_compile=False)

    wavelengths = [0.5, 0.7]
    theta = 0.2
    phi = 1.0
    n_inc = 1.0

    k0x, k0y = compute_k0xy(backend, wavelengths, theta, phi, n_inc=n_inc)

    assert k0x.device.type == backend.device.type
    assert k0y.device.type == backend.device.type
    assert k0x.dtype == backend.dtype
    assert k0y.dtype == backend.dtype
    
def test_k0xy_full():
    test_k0xy_normal_incidence()
    test_k0xy_oblique_phi_zero()
    test_k0xy_reduced_independent_of_lambda()
    test_k0xy_broadcast_theta_phi()
    test_k0xy_device_and_dtype()

    print("✓ full kvector test passed.")
    
''' Tests for Kx, Ky computation '''
def make_backend():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32
    return TorchBackend(device=device, dtype=dtype, use_compile=False)


def test_Kxy_shape_single_wavelength():
    backend = make_backend()

    # Single wavelength, kx0=ky0=0 (normal incidence in air)
    kx0 = torch.tensor([0.0])
    ky0 = torch.tensor([0.0])

    Lx = 0.5
    Ly = 0.5
    M = 2
    N = 2

    Kx, Ky = compute_Kxy(backend, kx0, ky0, Lx, Ly, M, N)

    assert Kx.shape == (1, 2 * M + 1, 2 * N + 1)
    assert Ky.shape == (1, 2 * M + 1, 2 * N + 1)


def test_Kxy_values_normal_incidence_unit_period():
    """
    For kx0=ky0=0, Lx=Ly=1, M=N=1, we expect:

    m ∈ {-1,0,1}, n ∈ {-1,0,1}

    Kx[0, :, :] = [ -2π, 0, 2π ] along axis-1, constant along axis-2
    Ky[0, :, :] = [ -2π, 0, 2π ] along axis-2, constant along axis-1
    """
    backend = make_backend()

    kx0 = torch.tensor([0.0])
    ky0 = torch.tensor([0.0])

    Lx = 1.0
    Ly = 1.0
    M = 1
    N = 1

    Kx, Ky = compute_Kxy(backend, kx0, ky0, Lx, Ly, M, N)

    # Extract batch 0
    Kx0 = Kx[0]  # shape (3, 3)
    Ky0 = Ky[0]  # shape (3, 3)

    m_vals = torch.tensor([-1.0, 0.0, 1.0], dtype=Kx0.dtype, device=Kx0.device)
    n_vals = torch.tensor([-1.0, 0.0, 1.0], dtype=Ky0.dtype, device=Ky0.device)

    expected_Kx_line = 2.0 * torch.pi * m_vals  # shape (3,)
    expected_Ky_line = 2.0 * torch.pi * n_vals  # shape (3,)

    # Kx should be constant along n, varying along m
    assert torch.allclose(Kx0[:, 0], expected_Kx_line, atol=1e-6)
    assert torch.allclose(Kx0[:, 1], expected_Kx_line, atol=1e-6)
    assert torch.allclose(Kx0[:, 2], expected_Kx_line, atol=1e-6)

    # Ky should be constant along m, varying along n
    assert torch.allclose(Ky0[0, :], expected_Ky_line, atol=1e-6)
    assert torch.allclose(Ky0[1, :], expected_Ky_line, atol=1e-6)
    assert torch.allclose(Ky0[2, :], expected_Ky_line, atol=1e-6)


def test_Kxy_batch_two_wavelengths():
    """
    Two different kx0 values should shift the entire Kx grid for each batch.
    """
    backend = make_backend()

    # Two wavelengths → two different base kx0; ky0 stays zero
    kx0 = torch.tensor([0.1, 0.3])
    ky0 = torch.tensor([0.0, 0.0])

    Lx = 1.0
    Ly = 1.0
    M = 1
    N = 1

    Kx, Ky = compute_Kxy(backend, kx0, ky0, Lx, Ly, M, N)

    assert Kx.shape == (2, 3, 3)
    assert Ky.shape == (2, 3, 3)

    # For each batch b, the whole grid should be shifted by kx0[b]
    # relative to the symmetric base [-2π,0,2π].
    m_vals = torch.tensor([-1.0, 0.0, 1.0], dtype=Kx.dtype, device=Kx.device)
    base_line = 2.0 * torch.pi * m_vals  # (3,)

    for b in range(2):
        Kx_b = Kx[b]  # (3,3)
        expected_line = base_line + kx0[b]

        # check first column; all columns must match this along axis-2
        assert torch.allclose(Kx_b[:, 0], expected_line, atol=1e-6)
        assert torch.allclose(Kx_b[:, 1], expected_line, atol=1e-6)
        assert torch.allclose(Kx_b[:, 2], expected_line, atol=1e-6)


def test_Kxy_spacing_matches_2pi_over_L():
    """
    Check that difference between adjacent harmonics along m and n
    equals 2π/Lx and 2π/Ly respectively.
    """
    backend = make_backend()

    kx0 = torch.tensor([0.0])
    ky0 = torch.tensor([0.0])

    Lx = 0.5
    Ly = 0.25
    M = 3
    N = 4

    Kx, Ky = compute_Kxy(backend, kx0, ky0, Lx, Ly, M, N)

    Kx0 = Kx[0]  # (2M+1, 2N+1)
    Ky0 = Ky[0]  # (2M+1, 2N+1)

    # Difference along m (axis 0) should be constant 2π/Lx
    delta_Kx = Kx0[1:, 0] - Kx0[:-1, 0]  # shape (2M,)
    expected_delta_Kx = torch.full_like(delta_Kx, 2.0 * torch.pi / Lx)
    assert torch.allclose(delta_Kx, expected_delta_Kx, atol=1e-6)

    # Difference along n (axis 1) should be constant 2π/Ly
    delta_Ky = Ky0[0, 1:] - Ky0[0, :-1]  # shape (2N,)
    expected_delta_Ky = torch.full_like(delta_Ky, 2.0 * torch.pi / Ly)
    assert torch.allclose(delta_Ky, expected_delta_Ky, atol=1e-6)
    
def test_Kxy_full():
    test_Kxy_shape_single_wavelength()
    test_Kxy_values_normal_incidence_unit_period()
    test_Kxy_batch_two_wavelengths()
    test_Kxy_spacing_matches_2pi_over_L()

    print("✓ full Kxy test passed.")