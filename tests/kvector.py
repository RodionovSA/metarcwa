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

    # 3 wavelengths, scalar theta, scalar phi, n_inc per-wavelength
    wavelengths = backend.asarray(torch.tensor([0.5, 0.8, 1.0]), complex=False)  # [Nw]
    theta       = backend.asarray(torch.tensor(0.7), complex=False)              # scalar
    phi         = backend.asarray(torch.tensor(1.1), complex=False)              # scalar
    n_inc       = backend.asarray(torch.ones_like(wavelengths) * 1.5,
                                  complex=False)                                  # [Nw]

    k0x, k0y = compute_k0xy(
        backend,
        wavelengths=wavelengths,
        theta=theta,
        phi=phi,
        n_inc=n_inc,
        reduced=True,
    )

    # With new broadcasting rules and scalar theta/phi:
    # shapes must be [Nw, 1, 1]
    Nw = wavelengths.shape[0]
    assert k0x.shape == (Nw, 1, 1)
    assert k0y.shape == (Nw, 1, 1)

    # Manual expected value: reduced → no 2π/λ, only direction cosines * n_inc
    sin_th = torch.sin(theta)      # scalar
    cos_ph = torch.cos(phi)        # scalar
    sin_ph = torch.sin(phi)        # scalar

    # n_inc is [Nw], so these are [Nw]
    expected_k0x = n_inc * sin_th * cos_ph   # [Nw]
    expected_k0y = n_inc * sin_th * sin_ph   # [Nw]

    # Match the new [Nw,1,1] output shape
    expected_k0x = expected_k0x.view(Nw, 1, 1)
    expected_k0y = expected_k0y.view(Nw, 1, 1)

    assert torch.allclose(k0x, expected_k0x, atol=1e-6)
    assert torch.allclose(k0y, expected_k0y, atol=1e-6)
    
def test_k0xy_broadcast_theta_phi():
    backend = TorchBackend(device="cpu", dtype=torch.float32, use_compile=False)

    # 3 wavelengths, scalar theta, scalar phi
    wavelengths = backend.asarray(torch.tensor([0.5, 0.6, 0.7]), complex=False)  # [Nw]
    theta       = backend.asarray(torch.tensor(0.3), complex=False)               # scalar
    phi         = backend.asarray(torch.tensor(0.9), complex=False)               # scalar
    n_inc       = backend.asarray(torch.tensor(2.5), complex=False)               # scalar

    k0x, k0y = compute_k0xy(backend, wavelengths, theta, phi, n_inc=n_inc)

    # Shapes should be [Nw, 1, 1]
    assert k0x.shape == (wavelengths.shape[0], 1, 1)
    assert k0y.shape == (wavelengths.shape[0], 1, 1)

    # Manually compute expected values per wavelength (1D), then reshape to [Nw,1,1]
    k0     = 2.0 * torch.pi / wavelengths        # [Nw]
    sin_th = torch.sin(theta)                    # scalar
    cos_ph = torch.cos(phi)                      # scalar
    sin_ph = torch.sin(phi)                      # scalar

    expected_k0x = k0 * n_inc * sin_th * cos_ph  # [Nw]
    expected_k0y = k0 * n_inc * sin_th * sin_ph  # [Nw]

    expected_k0x = expected_k0x.view(-1, 1, 1)   # [Nw,1,1]
    expected_k0y = expected_k0y.view(-1, 1, 1)   # [Nw,1,1]

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

    # Single wavelength, kx0=ky0=0 (normal incidence)
    kx0 = backend.reshape(backend.asarray(torch.tensor([0.0]), complex=False), (1, 1, 1))  # shape [1, 1, 1]
    ky0 = backend.reshape(backend.asarray(torch.tensor([0.0]), complex=False), (1, 1, 1))  # shape [1, 1, 1]

    Lx = 0.5
    Ly = 0.5
    M = 2
    N = 2

    Kx, Ky = compute_Kxy(backend, kx0, ky0, Lx, Ly, M, N)

    # Expected output: (*batch_shape, 2M+1, 2N+1)
    # batch_shape = (1,) → (1, 1, 1, 5, 5)
    expected_shape = (1, 1, 1, 2 * M + 1, 2 * N + 1)

    assert Kx.shape == expected_shape, f"Kx.shape={Kx.shape}, expected={expected_shape}"
    assert Ky.shape == expected_shape, f"Ky.shape={Ky.shape}, expected={expected_shape}"


def test_Kxy_values_normal_incidence_unit_period():
    """
    For kx0=ky0=0, Lx=Ly=1, M=N=1, we expect:

    m ∈ {-1,0,1}, n ∈ {-1,0,1}

    Kx[..., m, n]: varies along m, constant along n
    Ky[..., m, n]: varies along n, constant along m
    """
    backend = make_backend()

    # Single wavelength, single theta, single phi → (Nw,Nt,Np) = (1,1,1)
    kx0 = backend.reshape(
        backend.asarray(torch.tensor([0.0]), complex=False),
        (1, 1, 1),
    )
    ky0 = backend.reshape(
        backend.asarray(torch.tensor([0.0]), complex=False),
        (1, 1, 1),
    )

    Lx = 1.0
    Ly = 1.0
    M = 1
    N = 1

    Kx, Ky = compute_Kxy(backend, kx0, ky0, Lx, Ly, M, N)

    # Shapes should be (Nw,Nt,Np, 2M+1, 2N+1) = (1,1,1,3,3)
    assert Kx.shape == (1, 1, 1, 2 * M + 1, 2 * N + 1)
    assert Ky.shape == (1, 1, 1, 2 * M + 1, 2 * N + 1)

    # Extract the only (wavelength, theta, phi) combination → (3,3)
    Kx0 = Kx[0, 0, 0]  # (3,3)
    Ky0 = Ky[0, 0, 0]  # (3,3)

    m_vals = torch.tensor([-1.0, 0.0, 1.0], dtype=Kx0.dtype, device=Kx0.device)
    n_vals = torch.tensor([-1.0, 0.0, 1.0], dtype=Ky0.dtype, device=Ky0.device)

    expected_Kx_line = 2.0 * torch.pi * m_vals  # (3,)
    expected_Ky_line = 2.0 * torch.pi * n_vals  # (3,)

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
    Two different kx0 values should shift the entire Kx grid for each wavelength.
    """
    backend = make_backend()

    # Two wavelengths → shape (Nw,Nt,Np) = (2,1,1)
    kx0 = backend.reshape(
        backend.asarray(torch.tensor([0.1, 0.3]), complex=False),
        (2, 1, 1),
    )
    ky0 = backend.reshape(
        backend.asarray(torch.tensor([0.0, 0.0]), complex=False),
        (2, 1, 1),
    )

    Lx = 1.0
    Ly = 1.0
    M = 1
    N = 1

    Kx, Ky = compute_Kxy(backend, kx0, ky0, Lx, Ly, M, N)

    # (Nw, Nt, Np, 2M+1, 2N+1) = (2,1,1,3,3)
    assert Kx.shape == (2, 1, 1, 2 * M + 1, 2 * N + 1)
    assert Ky.shape == (2, 1, 1, 2 * M + 1, 2 * N + 1)

    # For each wavelength b, the whole grid should be shifted by kx0[b,0,0]
    # relative to the symmetric base [-2π, 0, 2π].
    m_vals = torch.tensor([-1.0, 0.0, 1.0], dtype=Kx.dtype, device=Kx.device)
    base_line = 2.0 * torch.pi * m_vals  # (3,)

    for b in range(2):
        # Extract the (Nt=0, Np=0) slice → shape (3,3)
        Kx_b = Kx[b, 0, 0]  # (3,3)
        kx0_b = kx0[b, 0, 0]

        expected_line = base_line + kx0_b  # (3,)

        # Kx should be constant along n, shifted along m
        assert torch.allclose(Kx_b[:, 0], expected_line, atol=1e-6)
        assert torch.allclose(Kx_b[:, 1], expected_line, atol=1e-6)
        assert torch.allclose(Kx_b[:, 2], expected_line, atol=1e-6)


def test_Kxy_spacing_matches_2pi_over_L():
    """
    Check that difference between adjacent harmonics along m and n
    equals 2π/Lx and 2π/Ly respectively.
    """
    backend = make_backend()

    # Single (wavelength, theta, phi) → (Nw,Nt,Np) = (1,1,1)
    kx0 = backend.reshape(
        backend.asarray(torch.tensor([0.0]), complex=False),
        (1, 1, 1),
    )
    ky0 = backend.reshape(
        backend.asarray(torch.tensor([0.0]), complex=False),
        (1, 1, 1),
    )

    Lx = 0.5
    Ly = 0.25
    M = 3
    N = 4

    Kx, Ky = compute_Kxy(backend, kx0, ky0, Lx, Ly, M, N)

    # Shape must be (1,1,1, 2M+1, 2N+1)
    assert Kx.shape == (1, 1, 1, 2 * M + 1, 2 * N + 1)
    assert Ky.shape == (1, 1, 1, 2 * M + 1, 2 * N + 1)

    # Extract the only (wavelength, theta, phi) slice → (2M+1, 2N+1)
    Kx0 = Kx[0, 0, 0]  # (2M+1, 2N+1)
    Ky0 = Ky[0, 0, 0]  # (2M+1, 2N+1)

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