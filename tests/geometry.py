import torch
from src.backend import TorchBackend
from src.model.geometry.geometry import Rectangle, Lattice

''' Test Rectangle geometry object:
    - real-space epsilon(x,y) and mu(x,y)
    - Fourier coefficients epsilon_{m,n} and mu_{m,n}
    - compare FFT vs analytic Fourier coefficients
    - test DC term correctness
    - test batch materials support
'''
def test_rectangle_dc_term():
    device = "cpu"
    dtype = torch.float64
    backend = TorchBackend(device=device, dtype=dtype)

    # Geometry / period
    Lx, Ly = 1.0, 1.0
    Nx, Ny = 256, 256
    period = (Lx, Ly)
    grid = (Nx, Ny)
    lattice = Lattice(period, grid)

    # Rectangle centered at 0 in [-L/2, L/2]
    center = (0.0, 0.0)
    size = (0.4, 0.6)  # any numbers < Lx,Ly

    eps_bg = 1.0
    eps_rect = 3.0 + 0.2j

    rect = Rectangle(
        backend=backend,
        lattice=lattice,
        center=center,
        size=size,
        epsilon=eps_rect,
        mu=1.0,
    )

    # Real-space permittivity
    eps_xy = rect.epsilon_xy(epsilonbg=eps_bg)  # (B, Nx, Ny)
    assert eps_xy.shape == (1, Nx, Ny)
    eps_xy_mean = eps_xy.mean()

    # Analytic Fourier coefficients with M=N=0 → only DC
    M = N = 0
    eps_mn = rect.epsilon_mn(M, N, epsilonbg=eps_bg)  # (B,1,1)
    assert eps_mn.shape == (1, 1, 1)

    # Compare DC mode:
    dc_numeric = eps_xy_mean
    dc_analytic = eps_mn[0, 0, 0]

    assert torch.allclose(dc_numeric, dc_analytic, atol=1e-3, rtol=1e-3), \
        f"DC mismatch: numeric={dc_numeric}, analytic={dc_analytic}"
        
def test_rectangle_fft_vs_analytic():
    device = "cpu"
    dtype = torch.float64
    backend = TorchBackend(device=device, dtype=dtype)

    # Period & grid
    Lx, Ly = 1.0, 1.0
    Nx, Ny = 512, 512      # fairly fine grid
    period = (Lx, Ly)
    grid = (Nx, Ny)
    lattice = Lattice(period, grid)

    center = (0.0, 0.0)    # in [-L/2, L/2]
    size = (0.3, 0.5)

    eps_bg = 1.0
    eps_rect = 3.0 + 0.2j

    rect = Rectangle(
        backend=backend,
        lattice=lattice,
        center=center,
        size=size,
        epsilon=eps_rect,
        mu=1.0,
    )

    # Real-space eps(x,y)
    eps_xy = rect.epsilon_xy(epsilonbg=eps_bg)  # (1, Nx, Ny)
    assert eps_xy.shape == (1, Nx, Ny)

    # Analytic Fourier coefficients for several harmonics
    M = N = 3  # compare low-order harmonics
    eps_mn_analytic = rect.epsilon_mn(M, N, epsilonbg=eps_bg)  # (1, 2M+1, 2N+1)
    assert eps_mn_analytic.shape == (1, 2*M + 1, 2*N + 1)

    # Numeric Fourier coefficients via FFT
    # FFT norm = sum / (Nx*Ny)
    eps_fft = backend.fft2(eps_xy) / (Nx * Ny)         # (1, Nx, Ny)
    eps_fft_shift = backend.fftshift(eps_fft)        # (1, Nx, Ny)

    # Crop central (2M+1, 2N+1) block
    cx = Nx // 2
    cy = Ny // 2
    m_lo = cx - M
    m_hi = cx + M + 1
    n_lo = cy - N
    n_hi = cy + N + 1

    eps_fft_crop = eps_fft_shift[:, m_lo:m_hi, n_lo:n_hi]   # (1, 2M+1, 2N+1)
    assert eps_fft_crop.shape == eps_mn_analytic.shape

    # Compute RMSE between numeric and analytic coefficients
    diff = eps_fft_crop - eps_mn_analytic
    rmse = torch.sqrt(torch.mean(torch.abs(diff) ** 2))

    print("FFT vs analytic RMSE:", rmse.item())
    assert rmse.item() < 1e-2, \
        f"FFT vs analytic coefficients differ too much, rmse={rmse.item()}"
        
def test_rectangle_batch_materials_shapes_and_dc():
    device = "cpu"
    dtype = torch.float64
    backend = TorchBackend(device=device, dtype=dtype)

    Lx, Ly = 1.0, 1.0
    Nx, Ny = 512, 512
    period = (Lx, Ly)
    grid = (Nx, Ny)
    lattice = Lattice(period, grid)

    center = (0.0, 0.0)
    size = (0.4, 0.4)

    # Two different materials in batch
    eps_rect = torch.tensor([2.0 + 0.1j, 4.0 + 0.3j], dtype=torch.complex128)
    eps_bg   = torch.tensor([1.0 + 0.0j, 1.2 + 0.0j], dtype=torch.complex128)

    rect = Rectangle(
        backend=backend,
        lattice=lattice,
        center=center,
        size=size,
        epsilon=eps_rect,
        mu=1.0,
    )

    # Real-space epsilon(x,y) for both batches
    eps_xy = rect.epsilon_xy(epsilonbg=eps_bg)  # (B, Nx, Ny)
    B = eps_rect.shape[0]
    assert eps_xy.shape == (B, Nx, Ny)

    # Check that each batch DC term = average
    eps_xy_mean = eps_xy.mean(dim=(-2, -1))  # (B,)

    M = N = 0
    eps_mn = rect.epsilon_mn(M, N, epsilonbg=eps_bg)  # (B,1,1)
    assert eps_mn.shape == (B, 1, 1)

    dc_analytic = eps_mn[:, 0, 0]  # (B,)

    assert torch.allclose(eps_xy_mean, dc_analytic, atol=1e-3, rtol=1e-3), \
        f"Batch DC mismatch: numeric={eps_xy_mean}, analytic={dc_analytic}"
        
def test_rectangle_mu_same_pipeline():
    device = "cpu"
    dtype = torch.float64
    backend = TorchBackend(device=device, dtype=dtype)

    Lx, Ly = 1.0, 1.0
    Nx, Ny = 512, 512
    period = (Lx, Ly)
    grid = (Nx, Ny)
    lattice = Lattice(period, grid)

    center = (0.0, 0.0)
    size = (0.4, 0.4)

    mu_bg = 1.0
    mu_rect = 1.5

    rect = Rectangle(
        backend=backend,
        lattice=lattice,
        center=center,
        size=size,
        epsilon=2.0,
        mu=mu_rect,
    )

    mu_xy = rect.mu_xy(mubg=mu_bg)  # (1, Nx, Ny)
    M = N = 0
    mu_mn = rect.mu_mn(M, N, mubg=mu_bg)  # (1,1,1)

    mu_xy_mean = mu_xy.mean()
    dc_mu = mu_mn[0, 0, 0]

    assert torch.allclose(mu_xy_mean, dc_mu, atol=1e-3, rtol=1e-3)
    
def run_all_geometry_tests():
    test_rectangle_dc_term()
    test_rectangle_fft_vs_analytic()
    test_rectangle_batch_materials_shapes_and_dc()
    test_rectangle_mu_same_pipeline()
    print("All geometry tests passed.")