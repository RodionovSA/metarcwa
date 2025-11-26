import torch
from src.backend import TorchBackend

'''
Pytorch backend tests
'''
def test_torch_backend_compile():
    # Create backend with compile enabled
    backend = TorchBackend(device="cuda", dtype=torch.float32, use_compile=True)

    # Prepare input
    x = backend.asarray(torch.randn(32, 32), complex=False)

    # A small backend-agnostic function
    def fn(t):
        # Validate input
        backend.validate(t)
        # Apply 2D FFT and inverse FFT
        y = backend.fft2(t)
        z = backend.ifft2(y)
        return z.real  # real is traceable

    # IMPORTANT: compile a *fresh* function that uses only tensors
    fn_compiled = backend.jit(fn)

    # Run both versions
    out_ref = fn(x)
    out_opt = fn_compiled(x)

    # Check correctness
    assert torch.allclose(out_ref, out_opt, atol=1e-5), "Compiled output mismatch"

    print("✓ backend.compile test passed.")
    print("Output dtype:", out_opt.dtype)
    print("Device:", out_opt.device)
    
def test_torch_backend_dtype():
    backend = TorchBackend(device="cuda", dtype=torch.float32, use_compile=False)

    x = backend.asarray([1.0, 2.0, 3.0], complex=False)
    xc = backend.asarray([1.0, 2.0, 3.0], complex=True)

    assert x.dtype == backend.dtype, f"Expected real dtype {backend.dtype}, got {x.dtype}"
    assert xc.dtype.is_complex, f"Expected complex dtype, got {xc.dtype}"

    print("✓ dtype test passed.")
    
def test_torch_backend_device():
    backend = TorchBackend(device="cuda", dtype=torch.float32, use_compile=False)

    x = backend.asarray(torch.randn(4), complex=False)
    xc = backend.asarray(torch.randn(4), complex=True)

    assert x.device.type == backend.device.type
    assert xc.device.type == backend.device.type

    print("✓ device test passed.")
    
def test_torch_backend_validate():
    backend = TorchBackend(device="cuda", dtype=torch.float32, use_compile=False)

    # Valid tensor
    x = backend.asarray([1.0, 2.0, 3.0])
    backend.validate(x)  # must not raise

    # Wrong type
    try:
        backend.validate(123)
        raise AssertionError("validate should fail on non-tensor input")
    except TypeError:
        pass

    # Wrong device
    x_cpu = torch.tensor([1.0, 2.0, 3.0], device="cpu")
    try:
        backend.validate(x_cpu)
        raise AssertionError("validate should fail on wrong device")
    except ValueError:
        pass

    print("✓ validate test passed.")
    
def test_torch_backend_fft_ifft():
    backend = TorchBackend(device="cuda", dtype=torch.float32, use_compile=False)

    x = backend.asarray(torch.randn(16, 16), complex=False)

    y = backend.fft2(x)
    z = backend.ifft2(y)

    # Should reconstruct original (within numerical accuracy)
    assert torch.allclose(z.real, x, atol=1e-5), "ifft2(fft2(x)) does not match x (real part)"

    print("✓ fft/ifft test passed.")
    
def test_torch_backend_matmul():
    backend = TorchBackend(device="cuda", dtype=torch.float32, use_compile=False)

    A = backend.asarray(torch.randn(4, 4), complex=False)
    B = backend.asarray(torch.randn(4, 4), complex=False)

    C = backend.matmul(A, B)

    assert C.shape == (4, 4)
    assert C.device.type == backend.device.type
    assert C.dtype == backend.dtype

    print("✓ matmul test passed.")
    
def test_torch_backend_conj():
    backend = TorchBackend(device="cuda", dtype=torch.float32, use_compile=False)

    x = backend.asarray(torch.randn(4), complex=True)
    xc = backend.conj(x)

    assert torch.allclose(xc, torch.conj(x))
    assert xc.device.type == backend.device.type
    assert xc.dtype.is_complex

    print("✓ conj test passed.")
    
def test_torch_backend_full():
    test_torch_backend_dtype()
    test_torch_backend_device()
    test_torch_backend_validate()
    test_torch_backend_fft_ifft()
    test_torch_backend_matmul()
    test_torch_backend_conj()
    test_torch_backend_compile()

    print("✓ full backend test passed.")
    


    

    
