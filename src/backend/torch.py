#src/backend/torch.py
# Backend Interface for RCWA/FMM computations using PyTorch

from typing import Any, Callable
import torch
from src.backend.base import Backend

class TorchBackend(Backend):
    """
    PyTorch backend implementation for RCWA/FMM computations.

    This backend provides:
    - dtype and device management,
    - array creation and casting,
    - FFT operations via torch.fft,
    - batched matrix multiplication,
    - optional JIT compilation via torch.compile.

    All returned arrays are guaranteed to be torch.Tensor objects.
    """

    def __init__(
        self,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float32,
        use_compile: bool = False,
    ):
        """
        Initialize the Torch backend.

        Parameters
        ----------
        device : str or torch.device, default "cpu"
            Device on which all tensors will be allocated.
            Can be "cuda", "cpu", "mps", etc.

        dtype : torch.dtype, default torch.float32
            Default dtype for real-valued arrays (torch.float32 or torch.float64).
            In the case of complex arrays, torch.complex64 or torch.complex128. 
            
        use_compile : bool, default False
            Whether to enable torch.compile when calling jit().
        """
        self.device_ = torch.device(device)
        
        # There is no much point in supporting other dtypes in this backend
        if dtype not in (torch.float32, torch.float64):
            raise ValueError("dtype must be torch.float32 or torch.float64")
        
        self.dtype_ = dtype
        
        self.use_compile = use_compile
    
    @property
    def device(self) -> torch.device:
        """The device on which tensors are allocated."""
        return self.device_
    
    @property
    def dtype(self) -> torch.dtype:
        """The dtype used by this backend."""
        return self.dtype_
    
    @property
    def xp(self):
        """Numerical backend module."""
        return torch
    
    @property
    def name(self):
        """Backend name identifier."""
        return "torch"
    # ---------------------------------------------------------
    # Input validation / configuration
    # ---------------------------------------------------------
    def validate(self, x: Any) -> None:
        """
        Validate that input `x` is a torch.Tensor on this 
        backend's device and correct dtype.

        Parameters
        ----------
        x : Any
            Input data to validate.

        Raises
        ------
        TypeError
            If `x` is not a torch.Tensor.
        ValueError
            If `x` is not on the correct device or has incorrect dtype.
        """
        if not isinstance(x, torch.Tensor):
            raise TypeError("Input must be a torch.Tensor")
        
        if x.device.type != self.device.type:
            raise ValueError(f"Tensor must be on device type {self.device.type}, got {x.device}")
        
        self._check_dtype(x)
        
    def _check_dtype(self, x: torch.Tensor) -> None:
        """
        Validate dtype according to backend rules.
        Allows: bool, integer, backend float dtype, backend complex dtype.
        """

        # Allow boolean
        if x.dtype == torch.bool:
            return

        # Allow signed/unsigned integers
        if x.dtype.is_signed or x.dtype.is_unsigned:
            return

        # Floating-point case
        if x.dtype == self.dtype:
            return

        # Complex case
        if x.dtype == self.complex_dtype:
            return

        raise ValueError(
            f"Tensor dtype must be {self.dtype} or {self.complex_dtype}, "
            f"or any bool/int type. Got {x.dtype} instead."
        )
    
    # -------------------------------------------------------------------------
    # Array Creation / Casting
    # -------------------------------------------------------------------------
    def asarray(self, x, complex: bool = False) -> torch.Tensor:
        """
        Convert input `x` into a PyTorch tensor with correct dtype and device.

        Parameters
        ----------
        x : Any
            Input convertible to a torch.Tensor.

        complex : bool, default False
            If True, cast result to the backend's complex dtype.
            Otherwise cast to the backend's real dtype.

        Returns
        -------
        torch.Tensor
            Tensor on backend.device with correct dtype.

        Notes
        -----
        - This method never modifies input in-place.
        - If x is already a tensor:
            - dtype will be cast to the correct one,
            - device will be moved if necessary.
        """
        if complex:
            # Map real dtype -> matching complex dtype
            if self.dtype == torch.float32:
                target_dtype = torch.complex64
            else:
                target_dtype = torch.complex128
        else:
            target_dtype = self.dtype

        # Convert input to tensor first
        t = torch.as_tensor(x)

        # Move to device + cast to correct dtype
        return t.to(device=self.device, dtype=target_dtype)
    
    def zeros(self, shape: tuple[int, ...]) -> torch.Tensor:
        """
        Create a tensor of zeros with specified shape, dtype, and device.

        Parameters
        ----------
        shape : tuple of int
            Shape of the output tensor.

        Returns
        -------
        torch.Tensor
            Tensor of zeros on backend.device with correct dtype.
        """
        return torch.zeros(shape, dtype=self.dtype, device=self.device)

    def ones(self, shape: tuple[int, ...]) -> torch.Tensor:
        """
        Create a tensor of ones with specified shape, dtype, and device.

        Parameters
        ----------
        shape : tuple of int
            Shape of the output tensor.

        Returns
        -------
        torch.Tensor
            Tensor of ones on backend.device with correct dtype.
        """
        return torch.ones(shape, dtype=self.dtype, device=self.device)
    
    def eye(self, n: int) -> torch.Tensor:
        """
        Create an identity matrix of size n x n with correct dtype and device.

        Parameters
        ----------
        n : int
            Size of the identity matrix.

        Returns
        -------
        torch.Tensor
            Identity matrix of shape (n, n) on backend.device with correct dtype.
        """
        return torch.eye(n, dtype=self.dtype, device=self.device)
    
    def zeros_like(self, x: torch.Tensor) -> torch.Tensor:
        """
        Create a tensor of zeros with the same shape as `x`.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor whose shape will be used.
        complex : bool, default False
            If True, use complex dtype. Otherwise use real dtype.

        Returns
        -------
        torch.Tensor
            Tensor of zeros with same shape as `x`, correct dtype and device.
        """
        return torch.zeros_like(x, dtype=self.dtype, device=self.device)
    
    def ones_like(self, x: torch.Tensor) -> torch.Tensor:
        """
        Create a tensor of ones with the same shape as `x`.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor whose shape will be used.

        Returns
        -------
        torch.Tensor
            Tensor of ones with same shape as `x`, correct dtype and device.
        """
        return torch.ones_like(x, dtype=self.dtype, device=self.device)
    
    def arange(self, start: int, stop: int, step: int = 1) -> torch.Tensor:
        """
        Create a 1D tensor of evenly spaced values within [start, stop) with given step.
        """
        arr = torch.arange(start, stop, step, dtype=self.dtype, device=self.device)
        return arr
    
    def linspace(self, start: float, stop: float, num: int) -> torch.Tensor:
        """
        Create a 1D tensor of `num` evenly spaced values between `start` and `stop`.
        """
        arr = torch.linspace(start, stop, num, dtype=self.dtype, device=self.device)
        return arr    
    
    def clone(self, x: torch.Tensor) -> torch.Tensor:
        """
        Create a copy of the input tensor `x`.
        """
        self.validate(x)
        return x.clone()
    
    # ---------------------------------------------------------
    # Array operations
    # ---------------------------------------------------------
    def resample(self, x: torch.Tensor, shape: tuple[int, int]) -> torch.Tensor:
        """
        Resample a 2D field.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (W, H).
        shape : tuple[int, int]
            Target spatial shape (W_new, H_new).

        Returns
        -------
        torch.Tensor
            Resampled tensor of shape (W_new, H_new).
        """
        self.validate(x)

        if x.ndim != 2:
            raise ValueError("x must have shape (W, H)")

        # add batch and channel dimension → (1, 1, W, H)
        x_ = x.unsqueeze(0).unsqueeze(0)

        y = torch.nn.functional.interpolate(
            x_,
            size=shape,
            mode="bilinear",
            align_corners=False
        )

        # remove channel dimension → (W_new, H_new)
        return y.squeeze(0).squeeze(0)
        
    def clamp(self, x: torch.Tensor, min_value: float, max_value: float) -> torch.Tensor:
        """
        Clamp all elements in `x` to be within the range [min_value, max_value].
        """
        self.validate(x)
        return torch.clamp(x, min=min_value, max=max_value)
    
    def cat(self, arrays: list[torch.Tensor], dim: int) -> torch.Tensor:
        """
        Concatenate a list of tensors along the specified dimension.
        """
        for arr in arrays:
            self.validate(arr)
        return torch.cat(arrays, dim=dim)
    
    def stack(self, arrays: list[torch.Tensor], dim: int) -> torch.Tensor:
        """
        Stack a sequence of tensors along a new dimension.
        """
        for arr in arrays:
            self.validate(arr)
        return torch.stack(arrays, dim=dim)
    
    def roll(self, x: torch.Tensor, shifts: int, dims: int) -> torch.Tensor:
        """
        Roll the elements of `x` along the specified dimension by `shifts`.
        """
        self.validate(x)
        return torch.roll(x, shifts=shifts, dims=dims)
    
    def amin(self, x: torch.Tensor, dim: int | tuple[int, ...] = None, keepdim: bool = False) -> torch.Tensor:
        """
        Compute the minimum of tensor elements over given axes.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor to compute the minimum over.
        dim : int or tuple of int, optional
            Axis or axes along which to compute the minimum. By default, compute over all axes.
        keepdim : bool, default=False
            If True, retains reduced dimensions with size 1.

        Returns
        -------
        torch.Tensor
            Minimum of the tensor elements over the specified axes.

        Requirements
        ------------
        - Must preserve device and dtype.
        """
        self.validate(x)
        return torch.amin(x, dim=dim, keepdim=keepdim)
    
    def amax(self, x: torch.Tensor, dim: int | tuple[int, ...] = None, keepdim: bool = False) -> torch.Tensor:
        """
        Compute the maximum of tensor elements over given axes.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor to compute the maximum over.
        dim : int or tuple of int, optional
            Axis or axes along which to compute the maximum. By default, compute over all axes.
        keepdim : bool, default=False
            If True, retains reduced dimensions with size 1.

        Returns
        -------
        torch.Tensor
            Maximum of the tensor elements over the specified axes.

        Requirements
        ------------
        - Must preserve device and dtype.
        """
        self.validate(x)
        return torch.amax(x, dim=dim, keepdim=keepdim)
    
    def nonzero(self, x: torch.Tensor) -> torch.Tensor:
        """
        Return the indices of the non-zero elements of `x`.
        """
        self.validate(x)
        return torch.nonzero(x, as_tuple=False)
    
    def expand(self, x: torch.Tensor, shape: tuple[int, ...]) -> torch.Tensor:
        """
        Expand tensor `x` to the specified shape using broadcasting rules.
        """
        self.validate(x)
        return x.expand(shape)
    
    def take_along_axis(self, arr: torch.Tensor, indices: torch.Tensor, dim: int) -> torch.Tensor:
        """
        Take values from `arr` along the specified axis using `indices`.
        """
        self.validate(arr)
        return torch.take_along_dim(arr, indices, dim=dim)
    
    def unsqueeze(self, x: torch.Tensor, dim: int) -> torch.Tensor:
        """
        Add a singleton dimension to `x` at the specified dim.
        """
        self.validate(x)
        return torch.unsqueeze(x, dim=dim)
    
    def squeeze(self, x: torch.Tensor, dim: int) -> torch.Tensor:
        """
        Remove a singleton dimension from `x` at the specified axis.
        """
        self.validate(x)
        return torch.squeeze(x, dim=dim)
    # -------------------------------------------------------------------------
    # FFT Operations
    # -------------------------------------------------------------------------
    def fft2(self, x: torch.Tensor, dim: tuple[int, int] = (-2, -1)) -> torch.Tensor:
        """
        Compute a 2D FFT using torch.fft.

        Parameters
        ----------
        x : torch.Tensor
            Input array (..., Nx, Ny). Must be on this backend's device.
        dim : tuple of int
            Dimensions over which to compute the FFT.

        Returns
        -------
        torch.Tensor
            2D FFT of x, same shape, complex dtype.

        Requirements
        ------------
        - No in-place modifications.
        - Preserves device.
        - Input may be real or complex; output is always complex.
        """
        self.validate(x)
        return torch.fft.fft2(x, dim=dim)

    def ifft2(self, x: torch.Tensor, dim: tuple[int, int] = (-2, -1)) -> torch.Tensor:
        """
        Compute a 2D inverse FFT using torch.fft.

        Parameters
        ----------
        x : torch.Tensor
            Input array in Fourier domain.
        dim : tuple of int
            Dimensions over which to compute the inverse FFT.

        Returns
        -------
        torch.Tensor
            Inverse FFT of x, same shape, complex dtype.

        Notes
        -----
        - Uses PyTorch's default normalization convention ("backward").
        """
        self.validate(x)
        return torch.fft.ifft2(x, dim=dim)

    def fftshift(self, x: torch.Tensor, dim: tuple[int, int] = (-2, -1)) -> torch.Tensor:    
        """
        Shift the zero-frequency component to the center of the spectrum.
        """
        self.validate(x)
        return torch.fft.fftshift(x, dim=dim)
    
    def ifftshift(self, x: torch.Tensor, dim: tuple[int, int] = (-2, -1)) -> torch.Tensor:
        """
        Inverse of fftshift: shift the center back to the original position.
        """
        self.validate(x)
        return torch.fft.ifftshift(x, dim=dim)
    # -------------------------------------------------------------------------
    # Linear Algebra
    # -------------------------------------------------------------------------
    def matmul(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """
        Perform matrix multiplication using PyTorch's @ operator.

        Parameters
        ----------
        a : torch.Tensor
            Left matrix (..., M, K).

        b : torch.Tensor
            Right matrix (..., K, N).

        Returns
        -------
        torch.Tensor
            Result (..., M, N).

        Requirements
        ------------
        - Supports batching through broadcasting.
        - Must not modify inputs.
        - Must run on the backend's device.
        """
        self.validate(a)
        self.validate(b)
        return a @ b

    def diag_embed(self, x: torch.Tensor, dim1: int = -2, dim2: int = -1) -> torch.Tensor:
        """
        Create a batch of diagonal matrices from the last dimension of `x`.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (..., N).
        dim1 : int, default=-2
            The first dimension of the output diagonal matrices.
        dim2 : int, default=-1
            The second dimension of the output diagonal matrices.

        Returns
        -------
        torch.Tensor
            Output tensor of shape (..., N, N) where each slice along the leading
            dimensions is a diagonal matrix with the corresponding elements from `x`.
        """
        self.validate(x)
        return torch.diag_embed(x, dim1=dim1, dim2=dim2)
    
    def solve(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """
        Solve the linear system A * X = B for X using torch.linalg.solve.

        Parameters
        ----------
        A : torch.Tensor
            Coefficient matrix of shape (..., M, M).
        B : torch.Tensor
            Ordinate or right-hand side matrix of shape (..., M, K).

        Returns
        -------
        torch.Tensor
            Solution matrix X of shape (..., M, K).
        """
        self.validate(A)
        self.validate(B)
        return torch.linalg.solve(A, B)
    
    def inv(self, A: torch.Tensor) -> torch.Tensor:
        """
        Compute the inverse of square matrix A using torch.linalg.inv.

        Parameters
        ----------
        A : torch.Tensor
            Input square matrix of shape (..., M, M).

        Returns
        -------
        torch.Tensor
            Inverse of A, shape (..., M, M).
        """
        self.validate(A)
        return torch.linalg.inv(A)
    
    def pinv(self, A: torch.Tensor) -> torch.Tensor:
        """
        Compute the Moore-Penrose pseudo-inverse of matrix A using torch.linalg.pinv.

        Parameters
        ----------
        A : torch.Tensor
            Input matrix of shape (..., M, N).

        Returns
        -------
        torch.Tensor
            Pseudo-inverse of A, shape (..., N, M).
        """
        self.validate(A)
        return torch.linalg.pinv(A)
    
    def eig(self, A: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the eigenvalues and right eigenvectors of square matrix A using torch.linalg.eig.

        Parameters
        ----------
        A : torch.Tensor
            Input square matrix of shape (..., M, M).

        Returns
        -------
        tuple of torch.Tensor
            - eigenvalues: tensor of shape (..., M)
            - eigenvectors: tensor of shape (..., M, M)
        """
        self.validate(A)
        return torch.linalg.eig(A)
    
    def eigh(self, A: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the eigenvalues and eigenvectors of a Hermitian matrix A using torch.linalg.eigh.

        Parameters
        ----------
        A : torch.Tensor
            Input Hermitian matrix of shape (..., M, M).

        Returns
        -------
        tuple of torch.Tensor
            - eigenvalues: tensor of shape (..., M)
            - eigenvectors: tensor of shape (..., M, M)
        """
        self.validate(A)
        return torch.linalg.eigh(A)
    
    def svd(self, A: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute the singular value decomposition (SVD) of matrix A using torch.linalg.svd.

        Parameters
        ----------
        A : torch.Tensor
            Input matrix of shape (..., M, N).

        Returns
        -------
        tuple of torch.Tensor
            - U: left singular vectors, shape (..., M, M)
            - S: singular values, shape (..., min(M, N))
            - Vh: right singular vectors (conjugate transpose), shape (..., N, N)
        """
        self.validate(A)
        return torch.linalg.svd(A, full_matrices=True)
    
    def qr(self, A: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the QR decomposition of matrix A using torch.linalg.qr.

        Parameters
        ----------
        A : torch.Tensor
            Input matrix of shape (..., M, N).

        Returns
        -------
        tuple of torch.Tensor
            - Q: orthogonal matrix, shape (..., M, M)
            - R: upper triangular matrix, shape (..., M, N)
        """
        self.validate(A)
        return torch.linalg.qr(A, mode='complete')
    # -------------------------------------------------------------------------
    # Complex Utilities
    # -------------------------------------------------------------------------
    def conj(self, x: torch.Tensor) -> torch.Tensor:
        """
        Return the complex conjugate of a tensor.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.

        Returns
        -------
        torch.Tensor
            Conjugated tensor. Real tensors are returned unchanged.

        Notes
        -----
        - Uses torch.conj which is differentiable.
        """
        self.validate(x)
        return torch.conj(x)
    
    # ---------------------------------------------------------
    # Math function utilities
    # ---------------------------------------------------------
    def mod(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor: 
        """
        Compute the element-wise modulus of `x` by `y`.

        Parameters
        ----------
        x : torch.Tensor
            Dividend tensor.
        y : torch.Tensor
            Divisor tensor.

        Returns
        -------
        torch.Tensor
            Element-wise modulus of x by y.

        Notes
        -----
        - Uses torch.remainder which is differentiable.
        """
        self.validate(x)
        self.validate(y)
        return torch.remainder(x, y)
    
    def sigmoid(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute the element-wise sigmoid function of `x`.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.

        Returns
        -------
        torch.Tensor
            Element-wise sigmoid of x.

        Notes
        -----
        - Uses torch.sigmoid which is differentiable.
        """
        self.validate(x)
        return torch.sigmoid(x)
    
    def sum(self, x: torch.Tensor, dim: int | tuple[int, ...] = None, keepdim: bool = False) -> torch.Tensor:
        """
        Compute the sum of tensor elements over given axes.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor to sum over.
        dim : int or tuple of int, optional
            Axis or axes along which to sum. By default, sum over all axes.
        keepdim : bool, default=False
            If True, retains reduced dimensions with size 1.

        Returns
        -------
        torch.Tensor
            Sum of the tensor elements over the specified axes.

        Notes
        -----
        - Uses torch.sum which is differentiable.
        """
        self.validate(x)
        return torch.sum(x, dim=dim, keepdim=keepdim)
    
    def besselj1(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute the Bessel function of the first kind of order one, J1(x).

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.

        Returns
        -------
        torch.Tensor
            Element-wise J1 of the input tensor.

        Notes
        -----
        - Uses torch.special.j1 which is differentiable.
        """
        self.validate(x)
        return torch.special.bessel_j1(x)
    
    def mean(self, x: torch.Tensor, dim: int | tuple[int, ...] = None, keepdim: bool = False) -> torch.Tensor:
        """
        Compute the mean of tensor elements over given axes.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor to compute the mean over.
        dim : int or tuple of int, optional
            Axis or axes along which to compute the mean. By default, compute over all axes.
        keepdim : bool, default=False
            If True, retains reduced dimensions with size 1.

        Returns
        -------
        torch.Tensor
            Mean of the tensor elements over the specified axes.

        Notes
        -----
        - Uses torch.mean which is differentiable.
        """
        self.validate(x)
        return torch.mean(x, dim=dim, keepdim=keepdim)
    # -------------------------------------------------------------------------
    # Optional 
    # -------------------------------------------------------------------------
    def astype(self, x: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        """
        Cast the input tensor `x` to the specified `dtype`.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor to cast.
        dtype : torch.dtype
            Target data type for the cast.

        Returns
        -------
        torch.Tensor
            A new tensor with the same data as `x` but cast to `dtype`.

        Requirements
        ------------
        - Must preserve device.
        """
        self.validate(x)
        return x.to(dtype=dtype, device=self.device)
    
    def detach(self, x: torch.Tensor) -> torch.Tensor:
        """
        Return a new tensor that is detached from the computation graph.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.

        Returns
        -------
        torch.Tensor
            A detached version of `x`.

        Notes
        -----
        - Uses x.detach() to create a tensor that does not track gradients.
        """
        self.validate(x)
        return x.detach()
    
    def no_grad(self) -> Any:
        """
        Context manager to disable gradient tracking using torch.no_grad().

        Returns
        -------
        context manager
            A context manager that disables gradient tracking within its scope.

        Notes
        -----
        - Uses torch.no_grad to disable gradient tracking.
        """
        return torch.no_grad()
    
    def jit(self, fn: Callable) -> Callable:
        """
        Wrap a function using torch.compile, if enabled.

        Parameters
        ----------
        fn : Callable
            Pure tensor function: fn(*tensors) -> tensors.

        Returns
        -------
        Callable
            If use_compile is True: a torch.compile optimized function.
            Otherwise: the original function unchanged.

        Notes
        -----
        - The returned function must not capture non-tensor state
          that torch.compile cannot trace (Python ints/floats okay).
        - Recommended: compile only high-level, pure functions.
        """
        if not self.use_compile:
            return fn

        # You can also specify mode="max-autotune" or others:
        # return torch.compile(fn, mode="max-autotune")
        return torch.compile(fn)

    def requires_grad(self, x: torch.Tensor, set: bool) -> torch.Tensor:
        """
        Set or unset the requires_grad flag on tensor `x`.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.
        set : bool
            If True, the returned tensor will require gradients. If False, it will not.
        Returns
        -------
        torch.Tensor
            A tensor that requires gradients.

        Notes
        -----
        - Uses x.requires_grad_(set) to enable or disable gradient tracking.
        """
        self.validate(x)
        if set not in (True, False):
            raise ValueError("set must be a boolean value")
        return x.requires_grad_(set)
    