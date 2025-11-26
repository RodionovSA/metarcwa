from abc import ABC, abstractmethod
from typing import Any, Callable

import torch

class Backend(ABC):
    """
    Abstract base class for computational backends used in RCWA/FMM simulations.

    A backend defines:
    - how arrays are created and stored,
    - how linear algebra and FFT operations are performed,
    - how complex numbers are handled,
    - and optionally how functions are JIT-compiled.

    This abstraction allows the same high-level RCWA code to run on
    different numerical engines (e.g. PyTorch, JAX, NumPy) without modification.
    All concrete backend implementations must satisfy the interface defined below.

    **Important guidelines for backend authors**
    -------------------------------------------
    - All returned arrays must be of the backend’s native array type.
    - Device and dtype must be consistent and controlled by the backend.
    - All operations must be pure (no in-place operations unless guaranteed safe).
    - `jit(fn)` must return a callable that is functionally equivalent to `fn`.
    """
    
    # ---------------------------------------------------------
    # Input validation / configuration
    # ---------------------------------------------------------
    @abstractmethod
    def validate(self, x: Any) -> None:
        """
        Validate that input `x` is compatible with this backend.
        """
    # ---------------------------------------------------------
    # Array creation / casting
    # ---------------------------------------------------------
    @abstractmethod
    def asarray(self, x, complex: bool = False) -> Any:
        """
        Convert input data to a backend-specific array.

        Parameters
        ----------
        x : Any
            Input data convertible to an array of this backend. This may be
            Python scalars, lists, NumPy arrays, or backend-native arrays.
        complex : bool, default=False
            If True, the returned array must use the backend's complex dtype.
            If False, the array must use the backend's real dtype.

        Returns
        -------
        array : backend-specific array type
            A tensor/array compatible with the backend's operations.

        Notes
        -----
        - Implementations must cast the array to the backend’s configured dtype.
        - Implementations must place the array on the backend’s configured device
          (e.g. CPU, GPU) if applicable.
        - This must never perform in-place modification of the input.
        """
        
    @abstractmethod
    def zeros(self, shape: tuple[int, ...]) -> Any:
        """
        Create a backend-specific array of zeros.

        Parameters
        ----------
        shape : tuple of int
            Shape of the output array.

        Returns
        -------
        array : backend-specific array type
            An array of zeros with the specified shape, dtype, and device.
        """
        
    @abstractmethod
    def ones(self, shape: tuple[int, ...]) -> Any:
        """
        Create a backend-specific array of ones.

        Parameters
        ----------
        shape : tuple of int
            Shape of the output array.

        Returns
        -------
        array : backend-specific array type
            An array of ones with the specified shape, dtype, and device.
        """
        
    @abstractmethod
    def zeros_like(self, x: Any) -> Any:
        """
        Create a backend-specific array of zeros with the same shape as `x`.

        Parameters
        ----------
        x : array-like
            Input array whose shape will be used.

        Returns
        -------
        array : backend-specific array type
            An array of zeros with the same shape as `x`, dtype, and device.
        """
        
    @abstractmethod
    def ones_like(self, x: Any) -> Any:
        """
        Create a backend-specific array of ones with the same shape as `x`.

        Parameters
        ----------
        x : array-like
            Input array whose shape will be used.

        Returns
        -------
        array : backend-specific array type
            An array of ones with the same shape as `x`, dtype, and device.
        """
        
    @abstractmethod
    def reshape(self, x: Any, shape: tuple[int, ...]) -> Any:
        """
        Reshape the input array `x` to the specified shape.
        """
        
    @abstractmethod
    def expand(self, x: Any, shape: tuple[int, ...]) -> Any:
        """
        Expand the input array `x` to the specified shape using broadcasting rules.
        """
        
    @abstractmethod
    def arange(self, start: int, stop: int, step: int = 1) -> Any:
        """
        Create a 1D array of evenly spaced values within a given interval.
        """
        
    @abstractmethod
    def shape(self, x: Any) -> tuple[int, ...]:
        """
        Return the shape of the input array `x`.
        """
        
    @abstractmethod
    def linspace(self, start: float, stop: float, num: int) -> Any:
        """
        Create a 1D array of `num` evenly spaced values between `start` and `stop`.
        """
        
    @abstractmethod
    def meshgrid(self, x: Any, y: Any, indexing: str = 'ij') -> tuple[Any, Any]:
        """
        Create coordinate matrices from coordinate vectors `x` and `y`.
        """
        
    @abstractmethod
    def clone(self, x: Any) -> Any:
        """
        Create a copy of the input array `x`.
        """
        
    @abstractmethod
    def where(self, condition: Any, x: Any, y: Any) -> Any:
        """
        Return elements chosen from `x` or `y` depending on `condition`.
        """

    # ---------------------------------------------------------
    # Fourier transforms
    # ---------------------------------------------------------
    @abstractmethod
    def fft2(self, x: Any) -> Any:
        """
        Compute the 2-dimensional discrete Fourier transform of `x`.

        Requirements for implementations
        --------------------------------
        - Must preserve dtype (real-to-complex cast happens automatically).
        - Must maintain device placement.
        - Must return a backend-native array with no Python objects inside.
        """

    @abstractmethod
    def ifft2(self, x: Any) -> Any:
        """
        Compute the 2-dimensional inverse discrete Fourier transform of `x`.

        Notes
        -----
        - Normalization convention must follow the backend's FFT semantics.
        - The output must be complex dtype if the input is complex.
        """

    @abstractmethod
    def fftshift(self, x: Any) -> Any:
        """
        Shift the zero-frequency component to the center of the spectrum.
        """
        
    @abstractmethod
    def ifftshift(self, x: Any) -> Any:
        """
        Inverse of fftshift: shift the center back to the original position.
        """
    # ---------------------------------------------------------
    # Linear algebra
    # ---------------------------------------------------------
    @abstractmethod
    def matmul(self, a: Any, b: Any) -> Any:
        """
        Perform matrix multiplication between backend arrays `a` and `b`.

        Parameters
        ----------
        a : array-like
            Left operand, typically of shape (..., M, K).
        b : array-like
            Right operand, typically of shape (..., K, N).

        Returns
        -------
        array
            Result of matrix multiplication, shape (..., M, N).

        Implementation Notes
        ---------------------
        - Must support batched matrix multiplication when the backend does.
        - Must propagate dtypes and devices correctly.
        - Must not modify inputs in place.
        """

    # ---------------------------------------------------------
    # Complex utilities
    # ---------------------------------------------------------
    @abstractmethod
    def conj(self, x: Any) -> Any:
        """
        Return the complex conjugate of `x`.

        Requirements
        ------------
        - For real inputs, must return a real array unchanged.
        - Must preserve device and dtype.
        """
        
    # ---------------------------------------------------------
    # Math function utilities
    # ---------------------------------------------------------
    @abstractmethod
    def sin(self, x: Any) -> Any:
        """
        Compute the element-wise sine of `x`.

        Requirements
        ------------
        - Must preserve device and dtype.
        """
        
    @abstractmethod
    def cos(self, x: Any) -> Any:
        """
        Compute the element-wise cosine of `x`.

        Requirements
        ------------
        - Must preserve device and dtype.
        """
        
    @abstractmethod
    def exp(self, x: Any) -> Any:
        """
        Compute the element-wise exponential of `x`.

        Requirements
        ------------
        - Must preserve device and dtype.
        """
        
    @abstractmethod
    def abs(self, x: Any) -> Any:
        """
        Compute the element-wise absolute value of `x`.

        Requirements
        ------------
        - Must preserve device and dtype.
        """

    @abstractmethod
    def mod(self, x, y): 
        """
        Compute the element-wise modulus of `x` by `y`.

        Requirements
        ------------
        - Must preserve device and dtype.
        """
    # ---------------------------------------------------------
    # Optional just-in-time compilation
    # ---------------------------------------------------------
    @abstractmethod
    def jit(self, fn: Callable) -> Callable:
        """
        JIT-compile or transform a function using backend-specific acceleration.

        Parameters
        ----------
        fn : Callable
            A pure function with signature `fn(*tensors) -> tensors`,
            where all inputs/outputs are backend-native arrays.

        Returns
        -------
        Callable
            A function equivalent to `fn`, optionally compiled or optimized.

        Behavior by backend type
        ------------------------
        - PyTorch backend: wraps the function with torch.compile(fn).
        - JAX backend: wraps with jax.jit(fn).
        - NumPy backend: typically returns `fn` unchanged (no-op).
        - Other backends: may implement specialized transformations.

        Requirements
        ------------
        - The returned function must be semantically identical to `fn`.
        - Must not capture Python objects that are not JIT-safe.
        - Must preserve differentiability if the backend supports it.
        """
        
    # ---------------------------------------------------------
    # Constants
    # ---------------------------------------------------------
    @property
    def pi(self) -> Any:
        """
        Return the mathematical constant π in the backend's array type.
        """
        return self.asarray(torch.pi)
        
class TorchBackend(Backend):
    """
    PyTorch backend implementation for RCWA/FMM computations.

    This backend provides:
    - dtype and device management,
    - array creation and casting,
    - FFT operations via torch.fft,
    - batched matrix multiplication,
    - complex utilities,
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
        
        if x.dtype != self.dtype and not x.dtype.is_complex:
            raise ValueError(
                f"Tensor dtype must be {self.dtype} or its complex counterpart, got {x.dtype}"
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
    
    def reshape(self, x: torch.Tensor, shape: tuple[int, ...]) -> torch.Tensor:
        """
        Reshape tensor `x` to the specified shape.
        """
        self.validate(x)
        return x.reshape(shape)

    def expand(self, x: torch.Tensor, shape: tuple[int, ...]) -> torch.Tensor:
        """
        Expand tensor `x` to the specified shape using broadcasting rules.
        """
        self.validate(x)
        return x.expand(shape)

    def arange(self, start: int, stop: int, step: int = 1) -> torch.Tensor:
        """
        Create a 1D tensor of evenly spaced values within [start, stop) with given step.
        """
        arr = torch.arange(start, stop, step, dtype=self.dtype, device=self.device)
        return arr
    
    def shape(self, x: torch.Tensor) -> tuple[int, ...]:
        """
        Return the shape of the input tensor `x`.
        """
        self.validate(x)
        return x.shape
    
    def linspace(self, start: float, stop: float, num: int) -> torch.Tensor:
        """
        Create a 1D tensor of `num` evenly spaced values between `start` and `stop`.
        """
        arr = torch.linspace(start, stop, num, dtype=self.dtype, device=self.device)
        return arr    
    
    def meshgrid(self, x, y, indexing = 'ij'):
        '''
        Create coordinate matrices from coordinate vectors `x` and `y`.
        '''
        return torch.meshgrid(x, y, indexing = indexing)
    
    def clone(self, x: torch.Tensor) -> torch.Tensor:
        """
        Create a copy of the input tensor `x`.
        """
        self.validate(x)
        return x.clone()
    
    def where(self, condition: torch.Tensor, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Return elements chosen from `x` or `y` depending on `condition`.
        """
        # condition must be boolean
        if not isinstance(condition, torch.Tensor):
            raise TypeError("condition must be a torch.Tensor")

        if condition.dtype != torch.bool:
            raise TypeError(f"condition must have dtype torch.bool, got {condition.dtype}")
        
        self.validate(x)
        self.validate(y)
        return torch.where(condition, x, y)
    
    # -------------------------------------------------------------------------
    # FFT Operations
    # -------------------------------------------------------------------------
    def fft2(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute a 2D FFT using torch.fft.

        Parameters
        ----------
        x : torch.Tensor
            Input array (..., Nx, Ny). Must be on this backend's device.

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
        return torch.fft.fft2(x)

    def ifft2(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute a 2D inverse FFT using torch.fft.

        Parameters
        ----------
        x : torch.Tensor
            Input array in Fourier domain.

        Returns
        -------
        torch.Tensor
            Inverse FFT of x, same shape, complex dtype.

        Notes
        -----
        - Uses PyTorch's default normalization convention ("backward").
        """
        self.validate(x)
        return torch.fft.ifft2(x)

    def fftshift(self, x: torch.Tensor) -> torch.Tensor:    
        """
        Shift the zero-frequency component to the center of the spectrum.
        """
        self.validate(x)
        return torch.fft.fftshift(x)
    
    def ifftshift(self, x: torch.Tensor) -> torch.Tensor:
        """
        Inverse of fftshift: shift the center back to the original position.
        """
        self.validate(x)
        return torch.fft.ifftshift(x)
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
    def sin(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute the element-wise sine of a tensor.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.

        Returns
        -------
        torch.Tensor
            Element-wise sine of x.

        Notes
        -----
        - Uses torch.sin which is differentiable.
        """
        self.validate(x)
        return torch.sin(x)
    
    def cos(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute the element-wise cosine of a tensor.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.

        Returns
        -------
        torch.Tensor
            Element-wise cosine of x.

        Notes
        -----
        - Uses torch.cos which is differentiable.
        """
        self.validate(x)
        return torch.cos(x)

    def exp(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute the element-wise exponential of a tensor.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.

        Returns
        -------
        torch.Tensor
            Element-wise exponential of x.

        Notes
        -----
        - Uses torch.exp which is differentiable.
        """
        self.validate(x)
        return torch.exp(x)
    
    def abs(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute the element-wise absolute value of a tensor.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.

        Returns
        -------
        torch.Tensor
            Element-wise absolute value of x.

        Notes
        -----
        - Uses torch.abs which is differentiable.
        """
        self.validate(x)
        return torch.abs(x)
    
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
    # -------------------------------------------------------------------------
    # Optional JIT Compilation
    # -------------------------------------------------------------------------
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

class JaxBackend(Backend):
    # Concrete implementation for JAX would go here
    pass

class NumpyBackend(Backend):
    # Concrete implementation for NumPy would go here
    pass