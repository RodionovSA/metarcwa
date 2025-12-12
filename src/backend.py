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
    # Data types
    # ---------------------------------------------------------
    @property
    @abstractmethod
    def float32(self) -> Any:
        """
        Return the backend's float32 data type.
        """
        
    @property
    @abstractmethod
    def float64(self) -> Any:
        """
        Return the backend's float64 data type.
        """
        
    @property
    @abstractmethod
    def complex64(self) -> Any:
        """
        Return the backend's complex64 data type.
        """
        
    @property
    @abstractmethod
    def complex128(self) -> Any:
        """
        Return the backend's complex128 data type.
        """
    
    @property
    @abstractmethod
    def long(self) -> Any:
        """
        Return the backend's long/integer data type.
        """
        
    @property
    @abstractmethod
    def int32(self) -> Any:
        """
        Return the backend's int32 data type.
        """
    
    @property
    @abstractmethod
    def bool(self) -> Any:
        """
        Return the backend's boolean data type.
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
    def eye(self, n: int) -> Any:
        """
        Create a backend-specific identity matrix of size n x n.

        Parameters
        ----------
        n : int
            Size of the identity matrix.

        Returns
        -------
        array : backend-specific array type
            An identity matrix of shape (n, n) with the correct dtype and device.
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

    @abstractmethod
    def unique(self, x: Any) -> Any:
        """
        Return the unique elements of the input array `x`.
        """
    
    @abstractmethod
    def any(self, x: Any) -> bool:
        """
        Return True if any element of `x` is True (non-zero).
        """
    
    @abstractmethod
    def clamp(self, x: Any, min_value: float, max_value: float) -> Any:
        """
        Clamp all elements in `x` to be within the range [min_value, max_value].
        """
    
    @abstractmethod
    def cat(self, arrays: list[Any], dim: int) -> Any:
        """
        Concatenate a list of arrays along the specified dimension.
        """
    
    @abstractmethod
    def roll(self, x: Any, shifts: int, dims: int) -> Any:
        """
        Roll the elements of `x` along the specified dimension by `shifts`.
        """
    
    @abstractmethod
    def amin(self, x: Any, dim: int | tuple[int, ...] = None, keepdim: bool = False) -> Any:
        """
        Compute the minimum of array elements over given axes.

        Parameters
        ----------
        x : array-like
            Input array to compute the minimum over.
        dim : int or tuple of int, optional
            Axis or axes along which to compute the minimum. By default, compute over all axes.
        keepdim : bool, default=False
            If True, retains reduced dimensions with size 1.

        Returns
        -------
        array
            Minimum of the array elements over the specified axes.

        Requirements
        ------------
        - Must preserve device and dtype.
        """
    
    @abstractmethod
    def amax(self, x: Any, dim: int | tuple[int, ...] = None, keepdim: bool = False) -> Any:
        """
        Compute the maximum of array elements over given axes.

        Parameters
        ----------
        x : array-like
            Input array to compute the maximum over.
        dim : int or tuple of int, optional
            Axis or axes along which to compute the maximum. By default, compute over all axes.
        keepdim : bool, default=False
            If True, retains reduced dimensions with size 1.

        Returns
        -------
        array
            Maximum of the array elements over the specified axes.

        Requirements
        ------------
        - Must preserve device and dtype.
        """
    
    @abstractmethod
    def nonzero(self, x: Any) -> Any:
        """
        Return the indices of the non-zero elements of `x`.
        """
    
    @abstractmethod
    def take_along_axis(self, arr: Any, indices: Any, axis: int) -> Any:
        """
        Take values from `arr` along the specified axis using `indices`.
        """
    
    @abstractmethod
    def unsqueeze(self, x: Any, axis: int) -> Any:
        """
        Add a singleton dimension to `x` at the specified axis.
        """
    
    @abstractmethod
    def repeat(self, x: Any, repeats: tuple[int, ...]) -> Any:
        """
        Repeat the elements of `x` according to `repeats` along each dimension.
        """
    
    @abstractmethod
    def repeat_interleave(self, x: Any, repeats: int, dim: int) -> Any:
        """
        Repeat elements of `x` along dimension `dim` `repeats` times.
        """
    
    @abstractmethod
    def stack(self, arrays: list[Any], dim: int) -> Any:
        """
        Stack a sequence of arrays along a new dimension.
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

    @abstractmethod
    def diag_embed(self, x: Any, dim1: int = -2, dim2: int = -1) -> Any:
        """
        Create a batch of diagonal matrices from the last dimension of `x`.

        Parameters
        ----------
        x : array-like
            Input array of shape (..., N), where N is the size of the diagonal.
        dim1 : int, default=-2
            The first dimension of the output diagonal matrices.
        dim2 : int, default=-1
            The second dimension of the output diagonal matrices.

        Returns
        -------
        array
            Output array of shape (..., N, N) where each slice along the leading
            dimensions is a diagonal matrix with the corresponding elements from `x`.
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

    @abstractmethod
    def sigmoid(self, x: Any) -> Any:
        """
        Compute the element-wise sigmoid function of `x`.

        Requirements
        ------------
        - Must preserve device and dtype.
        """
    
    @abstractmethod
    def sqrt(self, x: Any) -> Any:
        """
        Compute the element-wise square root of `x`.

        Requirements
        ------------
        - Must preserve device and dtype.
        """
    
    @abstractmethod
    def sum(self, x: Any, dim: int | tuple[int, ...] = None, keepdim: bool = False) -> Any:
        """
        Compute the sum of array elements over given axes.

        Parameters
        ----------
        x : array-like
            Input array to sum over.
        dim : int or tuple of int, optional
            Axis or axes along which to sum. By default, sum over all axes.
        keepdim : bool, default=False
            If True, retains reduced dimensions with size 1.

        Returns
        -------
        array
            Sum of the array elements over the specified axes.

        Requirements
        ------------
        - Must preserve device and dtype.
        """
    # ---------------------------------------------------------
    # Optional 
    # ---------------------------------------------------------
    @abstractmethod
    def astype(self, x: Any, dtype: Any) -> Any:
        """
        Cast the input array `x` to the specified `dtype`.

        Parameters
        ----------
        x : array-like
            Input array to cast.
        dtype : Any
            Target data type for the cast.

        Returns
        -------
        array
            A new array with the same data as `x` but cast to `dtype`.

        Requirements
        ------------
        - Must preserve device.
        """
    
    @abstractmethod
    def detach(self, x: Any) -> Any:
        """
        Return a new array that is detached from the computation graph, if applicable.

        Returns
        -------
        array
            A detached version of `x`.

        Notes
        -----
        - For backends without automatic differentiation, this may be a no-op.
        - For backends with AD (e.g. PyTorch, JAX), this should return an array
          that does not track gradients.
        """
    
    @abstractmethod
    def no_grad(self) -> Any:
        """
        Context manager to disable gradient tracking, if supported by the backend.

        Returns
        -------
        context manager
            A context manager that disables gradient tracking within its scope.

        Notes
        -----
        - For backends without automatic differentiation, this may be a no-op.
        - For backends with AD (e.g. PyTorch, JAX), this should disable gradient tracking.
        """
        
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
    # ---------------------------------------------------------
    # Data types
    # ---------------------------------------------------------
    @property
    def float32(self) -> torch.dtype:
        """The torch.float32 data type."""
        return torch.float32
    
    @property
    def float64(self) -> torch.dtype:
        """The torch.float64 data type."""
        return torch.float64
    
    @property
    def complex64(self) -> torch.dtype:
        """The torch.complex64 data type."""
        return torch.complex64
    
    @property
    def complex128(self) -> torch.dtype:
        """The torch.complex128 data type."""
        return torch.complex128
    
    @property
    def long(self) -> torch.dtype:
        """The torch.int64 (long) data type."""
        return torch.int64
    
    @property
    def int32(self) -> torch.dtype:
        """The torch.int32 data type."""
        return torch.int32
    
    @property
    def bool(self) -> torch.dtype:
        """The torch.bool data type."""
        return torch.bool
    
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
    
    def unique(self, x: torch.Tensor) -> torch.Tensor:
        """
        Return the unique elements of the input tensor `x`.
        """
        self.validate(x)
        return torch.unique(x)
    
    def any(self, x: torch.Tensor) -> torch.Tensor:
        """
        Return True if any element of `x` is True (non-zero).
        """
        # condition must be boolean
        if not isinstance(x, torch.Tensor):
            raise TypeError("x must be a torch.Tensor")

        if x.dtype != torch.bool:
            raise TypeError(f"x must have dtype torch.bool, got {x.dtype}")
        
        return torch.any(x)
    
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
    
    def take_along_axis(self, arr: torch.Tensor, indices: torch.Tensor, axis: int) -> torch.Tensor:
        """
        Take values from `arr` along the specified axis using `indices`.
        """
        self.validate(arr)
        return torch.take_along_dim(arr, indices, dim=axis)
    
    def unsqueeze(self, x: torch.Tensor, axis: int) -> torch.Tensor:
        """
        Add a singleton dimension to `x` at the specified axis.
        """
        self.validate(x)
        return torch.unsqueeze(x, dim=axis)
    
    def repeat(self, x: torch.Tensor, repeats: tuple[int, ...]) -> torch.Tensor:
        """
        Repeat the elements of `x` according to `repeats` along each dimension.
        """
        self.validate(x)
        return x.repeat(repeats)
    
    def repeat_interleave(self, x: torch.Tensor, repeats: int, dim: int) -> torch.Tensor:
        """
        Repeat elements of `x` along dimension `dim` `repeats` times.
        """
        self.validate(x)
        return x.repeat_interleave(repeats, dim=dim)
    
    def stack(self, arrays: list[torch.Tensor], dim: int) -> torch.Tensor:
        """
        Stack a sequence of tensors along a new dimension.
        """
        for arr in arrays:
            self.validate(arr)
        return torch.stack(arrays, dim=dim)
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
    
    def sqrt(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute the element-wise square root of `x`.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.

        Returns
        -------
        torch.Tensor
            Element-wise square root of x.

        Notes
        -----
        - Uses torch.sqrt which is differentiable.
        """
        self.validate(x)
        return torch.sqrt(x)
    
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

class JaxBackend(Backend):
    # Concrete implementation for JAX would go here
    pass

class NumpyBackend(Backend):
    # Concrete implementation for NumPy would go here
    pass