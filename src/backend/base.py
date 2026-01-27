#src/backend/base.py
# Backend Interface for RCWA/FMM computations

from abc import ABC, abstractmethod
from typing import Any, Callable

class Backend(ABC):
    """
    Abstract base class for computational backends used in RCWA/FMM simulations.

    A backend defines:
    - how arrays are created and stored,
    - how linear algebra and FFT operations are performed,
    - how complex numbers are handled,
    - and optionally how functions are JIT-compiled.

    This abstraction allows the same high-level RCWA code to run on
    different numerical engines (e.g. PyTorch, JAX, NumPy, Custom) without modification.
    All concrete backend implementations must satisfy the interface defined below.

    **Important guidelines for backend authors**
    -------------------------------------------
    - All returned arrays must be of the backend’s native array type.
    - Device and dtype must be consistent and controlled by the backend.
    - All operations must be pure (no in-place operations unless guaranteed safe).
    - `jit(fn)` must return a callable that is functionally equivalent to `fn`.
    - Must have xp attribute pointing to the backend module (e.g. torch, jax.numpy).
    """
    
    @property
    @abstractmethod
    def xp(self):
        """
        Numerical backend module.

        This module provides the array/tensor API used throughout the codebase,
        enabling backend-agnostic implementations (e.g., NumPy, PyTorch, JAX).

        The backend module **must** expose, at minimum, the following functions:
            cos, sin, exp, sqrt, abs, sign, where,
            shape, reshape, meshgrid, real, imag

        It must also define standard numerical dtypes such as:
            bool, float, long (or equivalent integer type)

        Returns
        -------
        module
            The numerical backend module (e.g., numpy, torch, jax.numpy).
        """
        pass


    @property
    @abstractmethod
    def name(self):
        """
        Backend name identifier.

        Returns
        -------
        str
            Canonical backend name (e.g., "numpy", "torch", "jax").
        """
        pass
    
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
    def asarray(self, x: Any, complex: bool = False) -> Any:
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
    def arange(self, start: int, stop: int, step: int = 1) -> Any:
        """
        Create a 1D array of evenly spaced values within a given interval.
        """
        
    @abstractmethod
    def linspace(self, start: float, stop: float, num: int) -> Any:
        """
        Create a 1D array of `num` evenly spaced values between `start` and `stop`.
        """
        
    @abstractmethod
    def clone(self, x: Any) -> Any:
        """
        Create a copy of the input array `x`.
        """
        
    # ---------------------------------------------------------
    # Array operations
    # ---------------------------------------------------------
    @abstractmethod
    def resample(self, x: Any, new_shape: tuple[int, int]) -> Any:
        """
        Resample 2D array `x` to the specified `new_shape` using interpolation.

        Parameters
        ----------
        x : array-like
            Input 2D array to be resampled.
        new_shape : tuple of int
            Desired output shape (new_Nx, new_Ny).

        Returns
        -------
        array : backend-specific array type
            Resampled array with shape `new_shape`.
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
    def stack(self, arrays: list[Any], dim: int) -> Any:
        """
        Stack a sequence of arrays along a new dimension.
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
    def expand(self, x: Any, shape: tuple[int, ...]) -> Any:
        """
        Expand the input array `x` to the specified shape using broadcasting rules.
        """
    
    @abstractmethod
    def take_along_axis(self, arr: Any, indices: Any, dim: int) -> Any:
        """
        Take values from `arr` along the specified axis using `indices`.
        """
    
    @abstractmethod
    def unsqueeze(self, x: Any, dim: int) -> Any:
        """
        Add a singleton dimension to `x` at the specified axis.
        """
    
    @abstractmethod
    def squeeze(self, x: Any, dim: int) -> Any:
        """
        Remove a singleton dimension from `x` at the specified axis.
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
    
    @abstractmethod
    def solve(self, A: Any, B: Any) -> Any:
        """
        Solve the linear system A * X = B for X.

        Parameters
        ----------
        A : array-like
            Coefficient matrix of shape (..., M, M).
        B : array-like
            Ordinate or right-hand side matrix of shape (..., M, K).

        Returns
        -------
        array
            Solution matrix X of shape (..., M, K).

        Implementation Notes
        ---------------------
        - Must support batched solving when the backend does.
        - Must propagate dtypes and devices correctly.
        - Must not modify inputs in place.
        """
    
    @abstractmethod
    def inv(self, A: Any) -> Any:
        """
        Compute the inverse of square matrix A.

        Parameters
        ----------
        A : array-like
            Input square matrix of shape (..., M, M).

        Returns
        -------
        array
            Inverse of A, shape (..., M, M).

        Implementation Notes
        ---------------------
        - Must support batched inversion when the backend does.
        - Must propagate dtypes and devices correctly.
        - Must not modify inputs in place.
        """
    
    @abstractmethod
    def pinv(self, A: Any) -> Any:
        """
        Compute the Moore-Penrose pseudo-inverse of matrix A.

        Parameters
        ----------
        A : array-like
            Input matrix of shape (..., M, N).

        Returns
        -------
        array
            Pseudo-inverse of A, shape (..., N, M).

        Implementation Notes
        ---------------------
        - Must support batched pseudo-inversion when the backend does.
        - Must propagate dtypes and devices correctly.
        - Must not modify inputs in place.
        """
    
    @abstractmethod
    def eig(self, A: Any) -> tuple[Any, Any]:
        """
        Compute the eigenvalues and right eigenvectors of square matrix A.

        Parameters
        ----------
        A : array-like
            Input square matrix of shape (..., M, M).

        Returns
        -------
        tuple of arrays
            - eigenvalues: array of shape (..., M)
            - eigenvectors: array of shape (..., M, M)

        Implementation Notes
        ---------------------
        - Must support batched eigen decomposition when the backend does.
        - Must propagate dtypes and devices correctly.
        - Must not modify inputs in place.
        """
    
    @abstractmethod
    def eigh(self, A: Any) -> tuple[Any, Any]:
        """
        Compute the eigenvalues and eigenvectors of a Hermitian matrix A.

        Parameters
        ----------
        A : array-like
            Input Hermitian matrix of shape (..., M, M).

        Returns
        -------
        tuple of arrays
            - eigenvalues: array of shape (..., M)
            - eigenvectors: array of shape (..., M, M)

        Implementation Notes
        ---------------------
        - Must support batched eigen decomposition when the backend does.
        - Must propagate dtypes and devices correctly.
        - Must not modify inputs in place.
        """
    
    @abstractmethod
    def svd(self, A: Any) -> tuple[Any, Any, Any]:
        """
        Compute the singular value decomposition (SVD) of matrix A.

        Parameters
        ----------
        A : array-like
            Input matrix of shape (..., M, N).

        Returns
        -------
        tuple of arrays
            - U: left singular vectors, shape (..., M, M)
            - S: singular values, shape (..., min(M, N))
            - Vh: right singular vectors (conjugate transpose), shape (..., N, N)

        Implementation Notes
        ---------------------
        - Must support batched SVD when the backend does.
        - Must propagate dtypes and devices correctly.
        - Must not modify inputs in place.
        """
    
    @abstractmethod
    def qr(self, A: Any) -> tuple[Any, Any]:
        """
        Compute the QR decomposition of matrix A.

        Parameters
        ----------
        A : array-like
            Input matrix of shape (..., M, N).

        Returns
        -------
        tuple of arrays
            - Q: orthogonal matrix, shape (..., M, M)
            - R: upper triangular matrix, shape (..., M, N)

        Implementation Notes
        ---------------------
        - Must support batched QR decomposition when the backend does.
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
    
    @abstractmethod
    def besselj1(self, x: Any) -> Any:
        """
        Compute the Bessel function of the first kind of order one, J1(x).

        Parameters
        ----------
        x : array-like
            Input array.

        Returns
        -------
        array
            Element-wise J1 of the input array.

        Requirements
        ------------
        - Must preserve device and dtype.
        """
    
    @abstractmethod
    def mean(self, x: Any, dim: int | tuple[int, ...] = None, keepdim: bool = False) -> Any:
        """
        Compute the mean of array elements over given axes.

        Parameters
        ----------
        x : array-like
            Input array to compute the mean over.
        dim : int or tuple of int, optional
            Axis or axes along which to compute the mean. By default, compute over all axes.
        keepdim : bool, default=False
            If True, retains reduced dimensions with size 1.

        Returns
        -------
        array
            Mean of the array elements over the specified axes.

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
        
    @abstractmethod
    def requires_grad(self, x: Any, set: bool) -> bool:
        """
        Set or unset the requires_grad flag on tensor `x`.

        Returns
        -------
        x : array-like
            The same array as `x`, but with requires_grad set or unset.
        
        Notes
        -----
        - For backends without automatic differentiation, this may be a no-op.
        """
    # ---------------------------------------------------------
    # Constants
    # ---------------------------------------------------------
    @property
    def pi(self) -> Any:
        """
        Return the mathematical constant π in the backend's array type.
        """
        return self.asarray(3.141592653589793, complex=False)
    
    # ---------------------------------------------------------
    # Attribute forwarding
    # ---------------------------------------------------------
    def __getattr__(self, name):
        """
        Called ONLY if attribute not found on Backend.
        Forward to xp.
        """
        try:
            return getattr(self.xp, name)
        except AttributeError:
            raise AttributeError(
                f"{type(self).__name__} has no attribute '{name}' "
                f"and xp={type(self.xp).__name__} has no attribute '{name}'"
            )
        