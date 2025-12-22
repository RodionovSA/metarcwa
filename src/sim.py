from abc import ABC, abstractmethod
from typing import Any, Tuple

from src.layer import Layer
from src.source import Source
from src.config import LayerConfig
from src.tvf import TVF
from src.compute import custom_fft, build_index_map, toeplitz_2d, diagonal_K_matrices

class LayerSolver:
    """ Public interface for RCWA layer solvers."""
    
    def __new__(cls, 
                layer: Layer, 
                source: Source, 
                cfg: LayerConfig, 
                n_inc: Any,
                index_map: Any = None):
        
        n_inc = layer.backend.asarray(n_inc, complex=False)
        cls._init_validation(layer, source, cfg, n_inc, index_map)

        solver_cls = _select_solver_class(layer, cfg)
        obj = super().__new__(solver_cls)
        return obj
    
    def __init__(self, 
                 layer: Layer, 
                 source: Source, 
                 cfg: LayerConfig,
                 n_inc: Any, 
                 index_map: Any = None):
        """
        Initialize the LayerSolver object with given layer and source.
        
        Parameters
        ----------
        layer : Layer
            The layer object defining the RCWA structure.
        source : Source
            The source object defining the incident wave parameters.
        cfg : LayerConfig
            Configuration parameters for the LayerSolver.
        n_inc : Any
            Refractive index of the incident medium.
        index_map : Any, optional
            Precomputed index lookup for 2D Toeplitz convolution.
        """
        self._layer = layer
        self._source = source
        self._cfg = cfg
        
        self._n_inc = layer.backend.asarray(n_inc, complex=False)
        if self._n_inc.shape[0] == 1:
            self._n_inc = self.layer.backend.expand(self._n_inc, (len(source.wavelength),))
        
        self._index_map = index_map if index_map is not None else build_index_map(
            layer.backend, cfg.M, cfg.N, circular=cfg.circ_truncation
        )
        
    @property
    def layer(self):
        return self._layer
    @property
    def source(self):
        return self._source
    @property
    def cfg(self):
        return self._cfg
    @property
    def backend(self):
        return self.layer.backend
    @property
    def lattice(self):
        return self.layer.lattice
    @property
    def index_map(self):
        return self._index_map
    @property
    def n_inc(self):
        return self._n_inc
    
    @staticmethod
    def _init_validation(layer: Layer, 
                         source: Source, 
                         cfg: LayerConfig,
                         n_inc: Any,
                         index_map: Any = None) -> None:
        
        if not isinstance(layer, Layer):
            raise TypeError("layer must be an instance of Layer.")
        if not isinstance(source, Source):
            raise TypeError("source must be an instance of Source.")
        if layer.backend != source.backend:
            raise ValueError("layer and source must use the same backend.")
        
        if len(source.wavelength) not in (layer.epsilon.shape[0], layer.mu.shape[0]):
            raise ValueError(
                "Length of source.wavelength must match length of layer.epsilon, layer.mu."
            )
        
        if not isinstance(cfg, LayerConfig):
            raise TypeError("cfg must be an instance of LayerConfig.")
        
        if len(n_inc.shape) != 1:
            raise ValueError("n_inc must be a 1D array of refractive indices.")
        
        if n_inc.shape[0] != len(source.wavelength) and n_inc.shape[0] > 1:
            raise ValueError(
                "Length of n_inc must match length of source.wavelength."
            )
        
        if index_map is not None and index_map.ndim != 2:
            raise ValueError("index_map must be a 2D array")
        
# Helper function to select solver class   
def _select_solver_class(layer: Layer, cfg: LayerConfig):
    """
    Decide which solver implementation to use.
    """

    if layer.type == "isotropic" and not layer.is_magnetic:
        return LayerSolverIsotropic

    raise NotImplementedError(
        f"No solver implemented for layer type: {layer}"
    )

class _BaseLayerSolver(LayerSolver, ABC):
    """
    Internal base class for all layer solvers.
    """
    
    @abstractmethod
    def _prepare_tvf(self):
        pass
    
    @abstractmethod
    def correction_term(self):
        pass
    
    @abstractmethod
    def Epsilon2(self):
        pass
    
    def apply_inverse(self, matrix_A: Any, matrix_B: Any) -> Any:
        """
        Apply the inverse of matrix_A to matrix_B using the configured method.

        Parameters
        ----------
        matrix_A : Any
            The matrix to be inverted.
        matrix_B : Any
            An auxiliary matrix.

        Returns
        -------
        matrix_A_inv_B : Any
            Product of the inverse of matrix_A and matrix_B.
        """
        if self.cfg.inverse_matrix_method == 'solve':
            matrix_A_inv_B = self.backend.solve(matrix_A, matrix_B)
        elif self.cfg.inverse_matrix_method == 'inv':
            matrix_A_inv_B = self.backend.matmul(self.backend.inv(matrix_A), matrix_B)
        elif self.cfg.inverse_matrix_method == 'pinv':
            matrix_A_inv_B = self.backend.matmul(self.backend.pinv(matrix_A), matrix_B)
        else:
            raise ValueError(f"Unknown inverse matrix method: {self.cfg.inverse_matrix_method}")
        
        return matrix_A_inv_B

    def _build_k_matrices(self):
        """
        Build the Kx and Ky diagonal matrices for the layer.
        """
        Kx, Ky = self.source.Kxy(self.n_inc, 2*self.cfg.M, 2*self.cfg.N, self.lattice)  # shape: (wvl, theta, phi, 4M+1, 4N+1)
        Kx_t, Ky_t = diagonal_K_matrices(self.backend, Kx, Ky, self._index_map[0], self._index_map[1]) # shape: (wvl, theta, phi, Nh, Nh)
        
        return Kx_t, Ky_t
    
    def Omega2(self) -> Any:
        """
        Construct the Omega^2 matrix for the layer.
        
        Returns
        -------
        Omega2_xx, Omega2_yy, Omega2_xy, Omega2_yx : Any
            Components of the Omega^2 matrix.
        """
        # Get P and Q matrices
        P_xx, P_yy, P_xy, P_yx = self.Pmatrix()  # shape: (wvl, theta, phi, Nh, Nh)
        Q_xx, Q_yy, Q_xy, Q_yx = self.Qmatrix()  # shape: (wvl, theta, phi, Nh, Nh)
        
        # Build Omega^2 matrix
        Omega2_xx = self.backend.matmul(P_xy, Q_yx) - self.backend.matmul(P_xx, Q_xx)  # shape: (wvl, theta, phi, Nh, Nh)
        Omega2_xy = self.backend.matmul(P_xy, Q_yy) - self.backend.matmul(P_xx, Q_xy)  # shape: (wvl, theta, phi, Nh, Nh)
        Omega2_yx = self.backend.matmul(P_yx, Q_yx) - self.backend.matmul(P_yy, Q_xx)  # shape: (wvl, theta, phi, Nh, Nh)
        Omega2_yy = self.backend.matmul(P_yx, Q_yy) - self.backend.matmul(P_yy, Q_xy)  # shape: (wvl, theta, phi, Nh, Nh)
        
        return Omega2_xx, Omega2_yy, Omega2_xy, Omega2_yx



class LayerSolverIsotropic(_BaseLayerSolver):
    """ Layer solver for isotropic, non-magnetic layers. """
        
    def _prepare_tvf(self) -> Tuple[Any, Any, Any, Any]:
        """
        Prepare the Tangent Vector Fields (TVF) for the layer.
        
        Returns
        -------
        Pxx_fft, Pyy_fft, Pxy_fft, Pyx_fft : Any
            Fourier coefficients of the projector components. Shape: (1, 4M+1, 4N+1)
        """
        # Extract permittivity field from layer 
        epsilon_xy = self.layer.epsilon_xy()[0,0,0,:,:]  # shape: (Nx, Ny) Take epsilon_xx for the first wavelength
        epsilon_xy = self.backend.unsqueeze(epsilon_xy, 0)  # shape: (1, Nx, Ny)
        
        # Initialize TVF object
        self._tvf = TVF(
            backend=self.backend,
            lattice=self.lattice,
            M=self.cfg.M,
            N=self.cfg.N,
            method=self.cfg.factorization,
            optimizer=self.cfg.tvf_optimizer,
        )
        
        # Compute TVF
        tx, ty = self._tvf.compute(epsilon_xy, 
                                   alpha=self.cfg.tvf_alpha, 
                                   beta=self.cfg.tvf_beta, 
                                   gamma=self.cfg.tvf_gamma, 
                                   steps=self.cfg.tvf_steps)
        
        # Compute required TVF products
        tx2 = self.backend.abs(tx)**2                      # |tx|^2
        ty2 = self.backend.abs(ty)**2                      # |ty|^2
        den = tx2 + ty2 + 1e-15                            # |tx|^2 + |ty|^2

        Pxx = tx2 / den                                    # projector components
        Pyy = ty2 / den
        Pxy = (tx * self.backend.conj(ty)) / den
        Pyx = self.backend.conj(Pxy)                       # enforce Hermitian
        
        # Do fft
        Pxx_fft = custom_fft(self.backend, Pxx, 2*self.cfg.M, 2*self.cfg.N)
        Pyy_fft = custom_fft(self.backend, Pyy, 2*self.cfg.M, 2*self.cfg.N)
        Pxy_fft = custom_fft(self.backend, Pxy, 2*self.cfg.M, 2*self.cfg.N)
        Pyx_fft = custom_fft(self.backend, Pyx, 2*self.cfg.M, 2*self.cfg.N)
        
        return Pxx_fft, Pyy_fft, Pxy_fft, Pyx_fft
    
    def correction_term(self) -> Tuple[Any, Any, Any, Any]:
        """
        Compute the Li's Factorization correction term for the isotropic layer.
        Notation is from S4 paper by Liu et al. (2012).
        
        Returns
        -------
        Delta_Pxx, Delta_Pyy, Delta_Pxy, Delta_Pyx : Any
            Correction terms for the projector components.
        """
        # Get Fourier coefficients for epsilon_xy
        epsilon_mn = self.layer.epsilon_mn(
            2*self.cfg.M, 2*self.cfg.N, use_closed_form=self.cfg.closed_form
        )[:,0,0,...]  # shape: (wvl, 4M+1, 4N+1)
        
        inv_epsilon_mn = self.layer.epsilon_mn(
            2*self.cfg.M, 2*self.cfg.N, use_closed_form=self.cfg.closed_form,
            inverse=True, regularized=True, regularization=self.cfg.inverse_regularization
        )[:,0,0,...]  # shape: (wvl, 4M+1, 4N+1)
        
        # Get TVF projectors
        Pxx_fft, Pyy_fft, Pxy_fft, Pyx_fft = self._prepare_tvf()  # shape: (1, 4M+1, 4N+1)
        
        # Get toeplitz convolution matrices
        Epsilon_t = toeplitz_2d(self.backend, epsilon_mn, self._index_map[0], self._index_map[1])  # shape: (wvl, Nh, Nh)
        Epsilon_inv_t = toeplitz_2d(self.backend, inv_epsilon_mn, self._index_map[0], self._index_map[1])  # shape: (wvl, Nh, Nh)
        Pxx_t = toeplitz_2d(self.backend, Pxx_fft, self._index_map[0], self._index_map[1])  # shape: (1, Nh, Nh)
        Pyy_t = toeplitz_2d(self.backend, Pyy_fft, self._index_map[0], self._index_map[1])  # shape: (1, Nh, Nh)
        Pxy_t = toeplitz_2d(self.backend, Pxy_fft, self._index_map[0], self._index_map[1])  # shape: (1, Nh, Nh)
        Pyx_t = toeplitz_2d(self.backend, Pyx_fft, self._index_map[0], self._index_map[1])  # shape: (1, Nh, Nh)
        
        # Invert Epsilon_inv
        Epsilon_inv_t_inv_Pxx = self.apply_inverse(Epsilon_inv_t, Pxx_t)  # shape: (wvl, Nh, Nh)
        Epsilon_inv_t_inv_Pyy = self.apply_inverse(Epsilon_inv_t, Pyy_t)  # shape: (wvl, Nh, Nh)
        Epsilon_inv_t_inv_Pxy = self.apply_inverse(Epsilon_inv_t, Pxy_t)  # shape: (wvl, Nh, Nh)
        Epsilon_inv_t_inv_Pyx = self.apply_inverse(Epsilon_inv_t, Pyx_t)  # shape: (wvl, Nh, Nh)
        
        # Compute correction term
        Delta_Pxx = self.backend.matmul(Epsilon_t, Pxx_t) - Epsilon_inv_t_inv_Pxx # shape: (wvl, Nh, Nh)
        Delta_Pyy = self.backend.matmul(Epsilon_t, Pyy_t) - Epsilon_inv_t_inv_Pyy  # shape: (wvl, Nh, Nh)
        Delta_Pxy = self.backend.matmul(Epsilon_t, Pxy_t) - Epsilon_inv_t_inv_Pxy  # shape: (wvl, Nh, Nh)
        Delta_Pyx = self.backend.matmul(Epsilon_t, Pyx_t) - Epsilon_inv_t_inv_Pyx  # shape: (wvl, Nh, Nh)
        
        return Delta_Pxx, Delta_Pyy, Delta_Pxy, Delta_Pyx
    
    def Epsilon2(self) -> Any:
        """
        Construct the permittivity tensor Epsilon2 for the isotropic layer ([D]=Epsilon2[E]).
        Notation is from S4 paper by Liu et al. (2012).
        ** Important Note **: In the S4 paper, there is a typo in equations (48) - (51), where x and y
        components are swapped for the diagonal terms. The implementation here follows the correct formulation.
        
        Returns
        -------
        Epsilon2_xx, Epsilon2_yy, Epsilon2_xy, Epsilon2_yx : Any
            Components of the permittivity tensor Epsilon2.
        """
        # Get Fourier coefficients for epsilon_xy
        epsilon_mn = self.layer.epsilon_mn(
            2*self.cfg.M, 2*self.cfg.N, use_closed_form=self.cfg.closed_form
        )[:,0,0,...]  # shape: (wvl, 4M+1, 4N+1)
        
        # Get toeplitz convolution matrices
        Epsilon_t = toeplitz_2d(self.backend, epsilon_mn, self._index_map[0], self._index_map[1])  # shape: (wvl, Nh, Nh)
        
        # Get Li's correction terms
        Delta_Pxx, Delta_Pyy, Delta_Pxy, Delta_Pyx = self.correction_term()  # shape: (wvl, Nh, Nh)
        
        Epsilon2_xx = Epsilon_t - Delta_Pxx  # shape: (wvl, Nh, Nh) (Note the swap due to typo in original paper)
        Epsilon2_yy = Epsilon_t - Delta_Pyy  # shape: (wvl, Nh, Nh) (Note the swap due to typo in original paper)
        Epsilon2_xy = - Delta_Pxy            # shape: (wvl, Nh, Nh)
        Epsilon2_yx = - Delta_Pyx            # shape: (wvl, Nh, Nh)
        
        return Epsilon2_xx, Epsilon2_yy, Epsilon2_xy, Epsilon2_yx
    
    def Pmatrix(self):
        """
        Construct the P matrix components.
        
        Returns
        -------
        Pmatrix_xx, Pmatrix_yy, Pmatrix_xy, Pmatrix_yx : Any
            Components of the P matrix.
        """ 
        # Get Kx and Ky matrices
        Kx_t, Ky_t = self._build_k_matrices() # shape: (wvl, theta, phi, Nh, Nh)
        
        # Get Fourier coefficients for epsilon_xy
        epsilon_mn = self.layer.epsilon_mn(
            2*self.cfg.M, 2*self.cfg.N, use_closed_form=self.cfg.closed_form
        )[:,0,0,...]  # shape: (wvl, 4M+1, 4N+1)
        
        # Get toeplitz convolution matrices
        Epsilon_t = toeplitz_2d(self.backend, epsilon_mn, self._index_map[0], self._index_map[1])  # shape: (wvl, Nh, Nh)
        
        wvl = Epsilon_t.shape[0]
        Nh  = Epsilon_t.shape[-1]
        theta = Kx_t.shape[1]
        phi   = Kx_t.shape[2]

        Epsilon_t = self.backend.reshape(Epsilon_t, (wvl, 1, 1, Nh, Nh))
        Epsilon_t = self.backend.expand(Epsilon_t, (wvl, theta, phi, Nh, Nh))
        
        # Build inverse components
        Epsilon_t_inv_Kx = self.apply_inverse(Epsilon_t, Kx_t)  # shape: (wvl, theta, phi, Nh, Nh)
        Epsilon_t_inv_Ky = self.apply_inverse(Epsilon_t, Ky_t)  # shape: (wvl, theta, phi, Nh, Nh)
        
        # Build identity matrix
        Nh = Epsilon_t.shape[-1]
        I = self.backend.eye(Nh, Nh)
        I = self.backend.reshape(I, (1, 1, 1, Nh, Nh))  # shape: (1, 1, 1, Nh, Nh)
        I = self.backend.expand(I, (wvl, theta, phi, Nh, Nh))  # shape: (wvl, theta, phi, Nh, Nh)
        
        # Build P matrix
        Pmatrix_xx = -self.backend.matmul(Kx_t, Epsilon_t_inv_Ky)  # shape: (wvl, theta, phi, Nh, Nh)
        Pmatrix_xy = - I + self.backend.matmul(Kx_t, Epsilon_t_inv_Kx)   # shape: (wvl, theta, phi, Nh, Nh)
        Pmatrix_yx = I - self.backend.matmul(Ky_t, Epsilon_t_inv_Ky)    # shape: (wvl, theta, phi, Nh, Nh)
        Pmatrix_yy = self.backend.matmul(Ky_t, Epsilon_t_inv_Kx)   # shape: (wvl, theta, phi, Nh, Nh)
        
        return Pmatrix_xx, Pmatrix_yy, Pmatrix_xy, Pmatrix_yx
    
    def Qmatrix(self):
        """
        Construct the Q matrix components.
        
        Returns
        -------
        Qmatrix_xx, Qmatrix_yy, Qmatrix_xy, Qmatrix_yx : Any
            Components of the Q matrix.
        """ 
        # Get Kx and Ky matrices
        Kx_t, Ky_t = self._build_k_matrices() # shape: (wvl, theta, phi, Nh, Nh)
        
        # Get Epsilon2 components
        Epsilon2_xx, Epsilon2_yy, Epsilon2_xy, Epsilon2_yx = self.Epsilon2()  # shape: (wvl, Nh, Nh)
        
        Nh = Epsilon2_xx.shape[-1]
        wvl = Epsilon2_xx.shape[0]
        theta = Kx_t.shape[1]
        phi   = Kx_t.shape[2]
        
        Epsilon2_xx = self.backend.reshape(Epsilon2_xx, (wvl, 1, 1, Nh, Nh))  # shape: (wvl, 1, 1, Nh, Nh)
        Epsilon2_yy = self.backend.reshape(Epsilon2_yy, (wvl, 1, 1, Nh, Nh))  # shape: (wvl, 1, 1, Nh, Nh)
        Epsilon2_xy = self.backend.reshape(Epsilon2_xy, (wvl, 1, 1, Nh, Nh))  # shape: (wvl, 1, 1, Nh, Nh)
        Epsilon2_yx = self.backend.reshape(Epsilon2_yx, (wvl, 1, 1, Nh, Nh))  # shape: (wvl, 1, 1, Nh, Nh)
        
        Epsilon2_xx = self.backend.expand(Epsilon2_xx, (wvl, theta, phi, Nh, Nh))  # shape: (wvl, theta, phi, Nh, Nh)
        Epsilon2_yy = self.backend.expand(Epsilon2_yy, (wvl, theta, phi, Nh, Nh))  # shape: (wvl, theta, phi, Nh, Nh) 
        Epsilon2_xy = self.backend.expand(Epsilon2_xy, (wvl, theta, phi, Nh, Nh))  # shape: (wvl, theta, phi, Nh, Nh)
        Epsilon2_yx = self.backend.expand(Epsilon2_yx, (wvl, theta, phi, Nh, Nh))  # shape: (wvl, theta, phi, Nh, Nh)
        
        # Build Q matrix
        Qmatrix_xx = - self.backend.matmul(Kx_t, Ky_t) - Epsilon2_yx # shape: (wvl, theta, phi, Nh, Nh)
        Qmatrix_xy = self.backend.matmul(Kx_t, Kx_t) - Epsilon2_yy    # shape: (wvl, theta, phi, Nh, Nh)
        Qmatrix_yx = Epsilon2_xx - self.backend.matmul(Ky_t, Ky_t)     # shape: (wvl, theta, phi, Nh, Nh)
        Qmatrix_yy = self.backend.matmul(Ky_t, Kx_t) + Epsilon2_xy  # shape: (wvl, theta, phi, Nh, Nh)
        
        return Qmatrix_xx, Qmatrix_yy, Qmatrix_xy, Qmatrix_yx
        
    
    
        
        
        

