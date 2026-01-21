from abc import ABC, abstractmethod
from typing import Any, Tuple

from src.model.layer import Layer
from src.model.source import Source
from src.config import LayerConfig
from src.tvf import TVF
from src.compute import build_index_map, toeplitz_2d, diagonal_K_matrices, kz_sign, flatten_Kxy
from src.compute import build_harmonic_grid, elliptical_truncation_mask, build_block, split_block
from src.model.geometry.geometry import custom_fft

class LayerSolver:
    """ Public interface for RCWA layer solvers."""
    
    def __new__(cls, 
                layer: Layer, 
                source: Source, 
                cfg: LayerConfig, 
                n_inc: Any,
                index_map: Any = None):
        
        n_inc = layer.backend.asarray(n_inc, complex=False)
        # promote scalar → 1D array
        if len(n_inc.shape) == 0:
            n_inc = layer.backend.reshape(n_inc, (1,))
            
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
        
        n_inc = layer.backend.asarray(n_inc, complex=False)
        # promote scalar → 1D array
        if len(n_inc.shape) == 0:
            n_inc = layer.backend.reshape(n_inc, (1,))
        # broadcast to wavelength if needed
        if n_inc.shape[0] == 1:
            n_inc = self.layer.backend.expand(n_inc, (len(source.wavelength),))
        
        self._n_inc = n_inc
        
        self._index_map = index_map if index_map is not None else build_index_map(
            layer.backend, cfg.M, cfg.N, circular=cfg.circ_truncation
        )
        
        # Dummy init
        self._W = None
        self._kz = None
        self._V = None
        
    def solve(self):
        #Calculate P and Q
        P = self._Pmatrix()
        Q = self._Qmatrix()
        
        # Calculate Omega^2
        Omega2 = self._Omega2(P, Q)
        
        # Solve eigenproblem and find lambda^2 and W
        lam2, self._W = self._eigensolver(Omega2)
        
        # Derive kz
        self._kz = kz_sign(self.backend, lam2)
        
        # Derive V=-jQW/kz
        self._V = self._compute_V(Q, self._W, self._kz)
        
    def Sm_int_right(self, W_ext: Any, V_ext: Any) -> Any:
        """ 
        Calculates S-matrix for an interface, where W_ext and V_ext for a medium on the left,
        and the layer on the right. 
        
        """
        WL, VL = W_ext, V_ext
        WR, VR = self._W, self._V
        
        return self._Sm_int(WL, VL, WR, VR)
    
    def Sm_int_left(self, W_ext: Any, V_ext: Any) -> Any:
        """ 
        Calculates S-matrix for an interface, where W_ext and V_ext for a medium on the right,
        and the layer on the left. 
        
        """
        WR, VR = W_ext, V_ext
        WL, VL = self._W, self._V
        
        return self._Sm_int(WL, VL, WR, VR)
    
    def Sm_prop(self):
        """Calculates S-matrix for a propagation inside the layer"""
        
        # Define z prime (k0*L)
        zp = self.source.k0 * self.layer.thickness # [wvl]
        zp = self.backend.reshape(zp, [zp.shape[0], 1, 1, 1]) # [wvl, 1, 1, 1]
        
        # Calculate propagation exponent
        exp = self.backend.exp(1j*self._kz*zp) # [wvl, theta, phi, 2Nh]
        
        # Diagonal propagation matrix
        Dexp = self.backend.diag_embed(exp)    # [wvl, theta, phi, 2Nh, 2Nh]
        
        # Zero matrices
        zero = self.backend.zeros_like(Dexp)
        
        # Build Smatrix
        Sm = build_block(self.backend, zero, Dexp, Dexp, zero)
        
        return Sm
        
    def Sm_layer(self, W_ext: Any, V_ext: Any) -> Any:
        """Calculates S-matrix for the layer surrounded by external medium symmetrically"""
        Sm_left = self.Sm_int_right(W_ext, V_ext)
        Sm_right = self.Sm_int_left(W_ext, V_ext)
        Sm_p = self.Sm_prop()
        
        Sm_layer = self._redheffer_star(Sm_right, self._redheffer_star(Sm_p, Sm_left))
        
        return Sm_layer
        
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
    
    def _redheffer_star(self, S2: Any, S1: Any) -> Any:
        """
        Calculates Redheffer star product between S1 and S2.
        S1 = (A B; C D), S2 = (E F; G H)
        
        S11 = A + B(I - ED)^{-1}EC 
        S12 = B(I - ED)^{-1}F
        S21 = G(I - DE)^{-1}C 
        S22 = H + G(I - DE)^{-1}DF    
        
        """
        A, B, C, D = split_block(S1)
        E, F, G, H = split_block(S2)
        
        # Calculates identity matrix
        batch_ndim = len(A.shape) - 2
        I = self.backend.eye(A.shape[-1])
        I = self.backend.reshape(I, [1]*batch_ndim + [A.shape[-1], A.shape[-1]])
        
        # Calculates I - ED and I - DE
        IED = I - self.backend.matmul(E, D)
        IDE = I - self.backend.matmul(D, E)
        
        # Calculate inverse
        inv_IED_EC = self._apply_inverse(IED, self.backend.matmul(E, C))
        inv_IED_F = self._apply_inverse(IED, F)
        inv_IDE_C = self._apply_inverse(IDE, C)
        inv_IDE_DF = self._apply_inverse(IDE, self.backend.matmul(D, F))

        # Calculate Smatrix components
        S11 = A + self.backend.matmul(B, inv_IED_EC)
        S12 = self.backend.matmul(B, inv_IED_F)
        S21 = self.backend.matmul(G, inv_IDE_C)
        S22 = H + self.backend.matmul(G, inv_IDE_DF)
        
        # Build block
        S = build_block(self.backend, S11, S12, S21, S22)

        return S

    def _Sm_int(self, WL: Any, VL: Any, WR: Any, VR: Any) -> Any:
        """Calculates S-matrix for an interface between Left and Right media"""
        block1 = build_block(self.backend, -WL, WR, -VL, -VR)
        block2 = build_block(self.backend, WL, -WR, -VL, -VR)
        
        Sm = self._apply_inverse(block1, block2)
        
        return Sm
        
    def _apply_inverse(self, matrix_A: Any, matrix_B: Any) -> Any:
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
        if layer.is_homogeneous and cfg.hsimplify:
            return LayerSolverHomogeneous
        return LayerSolverIsotropic

    raise NotImplementedError(
        f"No solver implemented for layer type: {layer}"
    )

"""Layer solver classes"""

class _BaseLayerSolver(LayerSolver, ABC):
    """
    Internal base class for all layer solvers.
    """
    
    @abstractmethod
    def _prepare_tvf(self):
        pass
    
    @abstractmethod
    def _correction_term(self):
        pass
    
    @abstractmethod
    def _Epsilon2(self):
        pass
    
    @abstractmethod
    def _Pmatrix(self):
        pass
    
    @abstractmethod
    def _Qmatrix(self):
        pass
    
    def _build_k_matrices(self):
        """
        Build the Kx and Ky diagonal matrices for the layer.
        """
        Kx, Ky = self.source.Kxy(self.n_inc, self.cfg.M, self.cfg.N, self.lattice)  # shape: (wvl, theta, phi, 2M+1, 2N+1)
        Kx_t, Ky_t = diagonal_K_matrices(self.backend, Kx, Ky, self.cfg.circ_truncation) # shape: (wvl, theta, phi, Nh, Nh)
        
        return Kx_t, Ky_t
    
    def _Omega2(self, P: Tuple[Any, Any, Any, Any], Q: Tuple[Any, Any, Any, Any]) -> Any:
        """
        Construct the Omega^2 matrix for the layer.
        Omega^2 = P @ Q
        
        Omega2_xx = P_xx * Q_xx + P_xy * Q_yx
        Omega2_xy = P_xx * Q_xy + P_xy * Q_yy   
        Omega2_yx = P_yx * Q_xx + P_yy * Q_yx
        Omega2_yy = P_yx * Q_xy + P_yy * Q_yy
        
        Parameters:
            P: Pxx, Pyy, Pxy, Pyx
            Q: Qxx, Qyy, Qxy, Qyx
            Each of shape [wvl, theta, phi, Nh, Nh]
        
        Returns
        -------
        Omega2_xx, Omega2_yy, Omega2_xy, Omega2_yx : Any
            Components of the Omega^2 matrix.
        """
        # Get P and Q matrices
        P_xx, P_yy, P_xy, P_yx = P  # shape: (wvl, theta, phi, Nh, Nh)
        Q_xx, Q_yy, Q_xy, Q_yx = Q  # shape: (wvl, theta, phi, Nh, Nh)
        
        # Build Omega^2 matrix
        Omega2_xx = self.backend.matmul(P_xx, Q_xx) + self.backend.matmul(P_xy, Q_yx)  # shape: (wvl, theta, phi, Nh, Nh)
        Omega2_xy = self.backend.matmul(P_xx, Q_xy) + self.backend.matmul(P_xy, Q_yy)  # shape: (wvl, theta, phi, Nh, Nh)
        Omega2_yx = self.backend.matmul(P_yx, Q_xx) + self.backend.matmul(P_yy, Q_yx)  # shape: (wvl, theta, phi, Nh, Nh)
        Omega2_yy = self.backend.matmul(P_yx, Q_xy) + self.backend.matmul(P_yy, Q_yy)  # shape: (wvl, theta, phi, Nh, Nh)
        
        return Omega2_xx, Omega2_yy, Omega2_xy, Omega2_yx
    
    def _mode_solver(self, A: Any) -> Any:
        """
        Select and return the appropriate mode solver function based on configuration.
        """
        if self.cfg.modes_solver == 'eig':
            return self.backend.eig(A)
        elif self.cfg.modes_solver == 'eigh':
            return self.backend.eigh(A)
        elif self.cfg.modes_solver == 'svd':
            raise NotImplementedError("SVD mode solver not implemented yet.")
        elif self.cfg.modes_solver == 'qr':
            raise NotImplementedError("QR mode solver not implemented yet.")
        else:
            raise ValueError(f"Unknown modes solver: {self.cfg.modes_solver}")
    
    def _eigensolver(self, Omega2: Tuple[Any, Any, Any, Any]):
        """
        Find eigenvalues and eigenvectors of the layer.
        
        Parameters:
            Omega2: Omega2_xx, Omega2_yy, Omega2_xy, Omega2_yx
            Each of shape: (wvl, theta, phi, Nh, Nh)

        Returns
        -------
        eigvals : Any
            Eigenvalues of the layer.
            Shape: (wvl, theta, phi, 2*Nh)
        eigvecs : Any
            Eigenvectors of the layer.
            Shape: (wvl, theta, phi, 2*Nh, 2*Nh)
        """
        Omega2_xx, Omega2_yy, Omega2_xy, Omega2_yx = Omega2  # shape: (wvl, theta, phi, Nh, Nh)
        
        # Create a block matrix for Omega^2
        Omega2 = build_block(self.backend, Omega2_xx, Omega2_xy, Omega2_yx, Omega2_yy)
        
        eigvals, eigvecs = self._mode_solver(Omega2)  # shape: (wvl, theta, phi, 2Nh), (wvl, theta, phi, 2Nh, 2Nh)
        
        return eigvals, eigvecs

    def _compute_V(self, Q: Tuple[Any, Any, Any, Any], W: Any, kz: Any):
        """
        Derive eigenvectors for magnetic field tangential components: V = -jQW/kz.
        
        Parameters:
            Q: Qxx, Qyy, Qxy, Qyx
            Each with shape (wvl, theta, phi, Nh, Nh)
            
            W: eigenvectors for electric field tangential components. 
            Shape (wvl, theta, phi, 2*Nh, 2*Nh)
            
            W: eigenvalues 
            Shape (wvl, theta, phi, 2*Nh)
            
        Return:
            V: eigenvectors for magnetic field tangential components. 
            Shape (wvl, theta, phi, 2*Nh, 2*Nh)
        """
        Qxx, Qyy, Qxy, Qyx = Q
        
        # Q blocks: [B, Nh, Nh]
        Q_block = build_block(self.backend, Qxx, Qxy, Qyx, Qyy)   # [B, 2Nh, 2Nh]
        
        # Compute QW
        QW = self.backend.matmul(Q_block, W)
        
        # Compute inv_kz
        inv_kz = 1.0 / (kz + 1e-12)
        
        V = -1j*self.backend.einsum('...ij,...j->...ij', QW, inv_kz) #[B, 2Nh, 2Nh]
        
        return V

class LayerSolverIsotropic(_BaseLayerSolver):
    """ Layer solver for isotropic, non-magnetic layers. """
        
    def _prepare_tvf(self) -> Tuple[Any, Any, Any, Any]:
        """
        Prepare the Tangent Vector Fields (TVF) for the layer.
        Pxx = |ty|^2 / (|tx|^2 + |ty|^2)
        Pyy = |tx|^2 / (|tx|^2 + |ty|^2)
        Pxy = (conj(tx) * ty) / (|tx|^2 + |ty|^2)
        Pyx = conj(Pxy)
        
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

        Pxx = ty2 / den                                    # projector components
        Pyy = tx2 / den
        Pxy = (self.backend.conj(tx) * ty) / den
        Pyx = self.backend.conj(Pxy)                       # enforce Hermitian
        
        # Do fft
        Pxx_fft = custom_fft(self.backend, Pxx, 2*self.cfg.M, 2*self.cfg.N)
        Pyy_fft = custom_fft(self.backend, Pyy, 2*self.cfg.M, 2*self.cfg.N)
        Pxy_fft = custom_fft(self.backend, Pxy, 2*self.cfg.M, 2*self.cfg.N)
        Pyx_fft = custom_fft(self.backend, Pyx, 2*self.cfg.M, 2*self.cfg.N)
        
        return Pxx_fft, Pyy_fft, Pxy_fft, Pyx_fft
    
    def _correction_term(self) -> Tuple[Any, Any, Any, Any]:
        """
        Compute the Li's Factorization correction term for the isotropic layer.
        Notation is from S4 paper by Liu et al. (2012).
        Delta = Epsilon - inv(Epsilon)^{-1}
        Delta_P = Delta * (Pxx, Pyy, Pxy, Pyx)
    
        Returns
        -------
        Delta_Pxx, Delta_Pyy, Delta_Pxy, Delta_Pyx : Any
            Correction terms for the projector components. Shape: (wvl, Nh, Nh), Nh=(2M+1)*(2N+1)
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
        Epsilon_inv_t_inv_Pxx = self._apply_inverse(Epsilon_inv_t, Pxx_t)  # shape: (wvl, Nh, Nh)
        Epsilon_inv_t_inv_Pyy = self._apply_inverse(Epsilon_inv_t, Pyy_t)  # shape: (wvl, Nh, Nh)
        Epsilon_inv_t_inv_Pxy = self._apply_inverse(Epsilon_inv_t, Pxy_t)  # shape: (wvl, Nh, Nh)
        Epsilon_inv_t_inv_Pyx = self._apply_inverse(Epsilon_inv_t, Pyx_t)  # shape: (wvl, Nh, Nh)
        
        # Compute correction term
        Delta_Pxx = self.backend.matmul(Epsilon_t, Pxx_t) - Epsilon_inv_t_inv_Pxx # shape: (wvl, Nh, Nh)
        Delta_Pyy = self.backend.matmul(Epsilon_t, Pyy_t) - Epsilon_inv_t_inv_Pyy  # shape: (wvl, Nh, Nh)
        Delta_Pxy = self.backend.matmul(Epsilon_t, Pxy_t) - Epsilon_inv_t_inv_Pxy  # shape: (wvl, Nh, Nh)
        Delta_Pyx = self.backend.matmul(Epsilon_t, Pyx_t) - Epsilon_inv_t_inv_Pyx  # shape: (wvl, Nh, Nh)
        
        return Delta_Pxx, Delta_Pyy, Delta_Pxy, Delta_Pyx
    
    def _Epsilon2(self) -> Any:
        """
        Construct the permittivity tensor Epsilon2 for the isotropic layer ([D]=Epsilon2[E]).
        Notation is from S4 paper by Liu et al. (2012).
        ** Important Note **: In the S4 paper, there is a typo in equations (48) - (51), where x and y
        components are swapped for the diagonal terms. The implementation here follows the correct formulation.
        
        Epsilon2_xx = Epsilon_t - Delta_Pxx
        Epsilon2_yy = Epsilon_t - Delta_Pyy
        Epsilon2_xy = Delta_Pxy
        Epsilon2_yx = Delta_Pyx
        
        Returns
        -------
        Epsilon2_xx, Epsilon2_yy, Epsilon2_xy, Epsilon2_yx : Any
            Components of the permittivity tensor Epsilon2. Shape: (wvl, Nh, Nh), Nh=(2M+1)*(2N+1)
        """
        # Get Fourier coefficients for epsilon_xy
        epsilon_mn = self.layer.epsilon_mn(
            2*self.cfg.M, 2*self.cfg.N, use_closed_form=self.cfg.closed_form
        )[:,0,0,...]  # shape: (wvl, 4M+1, 4N+1)
        
        # Get toeplitz convolution matrices
        Epsilon_t = toeplitz_2d(self.backend, epsilon_mn, self._index_map[0], self._index_map[1])  # shape: (wvl, Nh, Nh)
        
        # Get Li's correction terms
        if self.cfg.factorization not in ('None', None):
            Delta_Pxx, Delta_Pyy, Delta_Pxy, Delta_Pyx = self._correction_term()  # shape: (wvl, Nh, Nh)
        else:
            Delta_Pxx = self.backend.zeros_like(Epsilon_t)
            Delta_Pyy = self.backend.zeros_like(Epsilon_t)
            Delta_Pxy = self.backend.zeros_like(Epsilon_t)
            Delta_Pyx = self.backend.zeros_like(Epsilon_t)
        
        Epsilon2_xx = Epsilon_t - Delta_Pxx  # shape: (wvl, Nh, Nh) (Note the swap due to typo in original paper)
        Epsilon2_yy = Epsilon_t - Delta_Pyy  # shape: (wvl, Nh, Nh) (Note the swap due to typo in original paper)
        Epsilon2_xy = Delta_Pxy            # shape: (wvl, Nh, Nh)
        Epsilon2_yx = Delta_Pyx            # shape: (wvl, Nh, Nh)
        
        return Epsilon2_xx, Epsilon2_yy, Epsilon2_xy, Epsilon2_yx
    
    def _Pmatrix(self):
        """
        Construct the P matrix components.
        
        Pxx = -Kx * inv(Epsilon) * Ky
        Pxy = -I + Kx * inv(Epsilon) * Kx
        Pyx = I - Ky * inv(Epsilon) * Ky
        Pyy = Ky * inv(Epsilon) * Kx
        
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
        Epsilon_t_inv_Kx = self._apply_inverse(Epsilon_t, Kx_t)  # shape: (wvl, theta, phi, Nh, Nh)
        Epsilon_t_inv_Ky = self._apply_inverse(Epsilon_t, Ky_t)  # shape: (wvl, theta, phi, Nh, Nh)
        
        # Build identity matrix
        Nh = Epsilon_t.shape[-1]
        I = self.backend.eye(Nh)
        I = self.backend.reshape(I, (1, 1, 1, Nh, Nh))  # shape: (1, 1, 1, Nh, Nh)
        I = self.backend.expand(I, (wvl, theta, phi, Nh, Nh))  # shape: (wvl, theta, phi, Nh, Nh)
        
        # Build P matrix
        Pmatrix_xx = -self.backend.matmul(Kx_t, Epsilon_t_inv_Ky)  # shape: (wvl, theta, phi, Nh, Nh)
        Pmatrix_xy = - I + self.backend.matmul(Kx_t, Epsilon_t_inv_Kx)   # shape: (wvl, theta, phi, Nh, Nh)
        Pmatrix_yx = I - self.backend.matmul(Ky_t, Epsilon_t_inv_Ky)    # shape: (wvl, theta, phi, Nh, Nh)
        Pmatrix_yy = self.backend.matmul(Ky_t, Epsilon_t_inv_Kx)   # shape: (wvl, theta, phi, Nh, Nh)
        
        return Pmatrix_xx, Pmatrix_yy, Pmatrix_xy, Pmatrix_yx
    
    def _Qmatrix(self):
        """
        Construct the Q matrix components.
        
        Qxx = -Kx * Ky - Epsilon2_yx
        Qxy = Kx * Kx - Epsilon2_yy
        Qyx = Epsilon2_xx - Ky * Ky
        Qyy = Ky * Kx + Epsilon2_xy
        
        Returns
        -------
        Qmatrix_xx, Qmatrix_yy, Qmatrix_xy, Qmatrix_yx : Any
            Components of the Q matrix.
        """ 
        # Get Kx and Ky matrices
        Kx_t, Ky_t = self._build_k_matrices() # shape: (wvl, theta, phi, Nh, Nh)
        
        # Get Epsilon2 components
        Epsilon2_xx, Epsilon2_yy, Epsilon2_xy, Epsilon2_yx = self._Epsilon2()  # shape: (wvl, Nh, Nh)
        
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
        
class LayerSolverHomogeneous(LayerSolver):
    """ Layer solver for homogeneous, isotropic, non-magnetic layers. """
    
    def _prepare_kxy(self) -> Tuple[Any, Any]:
        """ 
        Compute kx and ky for all harmonics.
        
        Returns:
            kx, ky
            Each with shape (wvl, theta, phi, Nh)
        
        """
        Kx, Ky = self.source.Kxy(self.n_inc, self.cfg.M, self.cfg.N, self.lattice)
        kx, ky, _ = flatten_Kxy(self.backend, Kx, Ky) # [wvl, theta, phi, 
        if self.cfg.circ_truncation:
            mx, ny = build_harmonic_grid(self.backend, self.cfg.M, self.cfg.N)
            mask = elliptical_truncation_mask(mx, ny, self.cfg.M, self.cfg.N)
        else:
            mask = self.backend.astype(self.backend.ones((2*self.cfg.M+1)*(2*self.cfg.N+1)), self.backend.bool)
            
        kx = kx[...,mask]
        ky = ky[...,mask]
        
        return kx, ky
    
    def _compute_kz(self, kx: Any, ky: Any) -> Any:
        """ Compute kz analytically """
        
        # Compute initial quantities
        epsilon = self.layer.epsilon[:,0,0]
        
        # Broadcasr epsilon and compute (jkz)^2
        epsilon_b = self.backend.reshape(epsilon, [epsilon.shape[0], 1, 1, 1])
        epsilon_b = self.backend.expand(epsilon_b, kx.shape)
        kz2 = kx**2 + ky**2 - epsilon_b
        
        # Expand to eigvals size (match 2 polarizations)
        kz2 = self.backend.cat([kz2, kz2], dim=-1) # [wvl, theta, phi, 2*Nh]
        
        # Choose branch
        self._kz = kz_sign(self.backend, kz2)
        
        return self._kz
        
    def _compute_unit_vectors(self, kx: Any, ky: Any, kz: Any, tol: float = 1e-12) -> Tuple[Any, Any, Any, Any]:
        """Compute s- and p-polarization unit vectors"""
        
        # Compute initial quantities
        kt2 = kx*kx + ky*ky # [wvl, theta, phi, Nh]
        kt = self.backend.where(kt2 == 0, self.backend.zeros_like(kt2), self.backend.sqrt(kt2))  
        k2 = kt2 + self.backend.abs(kz)**2
        k = self.backend.where(k2 == 0, self.backend.zeros_like(k2), self.backend.sqrt(k2))  
        
        # Safe inverse
        inv_kt = 1.0 / self.backend.where(kt > tol, kt, self.backend.ones_like(kt))  # [wvl, theta, phi, Nh]
        inv_k = 1.0 / self.backend.where(k > tol, k, self.backend.ones_like(k))  # [wvl, theta, phi, Nh]
        
        # Unit vectors for s- and p-pol
        sx = -ky * inv_kt        # [wvl, theta, phi, Nh]                     
        sy =  kx * inv_kt                             
        px =  sy * kz * inv_k                             
        py =  -sx * kz * inv_k                                     

        # fix kt ~ 0 orders
        sx = self.backend.where(kt > tol, sx, self.backend.ones_like(sx))       # TE = (1,0)
        sy = self.backend.where(kt > tol, sy, self.backend.zeros_like(sy))      # [wvl, theta, phi, Nh]

        px = self.backend.where(kt > tol, px, self.backend.zeros_like(px))      # TM = (0,1)
        py = self.backend.where(kt > tol, py, self.backend.ones_like(py))       # [wvl, theta, phi, Nh]
        
        # Make Complex
        sx = self.backend.asarray(sx, complex=True)
        sy = self.backend.asarray(sy, complex=True)
        px = self.backend.asarray(px, complex=True)
        py = self.backend.asarray(py, complex=True)
        
        return sx, sy, px, py
    
    def _compute_W(self, sx: Any, sy: Any, px: Any, py: Any) -> Any:
        """Compute W analytically"""
        W_xx = self.backend.diag_embed(sx)   # [B, Nh, Nh]
        W_xy = self.backend.diag_embed(px)   # [B, Nh, Nh]
        W_yx = self.backend.diag_embed(sy)   # [B, Nh, Nh]
        W_yy = self.backend.diag_embed(py)   # [B, Nh, Nh]

        self._W   = build_block(self.backend, W_xx, W_xy, W_yx, W_yy)     # [B, 2Nh, 2Nh]
        
        return self._W
    
    def _compute_V(self, sx: Any, sy: Any, px: Any, py: Any, k: Any) -> Any:
        """Compute V analytically"""
        V_xx = 1j*self.backend.diag_embed(k*px)   # [B, Nh, Nh]
        V_xy = -1j*self.backend.diag_embed(k*sx)   # [B, Nh, Nh]
        V_yx = 1j*self.backend.diag_embed(k*py)   # [B, Nh, Nh]
        V_yy = -1j*self.backend.diag_embed(k*sy)   # [B, Nh, Nh]

        self._V   = build_block(self.backend, V_xx, V_xy, V_yx, V_yy)     # [B, 2Nh, 2Nh]
        
        return self._V
        
    def solve(self):
        #Calculate kx, ky, kz
        kx, ky = self._prepare_kxy()
        kz = self._compute_kz(kx, ky)[...,:kx.shape[-1]]
        k2 = kx**2 + ky**2 + self.backend.abs(kz)**2
        k = self.backend.where(k2 == 0, self.backend.zeros_like(k2), self.backend.sqrt(k2))  
        
        # Compute s and p
        sx, sy, px, py = self._compute_unit_vectors(kx, ky, kz)
        
        # Compute W
        self._compute_W(sx, sy, px, py)
        
        # Compute V
        self._compute_V(sx, sy, px, py, k)
        
        

