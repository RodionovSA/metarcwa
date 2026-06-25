# metarcwa/solver/layersolver/eigsolver.py
"""
eigsolver — eigenmode decomposition for patterned RCWA layers
=============================================================

Solves the generalised eigenvalue problem

    Ω² = P · Q,   Ω² · w_i = kz_i² · w_i

to extract the E-mode matrix W (eigenvectors) and H-mode matrix V = Q·W·diag(1/λ),
where λ = 1j·kz are the modal exponents in the convention shared with
:func:`homogeneous_modes`.

The public interface mirrors that of :func:`homogeneous_modes`:

  eigsolver(P, Q)   →   (lam, W, V)

Two public objects:

  eigsolver(P, Q, stable_eig_grad=True, tol=1e-12)   → (lam, W, V)
      Full patterned-layer mode solver.

  Eig                                                   autograd.Function
      Stable eigendecomposition with Lorentzian-broadened gradients.
      Inspired by TORCWA (github.com/kch3782/torcwa) but not reproduced
      verbatim — batching support and deprecation fixes were added.

All functions use the exp(−j ω t) time convention.
"""

import torch
from typing import Tuple

from metarcwa.solver.blockmatrix import Block, Block2x2


def eigsolver(P: Block2x2, Q: Block2x2,
              stable_eig_grad: bool = True,
              tol: float = 1e-12) -> Tuple[torch.Tensor, Block2x2, Block2x2]:
    """
    Compute patterned-layer modes via eigendecomposition of Ω² = P·Q.

    Solves the eigenvalue problem

        P·Q·w_i = kz_i²·w_i

    and returns modal exponents ``lam = 1j·kz``, the E-mode matrix ``W``
    (columns = eigenvectors), and the H-mode matrix ``V = Q·W·diag(1/lam)``.
    The result is fully compatible with :func:`S_layer` and
    :func:`homogeneous_modes` (same ``lam`` sign convention).

    Branch selection for kz:
      - Propagating modes (|Re(kz)| > tol): Re(kz) > 0
      - Evanescent  modes (|Re(kz)| ≤ tol): Im(kz) > 0

    Parameters
    ----------
    P : Block2x2
        P operator of the layer; shape ``(..., Nh, Nh)`` per block entry.
        Obtained from :func:`compute_isotropic` or similar.
    Q : Block2x2
        Q operator of the layer; same shape as P.
    stable_eig_grad : bool, optional
        If ``True`` (default), use :class:`Eig` with Lorentzian-broadened
        gradients for numerical stability near degenerate eigenvalues.
        Set to ``False`` to use ``torch.linalg.eig`` directly (faster but
        gradients can be NaN near degeneracies).
    tol : float, optional
        Threshold for classifying a mode as evanescent during branch
        selection (|Re(kz)| < tol).  Default ``1e-12``.

    Returns
    -------
    lam : torch.Tensor
        Modal exponents lam = 1j·kz. Shape ``(..., 2Nh)``.
    W : Block2x2
        E-mode matrix; each entry is a ``Block(DENSE, ...)`` of shape
        ``(..., Nh, Nh)``.
    V : Block2x2
        H-mode matrix Q·W·diag(1/lam); same entry shapes as W.

    Notes
    -----
    ``lam`` uses the same 1j·kz convention as :func:`homogeneous_modes` so
    the two solvers can be used interchangeably with :func:`S_layer`.
    """
    Omega2       = P @ Q
    Omega2_dense = Omega2.to_dense(P.a.shape[-1])   # [..., 2N, 2N]
    N            = Omega2_dense.shape[-1] // 2

    if stable_eig_grad:
        lam_sq, W_dense = Eig.apply(Omega2_dense)
    else:
        lam_sq, W_dense = torch.linalg.eig(Omega2_dense)

    # kz from eigenvalues with branch selection
    kz    = torch.sqrt(lam_sq)
    is_ev = kz.real.abs() < tol
    sign  = torch.where(is_ev, torch.sign(kz.imag), torch.sign(kz.real))
    sign  = torch.where(sign == 0, torch.ones_like(sign), sign)
    lam   = 1j * kz * sign                           # [..., 2N]

    # E-mode matrix: partition 2N×2N eigenvector matrix into four N×N blocks
    W = Block2x2(
        Block(Block.DENSE, W_dense[..., :N, :N]),    # top-left
        Block(Block.DENSE, W_dense[..., :N, N:]),    # top-right
        Block(Block.DENSE, W_dense[..., N:, :N]),    # bottom-left
        Block(Block.DENSE, W_dense[..., N:, N:]),    # bottom-right
    )

    # H-mode matrix: V = Q @ W @ diag(1/lam)
    kw      = dict(device=lam.device, dtype=lam.dtype)
    lam_inv = Block2x2(
        Block(Block.DIAG, 1.0 / lam[..., :N]),
        Block.zeros(**kw),
        Block.zeros(**kw),
        Block(Block.DIAG, 1.0 / lam[..., N:]),
    )
    V = Q @ W @ lam_inv

    return lam, W, V


class Eig(torch.autograd.Function):
    """
    Eigendecomposition with Lorentzian-broadened gradients.

    Standard ``torch.linalg.eig`` gradients involve differences of eigenvalues
    in the denominator:  F_ij = 1 / (λ_j − λ_i).  Near degenerate eigenvalues
    (λ_j ≈ λ_i) these blow up, causing NaN gradients during optimisation.

    This class replaces the singular 1/(λ_j − λ_i) terms with a Lorentzian
    regularisation::

        F_ij = conj(λ_j − λ_i) / (|λ_j − λ_i|² + ε)

    where ε = ``broadening_parameter``.  This introduces a small controlled
    error in the gradient but prevents numerical blow-up.

    .. note::
        Inspired by the eigendecomposition utility in TORCWA
        (github.com/kch3782/torcwa) but not reproduced verbatim.
        Batching support (``torch.diag_embed``, batched diagonal zeroing)
        and replacement of the deprecated ``torch.inverse`` with
        ``torch.linalg.inv`` were added.

    Class Attributes
    ----------------
    broadening_parameter : float or None
        ε for Lorentzian regularisation.  ``None`` falls back to the
        machine epsilon of the input dtype (may cause NaN near degeneracies).
        Default ``1e-10``.
    """

    broadening_parameter: float | None = 1e-10

    @staticmethod
    def forward(ctx, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute eigenvalues and eigenvectors; save via ``ctx.save_for_backward``.

        Parameters
        ----------
        x : torch.Tensor
            Square (batched) matrix, shape ``(..., n, n)``.

        Returns
        -------
        eigval : torch.Tensor
            Eigenvalues, shape ``(..., n)``.
        eigvec : torch.Tensor
            Eigenvectors (columns), shape ``(..., n, n)``.
        """
        eigval, eigvec = torch.linalg.eig(x)
        ctx.save_for_backward(eigval, eigvec)
        ctx.is_real_input = not torch.is_complex(x)
        return eigval, eigvec

    @staticmethod
    def backward(ctx,
                 grad_eigval: torch.Tensor,
                 grad_eigvec: torch.Tensor) -> torch.Tensor:
        """
        Lorentzian-regularised gradient of the eigendecomposition.

        Uses the analytic formula for d(eigvec)/dX with the singular
        denominator replaced by a Lorentzian:

            F_ij = conj(s_ij) / (|s_ij|² + ε),   s_ij = λ_j − λ_i

        Parameters
        ----------
        grad_eigval : torch.Tensor
            Upstream gradient w.r.t. eigenvalues, shape ``(..., n)``.
        grad_eigvec : torch.Tensor
            Upstream gradient w.r.t. eigenvectors, shape ``(..., n, n)``.

        Returns
        -------
        torch.Tensor
            Gradient w.r.t. the input matrix X, shape ``(..., n, n)``.
        """
        eigval, eigvec = ctx.saved_tensors
        eigval = eigval.to(grad_eigval.dtype)
        eigvec = eigvec.to(grad_eigvec.dtype)

        grad_eigval = torch.diag_embed(grad_eigval)          # [..., n, n]
        s = eigval.unsqueeze(-2) - eigval.unsqueeze(-1)      # [..., n, n]

        # Lorentzian broadening
        eps = Eig.broadening_parameter
        if eps is not None:
            F = torch.conj(s) / (torch.abs(s) ** 2 + eps)
        elif s.dtype == torch.complex64:
            F = torch.conj(s) / (torch.abs(s) ** 2 + 1.4e-45)
        else:
            F = torch.conj(s) / (torch.abs(s) ** 2 + 4.9e-324)

        F.diagonal(dim1=-2, dim2=-1).zero_()                 # remove self-terms

        XH  = torch.transpose(torch.conj(eigvec), -2, -1)
        tmp = torch.conj(F) * torch.matmul(XH, grad_eigvec)

        grad = torch.matmul(
            torch.matmul(torch.linalg.inv(XH), grad_eigval + tmp), XH
        )
        if ctx.is_real_input:
            grad = torch.real(grad)

        return grad
