# metarcwa/solver/layersolver/eigsolver.py
"""
eigsolver — eigenmode decomposition for patterned RCWA layers
=============================================================

Solves the generalised eigenvalue problem

    Ω² = P · Q,   Ω² · w_i = λ² · w_i

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
from metarcwa.solver.layersolver._modes import _branch_select, _lam_inv_block, _warn_grazing


def eigsolver(P: Block2x2, Q: Block2x2,
              stable_eig_grad: bool = True,
              tol: float = 1e-12) -> Tuple[torch.Tensor, Block2x2, Block2x2]:
    """
    Compute patterned-layer modes via eigendecomposition of Ω² = P·Q.

    Solves the eigenvalue problem

        P·Q·w_i = λ²·w_i

    and returns modal exponents ``lam = 1j·kz``, the E-mode matrix ``W``
    (columns = eigenvectors), and the H-mode matrix ``V = Q·W·diag(1/lam)``.
    The result is fully compatible with :func:`S_layer` and
    :func:`homogeneous_modes` (same ``lam`` sign convention).

    Branch selection for lam = 1j·kz:
      - Propagating modes (|Im(lam)| > tol): Im(lam) > 0
      - Evanescent  modes (|Im(lam)| ≤ tol): Re(lam) < 0

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
        selection (|Im(lam)| < tol).  Default ``1e-12``.

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

    # lam = sqrt(lam_sq) with branch selection to match lam = 1j*kz convention
    lam = _branch_select(torch.sqrt(lam_sq), tol)        # [..., 2N]

    # Fix grazing angle (lam=0) problem (opt-in; see _modes.WARN_GRAZING)
    _warn_grazing(lam, tol, "eigsolver")

    # E-mode matrix: partition 2N×2N eigenvector matrix into four N×N blocks
    W = Block2x2(
        Block(Block.DENSE, W_dense[..., :N, :N]),    # top-left
        Block(Block.DENSE, W_dense[..., :N, N:]),    # top-right
        Block(Block.DENSE, W_dense[..., N:, :N]),    # bottom-left
        Block(Block.DENSE, W_dense[..., N:, N:]),    # bottom-right
    )

    # H-mode matrix: V = Q @ W @ diag(1/lam), Lorentzian-regularized (E6)
    lam_inv = _lam_inv_block(lam, N)
    V = Q @ W @ lam_inv

    return lam, W, V


class Eig(torch.autograd.Function):
    """
    Eigendecomposition with Lorentzian-broadened gradients.

    Standard ``torch.linalg.eig`` gradients involve a divided difference of
    eigenvalues in the denominator, ``1 / (λ_j − λ_i)``.  Near degenerate
    eigenvalues (λ_j ≈ λ_i) this blows up, causing NaN gradients during
    optimisation.

    ``backward`` regularizes it with a Lorentzian in two steps: it first
    forms ``F_ij = conj(s_ij) / (|s_ij|² + ε)`` with ``s_ij = λ_j − λ_i``,
    then applies ``conj(F)`` when weighting the eigenvector term — so the
    factor actually multiplying ``Xᴴ·grad_eigvec`` is
    ``s_ij / (|s_ij|² + ε)``, the Lorentzian-regularized reciprocal of
    ``s_ij`` (ε = ``broadening_parameter``). This introduces a small
    controlled error in the gradient but prevents numerical blow-up.
    Verified against finite differences and against ``torch.linalg.eig``'s
    own gradient on a well-separated spectrum
    (``tests/solver/test_eigsolver.py``).

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
        divided difference replaced by a Lorentzian. ``F = conj(s) / (|s|² + ε)``
        is formed first (``s_ij = λ_j − λ_i``), then ``conj(F)`` is applied
        when weighting the eigenvector term, so the net factor is
        ``s / (|s|² + ε)`` — see the class docstring.

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
