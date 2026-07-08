# tests/solver/test_eigsolver.py
# Tests for metarcwa.solver.layersolver.eigsolver.Eig: gradient verification
# (E4 — is the Lorentzian-broadened adjoint in Eig.backward correct?).
#
# Eig.forward returns eigenvalues/eigenvectors in arbitrary order with
# arbitrary per-column scale (any torch.linalg.eig call does), so gradcheck
# cannot be applied directly to the raw outputs of Eig.apply — the map
# X -> (eigval, eigvec) is not even a well-defined function in that sense.
# Instead:
#   1. Gradcheck the eigenvalue output alone (order/scale-independent once
#      the spectrum is well-separated and each eigenvalue is tracked by its
#      *value*, which gradcheck does via finite differences on that same
#      branch).
#   2. Gradcheck a *matrix function* f(M) = V @ diag(g(lam)) @ V^{-1} built
#      from (lam, V) — this is exactly invariant to eigenvector column
#      scaling (rescaling V by diag(c) leaves V @ diag(g) @ V^{-1}
#      unchanged), so it is a genuine, well-defined function of M and a
#      valid gradcheck target.
#   3. Compare the backward gradient of that same matrix function computed
#      via Eig (Lorentzian-regularized) against torch.linalg.eig (standard
#      adjoint) on a well-separated spectrum, where the O(broadening_parameter)
#      correction should be negligible.

import torch
from torch.testing import assert_close

from metarcwa.solver.layersolver.eigsolver import Eig


def _well_separated_matrix(n: int = 4, seed: int = 0) -> torch.Tensor:
    """Random complex128 matrix with an artificially separated spectrum
    (diagonal shift by 4*k), so eigenvalue gaps are large. Off-diagonal
    randomness keeps eigenvectors non-trivial (not axis-aligned)."""
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, n, dtype=torch.complex128, generator=g)
    x = x + 4.0 * torch.diag(torch.arange(n, dtype=torch.float64).to(torch.complex128))
    return x


def _matrix_func(eig_fn, M: torch.Tensor) -> torch.Tensor:
    """f(M) = V @ diag(sin(lam)) @ V^{-1} -- gauge-invariant to eigenvector
    column scale, and a smooth (holomorphic) function of the eigenvalues."""
    lam, V = eig_fn(M)
    return V @ torch.diag_embed(torch.sin(lam)) @ torch.linalg.inv(V)


class TestEigGradcheck:

    def test_eigval_gradcheck(self):
        """Eigenvalue output alone, well-separated spectrum: gradcheck the
        eigenvalue branch of Eig against finite differences."""
        x = _well_separated_matrix().requires_grad_(True)
        assert torch.autograd.gradcheck(
            lambda m: Eig.apply(m)[0], (x,), check_forward_ad=False
        )

    def test_matrix_func_gradcheck(self):
        """Gauge-invariant matrix function built from (lam, V): gradcheck
        against finite differences -- exercises the eigenvector branch of
        Eig.backward without depending on arbitrary eigenvector phase/scale."""
        x = _well_separated_matrix(seed=1).requires_grad_(True)
        assert torch.autograd.gradcheck(
            lambda m: _matrix_func(Eig.apply, m), (x,), check_forward_ad=False
        )

    def test_backward_matches_torch_linalg_eig(self):
        """E4: compare Eig's Lorentzian-regularized adjoint against the
        standard torch.linalg.eig adjoint on the same gauge-invariant matrix
        function, for a well-separated spectrum where the Lorentzian
        correction (eps=1e-10) should be negligible."""
        x0 = _well_separated_matrix(seed=2)
        x_custom = x0.clone().requires_grad_(True)
        x_ref    = x0.clone().requires_grad_(True)

        loss_custom = _matrix_func(Eig.apply, x_custom).abs().sum()
        loss_ref    = _matrix_func(torch.linalg.eig, x_ref).abs().sum()

        loss_custom.backward()
        loss_ref.backward()

        assert_close(x_custom.grad, x_ref.grad, atol=1e-6, rtol=1e-6)
