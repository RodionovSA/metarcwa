# metarcwa/solver/layersolver/_modes.py
"""
_modes — shared branch-selection / inverse / grazing-warning helpers
======================================================================

Both mode solvers (:func:`homogeneous_modes` in ``homogeneous.py`` and
:func:`eigsolver` in ``eigsolver.py``) need the same three pieces of logic
on their modal exponent ``lam = 1j·kz``:

  1. propagating/evanescent branch-sign selection 
  2. the regularized diagonal inverse ``diag(1/lam)`` used to build ``V``
  3. an (opt-in) grazing-incidence warning

All functions use the exp(−j ω t) time convention.
"""

import torch
import warnings

from metarcwa.solver.blockmatrix import Block, Block2x2

#: Opt-in gate for the grazing-incidence warning in :func:`_warn_grazing`.
#: Default off: the check requires ``.any().item()``, a CUDA host↔device
#: sync, so it is validation to enable explicitly (e.g. at problem setup),
#: not something paid on every hot-path call.
WARN_GRAZING: bool = False


def _branch_select(lam: torch.Tensor, tol: float = 1e-4) -> torch.Tensor:
    """
    Select the physical branch of a modal exponent ``lam``.

    Time convention exp(−j ω t): forward-propagating fields vary as
    ``exp(+lam·k0·z)``, so a physical (non-growing) mode must have
    ``Re(lam) <= 0``. Classification therefore keys on ``Re(lam)`` — the
    decay rate, cleanly separated (``O(1)`` for evanescent vs ``~0`` for
    propagating) — not on ``Im(lam)``:

      - propagating modes (``|Re(lam)| <= tol·|lam|``): ``Im(lam) > 0``
      - evanescent  modes (``|Re(lam)| >  tol·|lam|``): ``Re(lam) < 0``

    Parameters
    ----------
    lam : torch.Tensor
        Unsigned (or arbitrarily-signed) modal exponent, e.g. from
        ``sqrt(lam_sq)``. Any shape.
    tol : float, optional
        Relative threshold (as a fraction of ``|lam|``) below which a mode
        is classified propagating-by-real-part vs evanescent. Scale-robust
        by construction (both sides scale with ``|lam|``), verified stable
        for ``tol`` from ``1e-6`` to ``1e-2`` across problem scales.
        Default ``1e-4``.

    Returns
    -------
    torch.Tensor
        ``lam`` with the sign of each entry corrected to the physical branch.
    """
    lam_abs = lam.abs().clamp_min(torch.finfo(lam.real.dtype).tiny)
    is_prop = lam.real.abs() <= tol * lam_abs
    sign    = torch.where(is_prop, torch.sign(lam.imag), -torch.sign(lam.real))
    sign    = torch.where(sign == 0, torch.ones_like(sign), sign)
    return lam * sign


def _lam_inv_block(lam: torch.Tensor, N: int, delta: float = 1e-30) -> Block2x2:
    """
    Build the regularized ``diag(1/lam)`` operator as a ``Block2x2``.

    Splits ``lam`` (shape ``[..., 2N]``) into its two ``N``-sized blocks and
    forms a block-diagonal ``Block2x2`` (left column = first block, right
    column = second block, off-diagonals zero).

    The inverse is Lorentzian-regularized:

        lam_inv = conj(lam) / (|lam|^2 + delta)

    which equals ``1/lam`` whenever ``|lam| >> sqrt(delta)`` and stays finite
    exactly at grazing incidence (``lam = 0``), where the plain ``1/lam``
    would be ``nan``/``inf``. This mirrors the Lorentzian-regularization
    style already used for the ``sqrt`` gradient in ``homogeneous.py`` and
    for ``Eig.backward``.

    Parameters
    ----------
    lam : torch.Tensor
        Modal exponents, shape ``[..., 2N]``.
    N : int
        Number of harmonics (half the size of the last dimension of ``lam``).
    delta : float, optional
        Lorentzian regularization scale. Default ``1e-30``.

    Returns
    -------
    Block2x2
        Block-diagonal operator with ``Block(DIAG, ...)`` entries on the
        diagonal and ``Block.zeros(...)`` off-diagonal.
    """
    lam_inv = lam.conj() / (lam.abs() ** 2 + delta)
    kw = dict(device=lam.device, dtype=lam.dtype)
    return Block2x2(
        Block(Block.DIAG, lam_inv[..., :N]),
        Block.zeros(**kw),
        Block.zeros(**kw),
        Block(Block.DIAG, lam_inv[..., N:]),
    )


def _regularize_eps(eps: torch.Tensor, reg: float) -> torch.Tensor:
    """
    Floor the imaginary part of a permittivity tensor to ``>= reg``.

    Regularizes the exact-grazing degeneracy (``kz² = kx²+ky²-eps == 0`` at
    normal incidence or, more subtly, at an exact critical angle for total
    internal reflection): a lossless real ``eps`` there both zeroes ``lam``
    and makes the ``Q`` operator (``homogeneous_Q``/``compute_Q0``)
    rank-deficient (``Kx²-eps`` and ``eps-Ky²`` both vanish), which turns the
    boundary S-matrix solve (``S_boundary`` / ``Block2x2.solve``) exactly
    singular — most visibly in ``float32``, where the cancellation underflows
    to exact zero even a fraction of a degree from grazing.

    Adding a tiny loss (``+j*reg``, consistent with the ``exp(-j*omega*t)``
    time convention) keeps ``kz²`` and the ``Q`` operator non-degenerate
    everywhere, at the cost of an ``O(reg)`` perturbation to the result.

    Flooring rather than unconditionally adding ``reg`` avoids double-counting
    loss already present in a physically lossy medium.

    Parameters
    ----------
    eps : torch.Tensor
        Permittivity, real or complex, any shape.
    reg : float
        Minimum imaginary part. ``reg <= 0`` is a no-op (returns ``eps``
        unchanged).

    Returns
    -------
    torch.Tensor
        Complex tensor with ``Im(eps) >= reg`` everywhere (dtype promoted to
        complex if ``eps`` was real); unchanged if ``reg <= 0``.
    """
    if reg <= 0:
        return eps
    if eps.is_complex():
        return torch.complex(eps.real, torch.clamp(eps.imag, min=reg))
    return torch.complex(eps, torch.full_like(eps, reg))


def _warn_grazing(lam: torch.Tensor, tol: float, source: str) -> None:
    """
    Emit a ``RuntimeWarning`` for near-grazing modes, if enabled.

    No-op unless module-level :data:`WARN_GRAZING` is ``True`` (default
    off — the check triggers a CUDA host↔device sync via ``.any().item()``,
    so it is opt-in validation, not a hot-path check).

    Parameters
    ----------
    lam : torch.Tensor
        Modal exponents to check.
    tol : float
        Modes with ``|lam| < tol`` are considered grazing.
    source : str
        Name of the calling function, used in the warning message.
    """
    if not WARN_GRAZING:
        return
    zero_mask = lam.abs() < tol
    if zero_mask.any():
        warnings.warn(
            f"{source}: {zero_mask.sum().item()} mode(s) have |lam| < {tol} "
            "(grazing incidence). Corresponding columns of V are regularized "
            "but accuracy degrades near this limit.",
            RuntimeWarning,
            stacklevel=3,
        )
