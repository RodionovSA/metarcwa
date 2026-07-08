# metarcwa/solver/layersolver/_modes.py
"""
_modes — shared branch-selection / inverse / grazing-warning helpers
======================================================================

Both mode solvers (:func:`homogeneous_modes` in ``homogeneous.py`` and
:func:`eigsolver` in ``eigsolver.py``) need the same three pieces of logic
on their modal exponent ``lam = 1j·kz``:

  1. propagating/evanescent branch-sign selection (E5 rule)
  2. the regularized diagonal inverse ``diag(1/lam)`` used to build ``V``
  3. an (opt-in) grazing-incidence warning

These were historically duplicated verbatim in both solvers — the
duplication itself was a recurring source of bugs (ANALYSIS.md A3). This
module is the single place that logic lives; both solvers import from here.

All functions use the exp(−j ω t) time convention.
"""

import torch
import warnings

from metarcwa.solver.blockmatrix import Block, Block2x2

#: Opt-in gate for the grazing-incidence warning in :func:`_warn_grazing`.
#: Default off: the check requires ``.any().item()``, a CUDA host↔device
#: sync, so it is validation to enable explicitly (e.g. at problem setup),
#: not something paid on every hot-path call (ANALYSIS.md B3).
WARN_GRAZING: bool = False


def _branch_select(lam: torch.Tensor, tol: float = 1e-12) -> torch.Tensor:
    """
    Select the physical branch of a modal exponent ``lam`` (E5 rule).

    Time convention exp(−j ω t): forward-propagating fields vary as
    ``exp(+lam·k0·z)``, so the sign of each ``lam`` is chosen so that:

      - propagating modes (``|Im(lam)| > tol``): ``Im(lam) > 0``
      - evanescent  modes (``|Im(lam)| <= tol``): ``Re(lam) < 0``

    Parameters
    ----------
    lam : torch.Tensor
        Unsigned (or arbitrarily-signed) modal exponent, e.g. from
        ``sqrt(lam_sq)``. Any shape.
    tol : float, optional
        Threshold below which a mode is classified evanescent-by-imaginary-
        part vs propagating. Default ``1e-12``.

    Returns
    -------
    torch.Tensor
        ``lam`` with the sign of each entry corrected to the physical branch.
    """
    is_ev = lam.imag.abs() < tol
    sign  = torch.where(is_ev, -torch.sign(lam.real), torch.sign(lam.imag))
    sign  = torch.where(sign == 0, torch.ones_like(sign), sign)
    return lam * sign


def _lam_inv_block(lam: torch.Tensor, N: int, delta: float = 1e-30) -> Block2x2:
    """
    Build the regularized ``diag(1/lam)`` operator as a ``Block2x2``.

    Splits ``lam`` (shape ``[..., 2N]``) into its two ``N``-sized blocks and
    forms a block-diagonal ``Block2x2`` (left column = first block, right
    column = second block, off-diagonals zero).

    The inverse is Lorentzian-regularized (E6)::

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


def _warn_grazing(lam: torch.Tensor, tol: float, source: str) -> None:
    """
    Emit a ``RuntimeWarning`` for near-grazing modes, if enabled.

    No-op unless module-level :data:`WARN_GRAZING` is ``True`` (default
    off — the check triggers a CUDA host↔device sync via ``.any().item()``,
    so it is opt-in validation, not a hot-path check; ANALYSIS.md B3).

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
