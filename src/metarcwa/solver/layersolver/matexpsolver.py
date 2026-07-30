# metarcwa/solver/layersolver/matexpsolver.py
"""
matexpsolver — patterned-layer solver via matrix exponential of the
first-order system
=====================================================================

Alternative to :func:`~metarcwa.solver.layersolver.eigsolver.eigsolver`: no
eigendecomposition (and so none of the ``Eig`` Lorentzian-broadening
machinery needed to keep gradients finite near degenerate eigenvalues, and
none of the GPU eigendecomposition bottleneck). Instead, the layer's transfer
matrix is formed directly from a matrix exponential of the coupled first-order
system, then converted to an S-matrix per slice. See ``docs/matrixexp.md``
for the full derivation; summary below.

Math
----
From ``docs/rcwa_core.md`` Eqs. 37-38, with psi = (s, u) the combined
transverse field vector and everything k0-normalized::

    d(psi)/dz = k0 * A * psi,   A = [[0, P], [Q, 0]]
    psi(d) = T @ psi(0),        T = expm(A * k0 * d)

``P``/``Q`` are the same operators :func:`eigsolver` consumes — built once by
:func:`~metarcwa.solver.layersolver.isotropic.compute_isotropic` before the
``modesolver`` dispatch in ``LayerSolver._patterned``, TVF correction and all.
No new physics; this module only differs in *how* it propagates them.

``T`` is expressed in the *field* basis (s, u); :func:`transfer_to_smatrix`
changes it into some homogeneous reference medium's *mode-amplitude* basis
via that medium's gap matrix ``Phi = [[W, W], [V, -V]]`` and converts the
resulting transfer matrix ``M`` to the repo's S-matrix convention::

    S11 = -M22^-1 M21     S12 = M22^-1
    S21 = M11 - M12 M22^-1 M21     S22 = M12 M22^-1

Each slice is referenced to a per-layer "gap medium" (``Config.matexp_gap``;
see :func:`TransferOperator.smatrix`) rather than plain vacuum.

Stability: slicing
-------------------
``T``'s entries span ``exp(+-|lam|*k0*d)`` -- exactly what
``docs/smatrix.md`` cites as the reason to prefer S- over T-matrices (the
large entries overflow and swamp the small ones, destroying the relative
precision an S-matrix would otherwise keep O(1)). The fix used here keeps the
T-matrix-then-convert route while avoiding the instability: slice the layer
into ``n`` identical thin sub-layers, exponentiate + convert *one* slice, and
recombine via ``n - 1`` Redheffer star products (:func:`star_power` does this
in O(log n) via repeated squaring, since every slice is identical). Each
slice is a true gap-medium-embedded S-matrix, so the recombination is exact,
not an approximation -- only the per-slice exponent shrinks.

:func:`slice_count` estimates ``n`` from a cheap upper bound on the modal
exponent (``max|lam| ~ sqrt(max(kx^2+ky^2) + max|eps|)``, no eigenvalues
needed) and ``Config.matexp_max_exponent``; :class:`Config` also allows an
explicit override (``matexp_slices``) or disabling slicing entirely
(``matexp_slicing=False``).
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass

import torch

from metarcwa._dtypes import _REAL_TO_COMPLEX
from metarcwa.solver.blockmatrix import Block2x2
from metarcwa.solver.config import Config
from metarcwa.solver.layersolver.operator import Background
from metarcwa.solver.smatrix import S_boundary

#: Default per-slice exponent budget ``max(k0*d*|lam|)``, keyed by the
#: *complex* dtype the layer's operators are computed in. Chosen so retained
#: relative accuracy is ``~ machine_eps * exp(2*budget)``: ~2e-9 for
#: complex128 at budget=8.0, ~5e-5 for complex64 at budget=3.0.
_DEFAULT_MAX_EXPONENT = {
    torch.complex64:  3.0,
    torch.complex128: 8.0,
}


def system_matrix(P: Block2x2, Q: Block2x2) -> Block2x2:
    """Assemble the first-order system matrix ``A = [[0, P], [Q, 0]]``.

    ``P``/``Q`` are ``Block2x2`` of ``Nh x Nh`` blocks (the ``compute_isotropic``
    output), so ``A`` is a depth-2 ``Block2x2`` — ``A.to_dense(Nh)`` gives the
    full ``[..., 4Nh, 4Nh]`` system matrix.
    """
    Z = P.zeros_like()
    return Block2x2(Z, P, Q, Z)


def transfer_matrix(A: Block2x2, Nh: int, k0: torch.Tensor, d: torch.Tensor) -> Block2x2:
    """Compute ``expm(A * k0 * d)``, re-embedded as a ``Block2x2`` matching
    ``A``'s nesting.

    Densifies (matrix_exp has no structured/batched-sparse form), so this is
    the one point where the ``4Nh x 4Nh`` matrix is fully materialized.

    Parameters
    ----------
    A : Block2x2
        First-order system matrix, from :func:`system_matrix`.
    Nh : int
        Harmonic count (leaf block size of ``A``).
    k0 : torch.Tensor
        Free-space wavenumber ``2*pi/wvl``.
    d : torch.Tensor
        Propagation distance (same units as ``1/k0``).
    """
    A_dense = A.to_dense(Nh)                                  # [..., 4Nh, 4Nh]
    kd = torch.as_tensor(k0) * torch.as_tensor(d)
    batch_ndim = A_dense.ndim - 2
    if kd.ndim < batch_ndim:
        kd = kd.reshape(*kd.shape, *([1] * (batch_ndim - kd.ndim)))
    T_dense = torch.linalg.matrix_exp(A_dense * kd[..., None, None])
    return Block2x2.from_dense(T_dense, A)


def transfer_to_smatrix(T: Block2x2, background: Background) -> Block2x2:
    """Convert a field-basis transfer matrix ``T`` to an S-matrix, referenced
    to ``background`` -- any homogeneous medium, not necessarily vacuum (see
    ``TransferOperator.smatrix``, which passes this a per-layer *gap medium*
    rather than the stack's shared vacuum background).

    Derived the same way :func:`~metarcwa.solver.smatrix.S_boundary` is: a
    single linear solve for the outgoing amplitudes in terms of the incoming
    ones, never materializing a monolithic ``Phi^-1``.

    Both faces sit in the same reference medium, so the field vector at
    each face is ``psi = Phi @ (c+, c-)`` with that medium's gap matrix
    ``Phi = [[W, W], [V, -V]]`` (``W`` is always the identity for an
    isotropic homogeneous medium, see ``homogeneous_modes``). Substituting
    into ``psi(d) = T psi(0)`` with ``T`` partitioned to match the ``(s, u)``
    structure of the first-order system (``T.a/.b/.c/.d`` = ``T11/T12/T21/T22``)
    and grouping outgoing (``c_L-``, ``c_R+``) vs. incoming (``c_L+``, ``c_R-``)
    amplitudes gives exactly the boundary-matching linear system
    ``left @ (c_R+, c_L-) = right @ (c_L+, c_R-)`` that :func:`S_boundary`
    solves — just built from ``T`` instead of plain continuity. One
    ``Block2x2.solve`` (never ``.inv()``, and only one solve, not two) gives
    the whole S-matrix.

    Why this matters over the naive ``Phi^-1 T Phi`` route: at reduced
    precision (``dtype=torch.float32``), forming ``Phi^-1`` explicitly is
    ill-conditioned whenever the reference medium has evanescent harmonics
    with widely differing magnitudes (common at high truncation / short
    wavelength) — this was caught by a real-structure regression comparing
    against ``modesolver="eig"`` (`docs/matrixexp.md`), where the naive route
    gave a physically invalid `T > 1` that grew *worse*, not better, with
    more slices — the signature of compounding conditioning error, not
    under-slicing. This route shares `S_boundary`'s conditioning instead.

    Sanity check: for a layer matching the reference medium exactly, this
    reduces exactly to :func:`~metarcwa.solver.smatrix.S_prop`.
    """
    W0, V0 = background.W0, background.V0
    TW0_s, TW0_u = T.a @ W0, T.c @ W0     # T11 W0, T21 W0
    TV0_s, TV0_u = T.b @ V0, T.d @ V0     # T12 V0, T22 V0

    left = Block2x2(
        W0,             -(TW0_s - TV0_s),
        V0,             -(TW0_u - TV0_u),
    )
    right = Block2x2(
        TW0_s + TV0_s,  -W0,
        TW0_u + TV0_u,   V0,
    )
    M = left.solve(right)   # [c_R+; c_L-] = M [c_L+; c_R-]
    # S convention wants [c_L-; c_R+] = S [c_L+; c_R-] -- swap M's rows.
    return Block2x2(M.c, M.d, M.a, M.b)


#: Resonance-guard magnitude threshold for an intermediate star_power
#: S-matrix, keyed by the *complex* dtype of its leaves. A lossless slab's
#: S-matrix entries are physically bounded (~O(1), reaching a few tens even
#: near a genuine sharp resonance -- see docs/matrixexp.md). Anything past
#: this is either a genuine near-pole under-resolved by too few slices, or
#: (see slice_count's docstring) a fictitious gap-medium-embedded sub-slab
#: resonance hit by the squaring ladder; either way, complex64 has already
#: lost several percent of relative accuracy by this point, so warn while
#: complex128 still has comfortable headroom to be trustworthy.
_RESONANCE_GUARD_MAGNITUDE = {
    torch.complex64:  50.0,
    torch.complex128: 1e4,
}


def _leaf_dtype(entry) -> torch.dtype:
    """Dtype of a Block2x2/Block tree's first leaf (descends via `.a`)."""
    node = entry
    while hasattr(node, "a"):
        node = node.a
    return node.data.dtype


def star_power(S: Block2x2, n: int) -> Block2x2:
    """Compose ``n`` copies of the identical S-matrix ``S`` via the Redheffer
    star product, using repeated squaring (``O(log n)`` star products instead
    of ``n - 1``).

    Does *not* seed from ``Block2x2.star_identity()`` — that returns SCALAR
    ``Block`` leaves, which cannot compose with ``S``'s (depth-2, DENSE-leaf)
    structure. Instead accumulates from the first set bit of ``n``.

    Also checks each squaring/accumulation step's largest-magnitude entry
    against :data:`_RESONANCE_GUARD_MAGNITUDE` and warns once if exceeded —
    a heuristic that catches gross under-slicing but not every case (see
    ``docs/matrixexp.md`` "Accuracy and conditioning").

    Parameters
    ----------
    S : Block2x2
        A single slice's S-matrix.
    n : int
        Number of identical slices to compose, ``n >= 1``.
    """
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")

    threshold = _RESONANCE_GUARD_MAGNITUDE.get(
        _leaf_dtype(S), _RESONANCE_GUARD_MAGNITUDE[torch.complex64],
    )
    warned = False

    def _check(block: Block2x2, label: str) -> None:
        nonlocal warned
        if warned:
            return
        mag = float(block.to_dense().detach().abs().amax())
        if mag > threshold:
            warned = True
            warnings.warn(
                f"star_power: intermediate S-matrix magnitude {mag:.3g} at "
                f"{label} exceeds the resonance guard ({threshold:.3g} for "
                f"{_leaf_dtype(S)}). This usually means a fictitious "
                "gap-medium-embedded matexp sub-slab hit a spurious "
                "near-pole (see Config.matexp_gap), or the layer is "
                "under-sliced for the accuracy budget; either way, accuracy "
                "at this wavelength/geometry is suspect. Try a different "
                "matexp_slices or matexp_gap, or dtype=torch.float64 (see "
                "docs/matrixexp.md 'Accuracy and conditioning').",
                RuntimeWarning,
                stacklevel=3,
            )

    result = None
    base = S
    while n:
        if n & 1:
            result = base if result is None else result.star(base)
            _check(result, "an accumulation step")
        n >>= 1
        if n:
            base = base.star(base)
            _check(base, "a squaring step")
    return result


def _resolve_max_exponent(config: Config) -> float:
    if config.matexp_max_exponent is not None:
        return config.matexp_max_exponent
    cdtype = _REAL_TO_COMPLEX.get(config.dtype, torch.complex64)
    return _DEFAULT_MAX_EXPONENT.get(cdtype, _DEFAULT_MAX_EXPONENT[torch.complex64])


def slice_count(lam_bound: torch.Tensor, k0: torch.Tensor, d: torch.Tensor,
                 config: Config) -> int:
    """Estimate the number of identical thin sub-layers to slice a patterned
    layer into before exponentiating (see module docstring).

    A detached scalar: ``n`` controls a Python loop count in
    :func:`star_power`, so it cannot vary per batch element and must not
    enter autograd (gradients through the S-matrix itself are exact for any
    fixed ``n`` — only the *choice* of ``n`` is non-differentiable, which is
    correct, not approximate).

    Resolution order: ``config.matexp_slicing=False`` -> ``1`` (warns if the
    exponent budget is exceeded); explicit ``config.matexp_slices`` -> that
    value; otherwise an automatic estimate from ``lam_bound``, nudged off the
    dyadic ladder (see below), clamped to ``config.matexp_max_slices`` (warns
    if clamped).

    Off-the-ladder nudge: an even ``n`` makes :func:`star_power`'s squaring
    ladder revisit exact dyadic fractions of ``d``, which can coincide with
    a sub-slab resonance and cost several percent relative error in
    ``complex64`` (see ``docs/matrixexp.md`` "Accuracy and conditioning").
    The automatic estimate is bumped to the next odd integer to avoid this.

    Parameters
    ----------
    lam_bound : torch.Tensor
        Cheap upper bound on the modal exponent magnitude for this layer
        (see ``TransferOperator.lam_bound``). Any shape; reduced via
        ``.abs().max()``.
    k0 : torch.Tensor
        Free-space wavenumber ``2*pi/wvl``.
    d : torch.Tensor
        Layer thickness.
    config : Config
        Solver configuration (``matexp_slicing``, ``matexp_slices``,
        ``matexp_max_slices``, ``matexp_max_exponent``).
    """
    budget = _resolve_max_exponent(config)
    exponent = (k0.detach() * torch.as_tensor(d).detach().abs()
                * lam_bound.detach().abs())
    max_exponent = float(exponent.max()) if exponent.numel() else 0.0

    if not config.matexp_slicing:
        if max_exponent > budget:
            warnings.warn(
                f"matexp_slicing is disabled but the estimated per-layer "
                f"exponent ({max_exponent:.3g}) exceeds the accuracy budget "
                f"({budget:.3g}); the unsliced matrix exponential may lose "
                "significant relative precision (or overflow) for this "
                "layer.",
                RuntimeWarning,
                stacklevel=2,
            )
        return 1

    if config.matexp_slices is not None:
        return max(1, int(config.matexp_slices))

    n_est = max(1, math.ceil(max_exponent / budget))
    if n_est % 2 == 0:
        n_est += 1   # off the dyadic squaring ladder -- see docstring above
    if n_est > config.matexp_max_slices:
        warnings.warn(
            f"matexp auto slice-count estimate ({n_est}) exceeds "
            f"matexp_max_slices={config.matexp_max_slices}; clamping, which "
            f"may leave the per-slice exponent above the accuracy budget "
            f"({budget:.3g}). Raise matexp_max_slices or set matexp_slices "
            "explicitly to avoid this.",
            RuntimeWarning,
            stacklevel=2,
        )
        return config.matexp_max_slices
    return n_est


@dataclass(frozen=True)
class TransferOperator:
    """
    Precomputed first-order-system operators of one patterned stack element,
    for the matrix-exponential solver (``Config.modesolver == "matexp"``).

    Counterpart to :class:`~metarcwa.solver.layersolver.operator.ModalOperator`
    for the eigenmode path: same ``LayerOperator`` contract
    (``smatrix``/``transfer``), but carries ``P``/``Q`` instead of
    ``(lam, W, V)`` and exponentiates instead of diagonalizing.

    Cost model is inverted relative to ``ModalOperator``: cheap to construct
    (no eigendecomposition — just ``compute_isotropic``), and the expensive
    work (the sliced matrix exponential) happens in :meth:`smatrix`, since it
    depends on ``thickness`` and must stay responsive to
    ``dataclasses.replace(op, thickness=...)``.

    Attributes
    ----------
    P, Q : Block2x2
        First-order system operators from :func:`compute_isotropic`
        (TVF-corrected if configured).
    Nh : int
        Harmonic count (leaf block size of ``P``/``Q``).
    lam_bound : torch.Tensor
        Detached, cheap upper bound on the modal exponent magnitude for this
        layer (``~ sqrt(max(kx^2+ky^2) + max|eps_grid|)``), used by
        :func:`slice_count` to size the automatic slicing without an
        eigendecomposition.
    config : Config
        Solver configuration (slicing controls).
    gap_background : Background
        The homogeneous "gap medium" every slice's S-matrix is referenced to
        (``Config.matexp_gap``), distinct from the stack background passed
        into :meth:`smatrix`. Required, with no default.
    thickness : torch.Tensor or None
        Layer thickness. ``None`` is not a valid state for this operator (a
        semi-infinite medium always goes through
        :class:`~metarcwa.solver.layersolver.operator.ModalOperator` /
        ``homogeneous_modes`` — see ``LayerSolver._medium``).
    """
    P: Block2x2
    Q: Block2x2
    Nh: int
    lam_bound: torch.Tensor
    config: Config
    gap_background: Background
    thickness: torch.Tensor | None = None

    def smatrix(self, background: Background, left: bool = True) -> Block2x2:
        """Assemble the S-matrix: slice, exponentiate each slice, convert
        each to an S-matrix against :attr:`gap_background`, recombine via
        :func:`star_power`, then sandwich with two boundary transitions back
        to ``background`` (mirroring :func:`~metarcwa.solver.smatrix.S_layer`
        on the "eig" path). Exact for any choice of gap medium; see
        ``docs/matrixexp.md`` "Accuracy and conditioning".

        ``left`` is accepted for interface parity with
        :meth:`~metarcwa.solver.layersolver.operator.ModalOperator.smatrix`
        but has no effect — a ``TransferOperator`` always represents a
        finite patterned layer, never a semi-infinite boundary.
        """
        if self.thickness is None:
            raise ValueError(
                "TransferOperator requires a finite thickness; semi-infinite "
                "media are solved via ModalOperator (LayerSolver._medium)."
            )
        k0 = 2 * torch.pi / torch.as_tensor(background.wvl)
        n = slice_count(self.lam_bound, k0, self.thickness, self.config)
        A = system_matrix(self.P, self.Q)
        T_slice = transfer_matrix(A, self.Nh, k0, self.thickness / n)
        S_slice = transfer_to_smatrix(T_slice, self.gap_background)
        S_gap = star_power(S_slice, n)

        S_in = S_boundary(background.W0, background.V0,
                           self.gap_background.W0, self.gap_background.V0)
        S_out = Block2x2(S_in.d, S_in.c, S_in.b, S_in.a)   # mirror trick, see docstring
        return S_in.star(S_gap).star(S_out)

    def transfer(self, background: Background, z: torch.Tensor) -> Block2x2:
        """Propagator psi(0) -> psi(z), directly via ``expm(A * k0 * z)`` —
        no eigendecomposition.

        Unlike :meth:`smatrix`, this is *not* sliced: it is intended for
        field evaluation at a single depth, not for cascading many layers,
        and large ``z`` faces the same conditioning limits described in the
        module docstring. Callers evaluating fields across a thick layer
        should sample ``z`` in bounded increments (mirroring the internal
        slicing in :meth:`smatrix`).
        """
        k0 = 2 * torch.pi / torch.as_tensor(background.wvl)
        A = system_matrix(self.P, self.Q)
        return transfer_matrix(A, self.Nh, k0, z)
