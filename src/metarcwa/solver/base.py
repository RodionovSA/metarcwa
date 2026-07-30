# metarcwa/solver/base.py
"""
Solver — top-level RCWA solver
==============================

Functional core (:func:`build_layersolver`, :func:`prepare`, :func:`run`,
:func:`reprepare`) plus a thin :class:`Solver` class wrapper for convenience.

Binds a :class:`~metarcwa.model.base.Model` and a :class:`Config` into a
ready-to-run simulation. :func:`prepare` does the expensive work: the model
is moved to the requested device/dtype, harmonics and the optional TVF are
pre-computed (:func:`build_layersolver`), and every stack element's modal
eigenproblem is solved once via :meth:`LayerSolver.prepare`. The result is a
frozen :class:`PreparedStack` snapshot. :func:`run` is then genuinely
cheap: pure Redheffer star-product composition of the precomputed
:class:`~metarcwa.solver.layersolver.operator.LayerOperator` objects, with no
TVF, convolution, or eigendecomposition work. :func:`run` keeps only the
reflection (``S11``) and transmission (``S21``) blocks a single-side (left)
excitation needs, in a :class:`ModalSolution` that also retains the
:class:`PreparedStack`, so downstream observables (and, eventually, field
reconstruction via :class:`FieldSolution`) can reach every layer's modes
without re-solving.

There is a single solver for both regular (batched wavelength/angle sweep)
runs and inverse-design optimization steps — they differ only in autograd
context (``torch.no_grad()`` vs a retained graph), not in structure.

Because the pattern/geometry is resolved once in :func:`prepare`
(``model.spec()``), an inverse-design loop must re-resolve it whenever the
pattern changes. Two cases:

- **Whole-stack geometry change** (source, lattice, or more than one layer's
  pattern changes): call :func:`prepare` again — everything is rebuilt.
- **Single-layer (or few-layer) geometry change at fixed source/lattice**
  (e.g. optimizing one metasurface layer's pattern while the rest of the
  stack is held fixed): call :func:`reprepare` instead. It reuses the
  :class:`LayerSolver` context (``kx``/``ky``/``W0``/``V0``/TVF — all
  source-and-lattice-derived) and every *other* layer's already-prepared
  :class:`LayerOperator` unchanged, re-running :meth:`LayerSolver.prepare`
  only for the elements named in ``layer_indices``. This is the efficiency
  lever for single-layer inverse design: one eigendecomposition per step
  instead of one per stack element.

Thickness gradients/updates are a separate, cheaper case still: operators
hold a *reference* to each layer's thickness tensor and read it inside
:func:`run`, so autograd through thickness (and in-place ``nn.Parameter``
thickness updates) work without rebuilding or re-preparing anything.
"""

from dataclasses import dataclass, replace

import torch

from metarcwa.model.base import Model, ModelSpec
from metarcwa.solver.layersolver.base import LayerSolver
from metarcwa.solver.layersolver.operator import LayerOperator
from metarcwa.solver.tvf import TVF
from metarcwa.solver.config import Config
from metarcwa.solver.harmonics import compute_kxy, harmonic_index_map
from metarcwa.solver.blockmatrix import Block2x2, Entry


def build_layersolver(model_spec: ModelSpec, config: Config) -> LayerSolver:
    """Build the harmonic + TVF context for a resolved model snapshot.

    This is the "context" tier: harmonic index map, ``kx``/``ky``, and the
    optional TVF are all derived from the source/lattice (``model_spec``)
    and ``config`` alone — not from any individual layer's pattern. The
    returned :class:`LayerSolver` also caches the vacuum background modes
    (``W0``/``V0``) at construction. It stays valid across
    :func:`reprepare` calls as long as source and lattice are unchanged.

    Parameters
    ----------
    model_spec : ModelSpec
        Resolved model snapshot (see :meth:`Model.spec`).
    config : Config
        Solver hyperparameters.

    Returns
    -------
    LayerSolver
        Pre-initialised layer-level modal solver / S-matrix assembler.
    """
    m_flat, n_flat = harmonic_index_map(
        config.m, config.n, config.truncation == "circular", config.device
    )
    
    k0 = 2 * torch.pi / model_spec.wavelength.reshape(-1)   # [N_wvl]
    kx, ky = compute_kxy(
        model_spec.kx0, model_spec.ky0,
        model_spec.a1,  model_spec.a2,
        m_flat, n_flat, k0=k0,
    )

    if config.factorization is not None:
        f = config.factorization
        tvf = TVF(model_spec.a1, model_spec.a2,
                  config.m, config.n,
                  f.method, f.optimizer,
                  f.alpha, f.beta, f.gamma, f.steps,
                  newton_chunk_size=f.newton_chunk_size,
                  newton_cg_max_iter=f.newton_cg_max_iter,
                  newton_cg_tol=f.newton_cg_tol)
    else:
        tvf = None

    return LayerSolver(config, model_spec.wavelength, kx, ky, m_flat, n_flat, tvf)


@dataclass(frozen=True)
class PreparedStack:
    """Frozen snapshot of a fully-prepared stack, ready for :func:`run`.

    Attributes
    ----------
    layersolver : LayerSolver
        The context tier (harmonics, vacuum modes, TVF) — reusable across
        :func:`reprepare` calls at fixed source/lattice.
    ops : tuple of LayerOperator
        One precomputed operator per stack element, ordered
        ``(incidence, *layers, transmission)``.
    model_spec : ModelSpec
        The resolved model snapshot ``ops`` was built from.
    """
    layersolver: LayerSolver
    ops: tuple  # tuple[LayerOperator, ...]
    model_spec: ModelSpec


def prepare(model: Model, config: Config) -> PreparedStack:
    """Resolve ``model`` and solve every stack element's modal eigenproblem.

    This is the expensive step (device transfer, harmonic pre-computation,
    TVF setup, and one modal eigensolve per element).

    Parameters
    ----------
    model : Model
        Stack + source description. Moved to ``config.dtype`` /
        ``config.device`` in-place.
    config : Config
        Solver hyperparameters (grid resolution, harmonic truncation, TVF
        factorization, eigensolver settings).

    Returns
    -------
    PreparedStack
        Frozen snapshot; pass to :func:`run` (cheap) or :func:`reprepare`
        (re-solve a subset of layers at fixed source/lattice).

    Notes
    -----
    ``model`` is mutated in place, not copied: ``nn.Module._apply`` (which
    backs ``Model.to()``) mutates and returns ``self``, so the caller's own
    ``model`` reference is silently cast to ``config.dtype``/``config.device``
    as a side effect of calling ``prepare`` — the stored ``model_spec`` is
    built from the *same object*, not a clone. If the caller needs to
    preserve the original (e.g. to build a second prepared stack at a
    different dtype/device from the same source model), pass a copy
    (``copy.deepcopy(model)``) instead of the original.
    """
    model = model.to(dtype=config.dtype, device=config.device)
    model_spec = model.spec(config.nx, config.ny)
    layersolver = build_layersolver(model_spec, config)

    # Solve the modal eigenproblem for every stack element once, here.
    # run() then only does cheap Redheffer star-product composition.
    ops = (
        layersolver.prepare(model_spec.incidence),
        *(layersolver.prepare(layer) for layer in model_spec.layers),
        layersolver.prepare(model_spec.transmission),
    )
    return PreparedStack(layersolver=layersolver, ops=ops, model_spec=model_spec)


def reprepare(prepared: PreparedStack, model: Model, config: Config,
              layer_indices) -> PreparedStack:
    """Re-solve only the named layers, reusing everything else.

    Re-resolves ``model.spec()`` (cheap rasterization) and rebuilds the
    :class:`LayerOperator` for each index in ``layer_indices`` via
    ``prepared.layersolver.prepare(...)``, but reuses ``prepared.layersolver``
    (the harmonics/vacuum-mode/TVF context) and every *other* element's
    already-prepared operator unchanged.

    This is the efficiency lever for single-layer (or few-layer) inverse
    design at fixed source/lattice: it costs one eigendecomposition per
    named layer instead of one per stack element.

    Parameters
    ----------
    prepared : PreparedStack
        A snapshot previously returned by :func:`prepare` (or
        :func:`reprepare`), built from the same ``model``/``config`` up to
        the layer patterns named in ``layer_indices``.
    model : Model
        The same model instance ``prepared`` was built from, with the
        targeted layers' pattern parameters updated in place.
    config : Config
        Same ``Config`` used to build ``prepared``.
    layer_indices : Iterable[int]
        Indices into ``model_spec.layers`` (0-based, *not* offset for the
        incidence/transmission boundaries) whose operators must be
        re-prepared.

    Returns
    -------
    PreparedStack
        New frozen snapshot with only the targeted operators rebuilt; all
        other operators and the ``layersolver`` are reused by reference
        (``is`` identity), not recomputed.

    Notes
    -----
    Assumes the source and lattice (wavelength, incidence angles, ``a1``,
    ``a2``) are unchanged from ``prepared`` — that is what makes reusing
    ``prepared.layersolver`` valid. If they *have* changed, call
    :func:`prepare` instead; reusing a stale ``layersolver`` in that case
    would silently produce wrong physics.

    Reused (non-rebuilt) operators still carry the autograd graph from the
    ``prepare``/``reprepare`` call that created them. In a long-running
    single-layer optimization, build the fixed layers once under
    ``torch.no_grad()`` (or ``.detach()`` their operators) so no stale graph
    is retained across steps.
    """
    model = model.to(dtype=config.dtype, device=config.device)
    model_spec = model.spec(config.nx, config.ny)
    layersolver = prepared.layersolver

    ops = list(prepared.ops)
    for i in layer_indices:
        ops[i + 1] = layersolver.prepare(model_spec.layers[i])

    return replace(prepared, ops=tuple(ops), model_spec=model_spec)

@dataclass(frozen=True)
class ModalSolution:
    """Cheap, excitation-independent output of :func:`run`.

    Bundles the scattering blocks needed for **single-side (left) excitation**
    with a handle back to the :class:`PreparedStack` they were assembled from.
    That handle is what lets downstream code reach each element's modes
    (``W``/``V``/``lam`` on every
    :class:`~metarcwa.solver.layersolver.operator.LayerOperator` in
    ``prepared.ops``) and the harmonic context (``kx``/``ky``/wavelength on
    ``prepared.layersolver``) without re-solving anything.

    Only the first *column* of the full-stack S-matrix is retained. With the
    incidence medium on the left and no illumination from the right
    (``in_right = 0``), the outgoing fields are ``out_left = S11·in_left``
    (reflection) and ``out_right = S21·in_left`` (transmission); the
    ``S12``/``S22`` blocks act only on ``in_right`` and are never used, so
    they are dropped to save memory. This bakes in the left-illumination
    convention — a right-side or two-sided excitation would need them.

    Nothing stored here depends on the incident polarization/amplitude: the
    S-matrix blocks and modes are excitation-independent by design
    (:class:`Source` carries no polarization). Reflection/transmission
    efficiencies are obtained downstream by applying a specific excitation to
    ``S11``/``S21``; full internal fields require the heavier
    :class:`FieldSolution` (which re-sweeps the stack from ``prepared`` and so
    does **not** rely on the dropped blocks).

    Attributes
    ----------
    S11 : Entry
        Reflection block (``a`` = ``S11`` = "reflection from left"). Maps the
        incident left-side amplitude to the reflected amplitude.
    S21 : Entry
        Transmission block (``c`` = ``S21`` = "transmission from left"). Maps
        the incident left-side amplitude to the transmitted amplitude.
    prepared : PreparedStack
        The snapshot the blocks were assembled from — carries the per-element
        :class:`~metarcwa.solver.layersolver.operator.LayerOperator` modes and
        the :class:`LayerSolver` context. Held by reference, not copied.
    """
    S11: Entry
    S21: Entry
    prepared: PreparedStack

@dataclass(frozen=True)
class FieldSolution:
    """On-demand output for internal-field reconstruction (planned).

    Produced by the field pipeline for a *specific* excitation, and a strict
    superset of :class:`ModalSolution`'s capabilities: it embeds the
    :class:`ModalSolution` (so reflection/transmission stay available) and
    adds the per-element modal amplitudes needed to evaluate E/H fields
    anywhere in the stack. The mode matrices themselves (``W``/``V``/``lam``)
    are **not** duplicated here — they are reached through
    ``modal.prepared.ops`` — so this object's own payload is only the
    amplitudes and the excitation they were built from.

    .. note::
        Not yet produced: ``Solver.run(fields=True)`` currently raises
        :class:`NotImplementedError`. This documents the intended shape.

    Attributes
    ----------
    modal : ModalSolution
        The embedded excitation-independent solution (S-matrix + prepared
        handle). Keeps R/T available and carries the modes used to expand the
        amplitudes into real-space fields.
    amplitudes
        Per-element forward/backward modal coefficients ``(c⁺, c⁻)``, aligned
        index-for-index with ``modal.prepared.ops`` — obtained by
        back-substituting the excitation through the stack's cut-plane partial
        S-matrices. (Type TBD; placeholder ``...`` annotation for now.)
    excitation
        The incident amplitude vector ``c_inc`` (the excitation mapped into
        the incidence-medium modes) this reconstruction was built from.
        (Type TBD; placeholder ``...`` annotation for now.)
    """
    modal: ModalSolution
    amplitudes: ...             # per-layer (c⁺, c⁻)
    excitation: ...             # the c_inc it was built from

def run(prepared: PreparedStack) -> ModalSolution:
    """Compute the full-stack scattering solution from a prepared snapshot.

    Assembles the S-matrix by star-multiplying the incidence boundary,
    every finite layer in order, and the transmission boundary, using the
    :class:`LayerOperator` objects in ``prepared.ops``. The full ``Block2x2``
    is formed transiently by the star product, but only its first column
    (``S11`` reflection, ``S21`` transmission) is retained in the returned
    :class:`ModalSolution` — the ``S12``/``S22`` blocks are unused for
    left-side excitation and are left to be garbage-collected.

    Parameters
    ----------
    prepared : PreparedStack
        Result of :func:`prepare` or :func:`reprepare`.

    Returns
    -------
    solution : ModalSolution
        The reflection/transmission blocks (``solution.S11``/``solution.S21``)
        bundled with the :class:`PreparedStack` handle
        (``solution.prepared``) they were computed from, so downstream
        observables and field reconstruction can reach the per-layer modes
        without re-solving.
    """
    ls = prepared.layersolver
    ops = prepared.ops
    S = ls.smatrix(ops[0], left=True)
    for op in ops[1:-1]:
        S = S.star(ls.smatrix(op))
    S = S.star(ls.smatrix(ops[-1], left=False))
    # Keep only the first column (a=S11 reflection, c=S21 transmission); the
    # b=S12 / d=S22 blocks act only on right-side input and are dropped.
    return ModalSolution(S11=S.a, S21=S.c, prepared=prepared)

class Solver:
    """Top-level RCWA solver — thin convenience wrapper over the functional
    core (:func:`prepare`, :func:`run`, :func:`reprepare`).

    Holds a fully-resolved model snapshot, a pre-initialised
    :class:`LayerSolver`, and the precomputed
    :class:`~metarcwa.solver.layersolver.operator.LayerOperator` for every stack
    element (incidence boundary, each finite layer, transmission boundary).
    Constructing a ``Solver`` is the expensive step (device transfer,
    harmonic pre-computation, TVF setup, and one modal eigensolve per
    element); calling :meth:`run` is then genuinely cheap — pure
    Redheffer star-product composition, no eigendecomposition.

    For single-layer (or few-layer) inverse design at fixed source/lattice,
    prefer :meth:`reprepare` over rebuilding a new ``Solver`` — see its
    docstring and :func:`reprepare`.

    Attributes
    ----------
    model : Model
        The simulation model, cast to the dtype/device given in ``config``.
    config : Config
        Solver hyperparameters.
    layersolver : LayerSolver
        Pre-initialised layer-level modal solver / S-matrix assembler.
    model_spec : ModelSpec
        The resolved model snapshot the current operators were built from.
    """

    def __init__(self, model: Model, config: Config) -> None:
        """
        Parameters
        ----------
        model : Model
            Stack + source description. Moved to ``config.dtype`` /
            ``config.device`` in-place.
        config : Config
            Solver hyperparameters (grid resolution, harmonic truncation,
            TVF factorization, eigensolver settings).

        Notes
        -----
        ``model`` is mutated in place, not copied: ``nn.Module._apply``
        (which backs ``Model.to()``) mutates and returns ``self``, so the
        caller's own ``model`` reference is silently cast to
        ``config.dtype``/``config.device`` as a side effect of constructing
        a ``Solver`` — ``self.model`` is the *same object*, not a clone. If
        the caller needs to preserve the original (e.g. to build a second
        ``Solver`` at a different dtype/device from the same source model),
        pass a copy (``copy.deepcopy(model)``) instead of the original.
        """
        self.config = config
        self._prepared = prepare(model, config)
        # prepare() moves `model` to config.dtype/device in place and returns
        # the same object from model.to(...); model_spec was built from it.
        self.model = model

    @property
    def layersolver(self) -> LayerSolver:
        return self._prepared.layersolver

    @property
    def model_spec(self) -> ModelSpec:
        return self._prepared.model_spec

    @property
    def _ops(self):
        return self._prepared.ops
    
    def reprepare(self, layer_indices) -> "Solver":
        """Re-solve only the named layers in place; reuse everything else.

        See :func:`reprepare` for the full contract (source/lattice must be
        unchanged; autograd-graph retention on reused operators).

        Parameters
        ----------
        layer_indices : Iterable[int]
            Indices into ``model_spec.layers`` whose operators must be
            re-prepared (e.g. after mutating that layer's pattern
            parameters in place).

        Returns
        -------
        Solver
            ``self``, updated in place, for chaining
            (``solver.reprepare([i]).run()``).
        """
        self._prepared = reprepare(self._prepared, self.model, self.config, layer_indices)
        return self

    def run(self, fields: bool = False) -> ModalSolution | FieldSolution:
        """Solve the bound stack and return its scattering solution.

        Cheap: pure Redheffer star-product composition of the precomputed
        operators, with no eigendecomposition. Safe to call repeatedly after
        :meth:`reprepare` or an in-place thickness update, without rebuilding
        the ``Solver``.

        Parameters
        ----------
        fields : bool, default False
            If ``False``, return only the excitation-independent
            :class:`ModalSolution` (S-matrix + prepared handle) — sufficient
            for reflection/transmission observables. If ``True``, also
            reconstruct the internal modal amplitudes and return a
            :class:`FieldSolution` for near-field / absorption observables.
            **Not implemented yet — raises** :class:`NotImplementedError`.

        Returns
        -------
        ModalSolution or FieldSolution
            :class:`ModalSolution` when ``fields=False``;
            :class:`FieldSolution` when ``fields=True`` (once implemented).

        Raises
        ------
        NotImplementedError
            If ``fields=True`` — field reconstruction is not built yet.
        """
        if fields:
            raise NotImplementedError("Fields computation is not implemented yet.")
        return run(self._prepared)

    
