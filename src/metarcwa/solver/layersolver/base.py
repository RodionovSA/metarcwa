# metarcwa/solver/layersolver/base.py
"""
base — LayerSolver: per-element modal solver and S-matrix assembler
=====================================================================

``LayerSolver`` is the central mode-solving orchestrator. It splits work into
two phases with very different cost (for the default ``"eig"`` modesolver —
see the note on ``"matexp"`` below):

  - :meth:`prepare` — EXPENSIVE. Solves the per-element eigenproblem (TVF
    field, convolution matrices, eigendecomposition for patterned layers;
    closed-form modes for homogeneous layers/media) and returns a
    :class:`~metarcwa.solver.layersolver.operator.LayerOperator` — a plain
    snapshot of the modal solution.
  - :meth:`smatrix` — CHEAP. Assembles the ``Block2x2`` S-matrix from an
    already-prepared operator via boundary matching + propagation.

:meth:`solve` composes the two for callers that don't need to reuse the
prepared operator. The caller owns the operator's lifetime: rebuild it
whenever the underlying geometry/pattern/source changes (e.g. every
inverse-design step); reuse it across repeated ``smatrix()`` calls at fixed
geometry (e.g. a thickness sweep, where only ``op.thickness`` changes).

Supported element types and their solvers:

  HomogeneousLayer   → :func:`homogeneous_modes` (closed-form, no eigensolver)
                       → :class:`~metarcwa.solver.layersolver.operator.ModalOperator`
  PatternedLayer     → :func:`compute_isotropic`, then either:
                         "eig"    → :func:`eigsolver` → ``ModalOperator``
                         "matexp" → :mod:`~metarcwa.solver.layersolver.matexpsolver`
                                    → :class:`~metarcwa.solver.layersolver.matexpsolver.TransferOperator`
                       (``Config.modesolver`` selects the branch)
  MediumSpec         → :func:`homogeneous_modes` → ``ModalOperator``
                       (semi-infinite input/output medium; boundary-only,
                       no propagation)

``LayerOperator`` itself is a structural contract (see ``operator.py``), not
a single concrete type — ``ModalOperator`` and ``TransferOperator`` both
satisfy it. ``smatrix()`` below therefore delegates to the operator itself
(``op.smatrix(self.background, left)``) rather than assuming a
``(lam, W, V)`` triple.

Note on cost model under ``"matexp"``: it inverts. ``prepare()`` becomes
cheap (no eigendecomposition — just ``P``/``Q``) and ``smatrix()`` becomes
the expensive step (a sliced matrix exponential, computed there rather than
in ``prepare()`` specifically so it stays responsive to
``dataclasses.replace(op, thickness=...)``, matching the eig path's
late-binding of ``thickness``). See ``docs/matrixexp.md``.
"""

import torch
from torch.utils.checkpoint import checkpoint
from typing import Tuple

from metarcwa.model.layer import HomogeneousLayer, PatternedLayer
from metarcwa.model.medium import MediumSpec, IsotropicMediumSpec
from metarcwa.solver.blockmatrix import Block2x2
from metarcwa.solver.layersolver.homogeneous import homogeneous_modes
from metarcwa.solver.layersolver.isotropic import compute_isotropic
from metarcwa.solver.layersolver.eigsolver import eigsolver
from metarcwa.solver.layersolver.matexpsolver import TransferOperator
from metarcwa.solver.layersolver.operator import Background, LayerOperator, ModalOperator
from metarcwa.solver.layersolver._modes import _regularize_eps
from metarcwa.solver.config import Config


class LayerSolver:
    """
    Per-element modal solver and S-matrix assembler for RCWA layer stacks.

    Pre-computes the background (vacuum) mode matrices ``W0`` and ``V0``
    from ``kx``, ``ky`` at construction time; reuses them for every
    homogeneous and patterned layer in the stack.

    Attributes
    ----------
    config : Config
        Solver hyperparameters (grid, truncation, factorization, modesolver).
    wvl : torch.Tensor
        Free-space wavelengths, shape ``[N_wvl]``.
    kx : torch.Tensor
        In-plane x-wavevectors for all harmonics, shape ``[N_wvl, N_theta, N_phi, Nh]``.
    ky : torch.Tensor
        In-plane y-wavevectors for all harmonics, shape ``[N_wvl, N_theta, N_phi, Nh]``.
    m_flat : torch.Tensor
        Integer harmonic indices along a1, shape ``[Nh]``.
    n_flat : torch.Tensor
        Integer harmonic indices along a2, shape ``[Nh]``.
    tvf : TVF or None
        Configured TVF instance for Li-factorization, or ``None``.
    W0 : Block2x2
        Background E-mode matrix (identity for vacuum).
    V0 : Block2x2
        Background H-mode matrix computed from vacuum dispersion.
    background : Background
        ``W0``/``V0``/``wvl`` bundled into the reference every
        :class:`~metarcwa.solver.layersolver.operator.LayerOperator` is
        assembled against; passed to ``op.smatrix()``/``op.transfer()``.
    """

    def __init__(self, config: Config, wvl: torch.Tensor,
                 kx: torch.Tensor, ky: torch.Tensor,
                 m_flat: torch.Tensor, n_flat: torch.Tensor,
                 tvf=None):
        """
        Parameters
        ----------
        config : Config
            Solver hyperparameters.
        wvl : torch.Tensor
            Free-space wavelengths, shape ``[N_wvl]``.
        kx : torch.Tensor
            In-plane x-wavevectors, shape ``[N_wvl, Nh]``.
        ky : torch.Tensor
            In-plane y-wavevectors, shape ``[N_wvl, Nh]``.
        m_flat : torch.Tensor
            Integer harmonic indices along a1, shape ``[Nh]``.
        n_flat : torch.Tensor
            Integer harmonic indices along a2, shape ``[Nh]``.
        tvf : TVF or None, optional
            Configured TVF instance for Li-factorization.  Pass ``None``
            (default) to use the plain Laurent convolution rule.
        """
        self.config  = config
        self.wvl     = wvl
        self.kx      = kx
        self.ky      = ky
        self.m_flat  = m_flat
        self.n_flat  = n_flat
        self.tvf     = tvf
        self.W0, self.V0 = self._prepare_vacuum()
        self.background = Background(self.W0, self.V0, self.wvl)

    def _prepare_vacuum(self) -> Tuple[Block2x2, Block2x2]:
        """Compute the vacuum background mode matrices W0 = I and V0.

        Called once at construction.  Uses ε = 1 (vacuum) with the same
        ``kx``/``ky`` grid as the rest of the stack. Regularized by
        ``config.grazing_eps_reg`` — the lossless reference is otherwise
        itself exactly grazing whenever a harmonic's specular in-plane
        wavevector reaches unit magnitude (see :func:`_regularize_eps`).
        """
        eps = torch.ones(self.kx.shape[0], dtype=self.kx.dtype, device=self.kx.device)
        eps = _regularize_eps(eps, self.config.grazing_eps_reg)
        _, V0 = homogeneous_modes(eps, self.kx, self.ky)
        return V0.eye_like(), V0

    def prepare(self, element: HomogeneousLayer | PatternedLayer | MediumSpec
                ) -> LayerOperator:
        """
        Solve the modal eigenproblem for a single stack element (expensive).

        Dispatches to :meth:`_homogeneous`, :meth:`_patterned`, or
        :meth:`_medium` depending on the element type. The returned
        :class:`LayerOperator` is pure data — reuse it across multiple
        :meth:`smatrix` calls (e.g. a thickness sweep) instead of calling
        ``prepare`` again.

        Parameters
        ----------
        element : HomogeneousLayer or PatternedLayer or MediumSpec
            The layer or medium specification to solve.

        Returns
        -------
        op : LayerOperator
            Precomputed modal solution; pass to :meth:`smatrix` to assemble
            the S-matrix.
        """
        if isinstance(element, HomogeneousLayer):
            return self._homogeneous(element)
        elif isinstance(element, PatternedLayer):
            return self._patterned(element)
        elif isinstance(element, MediumSpec):
            return self._medium(element)
        else:
            raise TypeError(
                f"element must be HomogeneousLayer, PatternedLayer, or "
                f"MediumSpec, but got {type(element)}"
            )

    def smatrix(self, op: LayerOperator, left: bool = True) -> Block2x2:
        """
        Assemble the S-matrix from a prepared operator (cheap for the "eig"
        modesolver; see the module docstring's note on "matexp").

        Delegates to the operator itself — ``op.smatrix(self.background,
        left)`` — since different operator families (``ModalOperator``,
        ``TransferOperator``) assemble it differently (modal boundary
        matching + propagation vs. a sliced matrix exponential). See
        ``operator.py`` for the shared contract.

        Parameters
        ----------
        op : LayerOperator
            Result of :meth:`prepare`.
        left : bool, optional
            For semi-infinite media only: ``True`` (default) treats the
            medium as the left (input) semi-infinite region; ``False``
            treats it as the right (output) semi-infinite region. Has no
            effect for finite layers.

        Returns
        -------
        S : Block2x2
            S-matrix of the element; compose successive elements with
            ``S1.star(S2)`` (Redheffer star product).
        """
        return op.smatrix(self.background, left)

    def run(self, element: HomogeneousLayer | PatternedLayer | MediumSpec,
            left: bool = True) -> Block2x2:
        """
        Compute the S-matrix for a single stack element.

        Convenience wrapper equivalent to ``smatrix(prepare(element), left)``.
        Prefer calling :meth:`prepare` once and :meth:`smatrix` repeatedly
        when the same element is run more than once (e.g. spectral sweeps
        or thickness-only optimization at fixed geometry).

        Parameters
        ----------
        element : HomogeneousLayer or PatternedLayer or MediumSpec
            The layer or medium specification to solve.
        left : bool, optional
            For ``MediumSpec`` only: ``True`` (default) treats the medium
            as the left (input) semi-infinite region;  ``False`` treats it
            as the right (output) semi-infinite region.  Has no effect for
            layer types.

        Returns
        -------
        S : Block2x2
            S-matrix of the element; compose successive elements with
            ``S1.star(S2)`` (Redheffer star product).
        """
        return self.smatrix(self.prepare(element), left)

    def _homogeneous(self, layer: HomogeneousLayer) -> LayerOperator:
        """
        Solve the modes of a homogeneous layer using closed-form modes.
        """
        medium = layer.medium
        if isinstance(medium, IsotropicMediumSpec):
            eps = _regularize_eps(medium.eps, self.config.grazing_eps_reg)
            lam, V = homogeneous_modes(eps, self.kx, self.ky)
            W      = V.eye_like()
        else:
            raise NotImplementedError(
                f"Homogeneous solver not implemented for {type(medium)}"
            )
        return ModalOperator(lam, W, V, layer.thickness)

    def _patterned(self, layer: PatternedLayer) -> LayerOperator:
        """Solve a patterned layer via ``Config.modesolver`` ("eig" or
        "matexp").

        Builds the permittivity grid from ``medium_solid`` and ``medium_void``
        weighted by ``pattern``, computes P and Q operators via
        :func:`compute_isotropic` (shared by both modesolvers — same TVF
        correction, same convolution matrices), then dispatches:

          "eig"    → :func:`eigsolver` → :class:`ModalOperator`
          "matexp" → :class:`~metarcwa.solver.layersolver.matexpsolver.TransferOperator`
                     (no eigendecomposition; P/Q and a cheap modal-exponent
                     bound are carried directly, exponentiated at
                     ``smatrix()`` time)
        """
        medium_solid = layer.medium_solid
        medium_void  = layer.medium_void
        pattern      = layer.pattern

        if isinstance(medium_solid, IsotropicMediumSpec) and \
                isinstance(medium_void, IsotropicMediumSpec):
            eps_solid = medium_solid.eps   # [*B] (B may be empty, i.e. 0-dim)
            eps_void  = medium_void.eps    # [*B]
            eps_grid  = (eps_solid[..., None, None] * pattern
                         + (1 - pattern) * eps_void[..., None, None])
            eps_grid  = _regularize_eps(eps_grid, self.config.grazing_eps_reg)

            if self.tvf is not None:
                # TVF is geometry-only (detached, sign/scale-invariant in the A-blocks):
                # compute once from the pattern mask, [1, Ny, Nx], not per wavelength.
                tvf_fields = self.tvf.compute(pattern[None])
            else:
                tvf_fields = None

            P, Q = compute_isotropic(
                eps_grid, self.m_flat, self.n_flat,
                self.kx, self.ky, tvf_fields,
            )
            if self.config.modesolver == "eig":
                if self.config.checkpoint_eig:
                    lam, W, V = checkpoint(
                        eigsolver, P, Q, self.config.eigsolver_stable,
                        use_reentrant=False,
                    )
                else:
                    lam, W, V = eigsolver(P, Q, self.config.eigsolver_stable)
                return ModalOperator(lam, W, V, layer.thickness)
            elif self.config.modesolver == "matexp":
                # Cheap upper bound on the modal exponent, no eigenvalues
                # needed: lam^2 ~ kx^2 + ky^2 - eps for the isotropic system,
                # so max|lam| <~ sqrt(max(kx^2+ky^2) + max|eps|). Detached —
                # only sizes the auto slice count (matexpsolver.slice_count),
                # never enters autograd.
                Nh = self.m_flat.shape[0]
                kxy2_max = (self.kx.detach().abs() ** 2
                            + self.ky.detach().abs() ** 2).amax()
                eps_max = eps_grid.detach().abs().amax().to(kxy2_max.dtype)
                lam_bound = torch.sqrt(kxy2_max + eps_max)
                return TransferOperator(
                    P, Q, Nh, lam_bound, self.config, layer.thickness,
                )
            else:
                raise NotImplementedError(
                    f"modesolver '{self.config.modesolver}' is not supported. "
                    "Currently 'eig' and 'matexp' are implemented."
                )
        else:
            raise NotImplementedError(
                f"Patterned solver not implemented for "
                f"({type(medium_solid)}, {type(medium_void)})"
            )

    def _medium(self, medium: MediumSpec) -> LayerOperator:
        """Solve the modes of a semi-infinite medium.

        Parameters
        ----------
        medium : MediumSpec
            The semi-infinite medium specification.
        """
        if isinstance(medium, IsotropicMediumSpec):
            eps = _regularize_eps(medium.eps, self.config.grazing_eps_reg)
            lam, V = homogeneous_modes(eps, self.kx, self.ky)
            W      = V.eye_like()
        else:
            raise NotImplementedError(
                f"Medium solver not implemented for {type(medium)}"
            )
        return ModalOperator(lam, W, V, thickness=None)
