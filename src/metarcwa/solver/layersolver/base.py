# metarcwa/solver/layersolver/base.py
"""
base — LayerSolver: per-element modal solver and S-matrix assembler
=====================================================================

``LayerSolver`` is the central mode-solving orchestrator. It splits work into
two phases with very different cost:

  - :meth:`prepare` — EXPENSIVE. Solves the per-element eigenproblem (TVF
    field, convolution matrices, eigendecomposition for patterned layers;
    closed-form modes for homogeneous layers/media) and returns a
    :class:`LayerOperator` — a plain snapshot of the modal solution.
  - :meth:`smatrix` — CHEAP. Assembles the ``Block2x2`` S-matrix from an
    already-prepared operator via boundary matching + propagation.

:meth:`solve` composes the two for callers that don't need to reuse the
prepared operator. The caller owns the operator's lifetime: rebuild it
whenever the underlying geometry/pattern/source changes (e.g. every
inverse-design step); reuse it across repeated ``smatrix()`` calls at fixed
geometry (e.g. a thickness sweep, where only ``op.thickness`` changes).

Supported element types and their solvers:

  HomogeneousLayer   → :func:`homogeneous_modes` (closed-form, no eigensolver)
  PatternedLayer     → :func:`compute_isotropic` + :func:`eigsolver`
  MediumSpec         → :func:`homogeneous_modes`
                       (semi-infinite input/output medium; boundary-only,
                       no propagation)
"""

import torch
from torch.utils.checkpoint import checkpoint
from dataclasses import dataclass
from typing import Tuple

from metarcwa.model.layer import HomogeneousLayer, PatternedLayer
from metarcwa.model.medium import MediumSpec, IsotropicMediumSpec
from metarcwa.solver.blockmatrix import Block2x2
from metarcwa.solver.smatrix import S_layer, S_boundary
from metarcwa.solver.layersolver.homogeneous import homogeneous_modes
from metarcwa.solver.layersolver.isotropic import compute_isotropic
from metarcwa.solver.layersolver.eigsolver import eigsolver
from metarcwa.solver.config import Config


@dataclass(frozen=True)
class LayerOperator:
    """
    Precomputed modal solution of one stack element.

    Produced by :meth:`LayerSolver.prepare` (the expensive step: TVF field,
    convolution matrices, eigendecomposition). Consumed by
    :meth:`LayerSolver.smatrix` (cheap: boundary matching + propagation).

    The caller owns the lifetime: rebuild when the geometry or source
    changes (e.g. every inverse-design step); reuse across repeated
    ``smatrix()`` calls at fixed geometry. ``thickness`` is read at
    ``smatrix()`` time, so thickness-only changes (e.g. an in-place
    ``nn.Parameter`` update, or ``dataclasses.replace(op, thickness=...)``)
    do NOT require re-preparing.

    Attributes
    ----------
    lam : torch.Tensor
        Modal exponents lam = 1j·kz, shape ``[..., 2Nh]``.
    W : Block2x2
        E-mode matrix (columns = eigenvectors of the E-field).
    V : Block2x2
        H-mode matrix.
    thickness : torch.Tensor or None
        Layer thickness. ``None`` marks a semi-infinite medium — a
        boundary-only element with no propagation.
    """
    lam: torch.Tensor
    W: Block2x2
    V: Block2x2
    thickness: torch.Tensor | None = None


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

    def _prepare_vacuum(self) -> Tuple[Block2x2, Block2x2]:
        """Compute the vacuum background mode matrices W0 = I and V0.

        Called once at construction.  Uses ε = 1 (vacuum) with the same
        ``kx``/``ky`` grid as the rest of the stack.
        """
        eps = torch.ones(self.kx.shape[0], dtype=self.kx.dtype, device=self.kx.device)
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
        Assemble the S-matrix from a prepared operator (cheap).

        For a finite layer (``op.thickness is not None``) this cascades
        ``S_in ⋆ S_prop ⋆ S_out`` via :func:`S_layer`. For a semi-infinite
        medium (``op.thickness is None``) it computes a single boundary
        S-matrix via :func:`S_boundary`.

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
        if op.thickness is None:
            if left:
                return S_boundary(op.W, op.V, self.W0, self.V0)
            else:
                return S_boundary(self.W0, self.V0, op.W, op.V)
        return S_layer(self.W0, self.V0, op.W, op.V, op.lam, op.thickness, self.wvl)

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
            lam, V = homogeneous_modes(medium.eps, self.kx, self.ky)
            W      = V.eye_like()
        else:
            raise NotImplementedError(
                f"Homogeneous solver not implemented for {type(medium)}"
            )
        return LayerOperator(lam, W, V, layer.thickness)

    def _patterned(self, layer: PatternedLayer) -> LayerOperator:
        """Solve the modes of a patterned layer via eigensolver.

        Builds the permittivity grid from ``medium_solid`` and ``medium_void``
        weighted by ``pattern``, computes P and Q operators via
        :func:`compute_isotropic`, then solves for modes with
        :func:`eigsolver`.
        """
        medium_solid = layer.medium_solid
        medium_void  = layer.medium_void
        pattern      = layer.pattern

        if isinstance(medium_solid, IsotropicMediumSpec) and \
                isinstance(medium_void, IsotropicMediumSpec):
            eps_solid = medium_solid.eps   # [N_wvl]
            eps_void  = medium_void.eps    # [N_wvl]
            eps_grid  = (eps_solid[:, None, None] * pattern[None, ...]
                         + (1 - pattern[None, ...]) * eps_void[:, None, None])

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
            else:
                raise NotImplementedError(
                    f"modesolver '{self.config.modesolver}' is not supported. "
                    "Currently only 'eig' is implemented."
                )
        else:
            raise NotImplementedError(
                f"Patterned solver not implemented for "
                f"({type(medium_solid)}, {type(medium_void)})"
            )
        return LayerOperator(lam, W, V, layer.thickness)

    def _medium(self, medium: MediumSpec) -> LayerOperator:
        """Solve the modes of a semi-infinite medium.

        Parameters
        ----------
        medium : MediumSpec
            The semi-infinite medium specification.
        """
        if isinstance(medium, IsotropicMediumSpec):
            lam, V = homogeneous_modes(medium.eps, self.kx, self.ky)
            W      = V.eye_like()
        else:
            raise NotImplementedError(
                f"Medium solver not implemented for {type(medium)}"
            )
        return LayerOperator(lam, W, V, thickness=None)
