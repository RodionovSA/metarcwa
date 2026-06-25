# metarcwa/solver/layersolver/base.py
"""
base — LayerSolver: per-element S-matrix dispatcher
====================================================

``LayerSolver`` is the central mode-solving orchestrator.  It:

  - Pre-computes the vacuum background modes (W0, V0) once at construction.
  - Dispatches each stack element to the appropriate sub-solver via
    :meth:`solve`.
  - Returns a ``Block2x2`` S-matrix for each element, ready to be composed
    with the Redheffer star product.

Supported element types and their solvers:

  HomogeneousLayer   → :func:`homogeneous_modes` (closed-form, no eigensolver)
  PatternedLayer     → :func:`compute_isotropic` + :func:`eigsolver`
  MediumSpec         → :func:`homogeneous_modes` + :func:`S_boundary`
                       (semi-infinite input/output medium)
"""

import torch
from typing import Tuple

from metarcwa.model.layer import HomogeneousLayer, PatternedLayer
from metarcwa.model.medium import MediumSpec, IsotropicMediumSpec
from metarcwa.solver.blockmatrix import Block2x2
from metarcwa.solver.smatrix import S_layer, S_boundary
from metarcwa.solver.layersolver.homogeneous import homogeneous_modes
from metarcwa.solver.layersolver.isotropic import compute_isotropic
from metarcwa.solver.layersolver.eigsolver import eigsolver
from metarcwa.solver.config import Config


class LayerSolver:
    """
    Per-element S-matrix solver for RCWA layer stacks.

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

    def solve(self, element: HomogeneousLayer | PatternedLayer | MediumSpec,
              left: bool = True) -> Block2x2:
        """
        Compute the S-matrix for a single stack element.

        Dispatches to :meth:`_homogeneous`, :meth:`_patterned`, or
        :meth:`_medium` depending on the element type.

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
        if isinstance(element, HomogeneousLayer):
            return self._homogeneous(element)
        elif isinstance(element, PatternedLayer):
            return self._patterned(element)
        elif isinstance(element, MediumSpec):
            return self._medium(element, left)
        else:
            raise TypeError(
                f"element must be HomogeneousLayer, PatternedLayer, or "
                f"MediumSpec, but got {type(element)}"
            )

    def _homogeneous(self, layer: HomogeneousLayer) -> Block2x2:
        """
        Compute S-matrix for a homogeneous layer using closed-form modes.
        """
        medium = layer.medium
        d      = layer.thickness
        if isinstance(medium, IsotropicMediumSpec):
            lam, V = homogeneous_modes(medium.eps, self.kx, self.ky)
            W      = V.eye_like()
        else:
            raise NotImplementedError(
                f"Homogeneous solver not implemented for {type(medium)}"
            )
        return S_layer(self.W0, self.V0, W, V, lam, d, self.wvl)

    def _patterned(self, layer: PatternedLayer) -> Block2x2:
        """Compute S-matrix for a patterned layer via eigensolver.

        Builds the permittivity grid from ``medium_solid`` and ``medium_void``
        weighted by ``pattern``, computes P and Q operators via
        :func:`compute_isotropic`, then solves for modes with
        :func:`eigsolver`.
        """
        medium_solid = layer.medium_solid
        medium_void  = layer.medium_void
        d            = layer.thickness
        pattern      = layer.pattern

        if isinstance(medium_solid, IsotropicMediumSpec) and \
                isinstance(medium_void, IsotropicMediumSpec):
            eps_solid = medium_solid.eps   # [N_wvl]
            eps_void  = medium_void.eps    # [N_wvl]
            eps_grid  = (eps_solid[:, None, None] * pattern[None, ...]
                         + (1 - pattern[None, ...]) * eps_void[:, None, None])

            P, Q = compute_isotropic(
                eps_grid, self.m_flat, self.n_flat,
                self.kx, self.ky, self.tvf,
            )
            if self.config.modesolver == "eig":
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
        return S_layer(self.W0, self.V0, W, V, lam, d, self.wvl)

    def _medium(self, medium: MediumSpec, left: bool) -> Block2x2:
        """Compute the interface S-matrix for a semi-infinite medium.

        Uses :func:`S_boundary` to match the medium modes against the
        background (vacuum) modes.  The ``left`` flag controls which side
        of the interface the medium occupies:

          left=True  → medium is on the left  (vacuum on the right)
          left=False → medium is on the right (vacuum on the left)

        Parameters
        ----------
        medium : MediumSpec
            The semi-infinite medium specification.
        left : bool
            See description above.
        """
        if isinstance(medium, IsotropicMediumSpec):
            _, V = homogeneous_modes(medium.eps, self.kx, self.ky)
            W    = V.eye_like()
        else:
            raise NotImplementedError(
                f"Medium solver not implemented for {type(medium)}"
            )
        if left:
            return S_boundary(W, V, self.W0, self.V0)
        else:
            return S_boundary(self.W0, self.V0, W, V)
