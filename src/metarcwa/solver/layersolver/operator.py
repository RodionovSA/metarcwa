# metarcwa/solver/layersolver/operator.py
"""
operator — the LayerOperator contract shared by every mode-solver family
==========================================================================

A prepared stack element's S-matrix can be assembled in more than one way —
from an eigenmode decomposition, or from a sliced matrix exponential
(:mod:`metarcwa.solver.layersolver.matexpsolver`) — and the two produce
different data (modes vs. first-order-system operators). ``LayerOperator``
is therefore a structural :class:`~typing.Protocol` (the same pattern
``Entry`` uses in :mod:`metarcwa.solver.blockmatrix` for
``Block``/``Block2x2``), not one dataclass, so each solver family can carry
exactly what it needs:

  :class:`ModalOperator`    — eigenmode / closed-form solvers (eigsolver,
                              homogeneous_modes). Carries ``lam``, ``W``, ``V``.
  :class:`TransferOperator` — matrix-exponential solver. Carries ``P``, ``Q``
                              (the same first-order-system operators the eig
                              path uses) and builds its S-matrix from a sliced
                              matrix exponential.

Both satisfy the same two-method contract:

  ``smatrix(background, left=True)`` — assemble this element's S-matrix
      against the shared vacuum background.
  ``transfer(background, z)`` — the propagator mapping the transverse field
      vector psi(0) to psi(z) at depth z inside the element, in the vacuum
      mode-amplitude basis. This is the basis-free primitive interior-field
      evaluation needs; both families implement it natively (modal: the gap-
      matrix similarity transform; matexp: the exponential directly, with no
      eigendecomposition at all), so field visualization can be built against
      this shared interface rather than against ``lam``/``W``/``V`` directly.

``LayerSolver.prepare()`` (expensive) returns one of these per stack element;
``LayerSolver.smatrix()`` (cheap) calls ``op.smatrix(background)`` — the
operator itself knows how to assemble its S-matrix.
"""

from __future__ import annotations

import torch
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from metarcwa.solver.blockmatrix import Block, Block2x2
from metarcwa.solver.smatrix import S_boundary, S_layer


@dataclass(frozen=True)
class Background:
    """The homogeneous (vacuum) reference the whole stack is embedded in.

    Precomputed once by :class:`~metarcwa.solver.layersolver.base.LayerSolver`
    and passed to every element's :meth:`LayerOperator.smatrix`/``transfer``
    call — this is what makes those calls cheap: no per-element vacuum
    recomputation.

    Attributes
    ----------
    W0 : Block2x2
        Vacuum E-mode matrix (identity).
    V0 : Block2x2
        Vacuum H-mode matrix.
    wvl : torch.Tensor
        Free-space wavelengths, shape ``[N_wvl]``.
    """
    W0: Block2x2
    V0: Block2x2
    wvl: torch.Tensor


@runtime_checkable
class LayerOperator(Protocol):
    """Structural contract for a prepared stack-element operator.

    Any object with a ``thickness`` attribute and ``smatrix``/``transfer``
    methods of the right shape satisfies this protocol — see module
    docstring. ``isinstance(op, LayerOperator)`` works at runtime (like
    ``Entry`` in :mod:`metarcwa.solver.blockmatrix`), checking attribute/
    method presence only, not signatures.
    """
    thickness: torch.Tensor | None

    def smatrix(self, background: Background, left: bool = True) -> Block2x2:
        """Assemble this element's S-matrix against ``background``."""
        ...

    def transfer(self, background: Background, z: torch.Tensor) -> Block2x2:
        """Propagator mapping psi(0) -> psi(z) inside this element, expressed
        in ``background``'s vacuum mode-amplitude basis."""
        ...


@dataclass(frozen=True)
class ModalOperator:
    """
    Precomputed eigenmode solution of one stack element.

    Produced by eigenmode-based solvers (:func:`eigsolver`,
    :func:`homogeneous_modes`). The caller owns the lifetime: rebuild when the
    geometry or source changes; reuse across repeated :meth:`smatrix` calls at
    fixed geometry. ``thickness`` is read at :meth:`smatrix` time, so
    thickness-only changes (e.g. an in-place ``nn.Parameter`` update, or
    ``dataclasses.replace(op, thickness=...)``) do NOT require re-preparing.

    Attributes
    ----------
    lam : torch.Tensor
        Modal exponents lam = 1j*kz, shape ``[..., 2Nh]``.
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

    def smatrix(self, background: Background, left: bool = True) -> Block2x2:
        """Assemble the S-matrix (cheap): boundary matching + propagation.

        For a finite layer (``thickness is not None``) this cascades
        ``S_in (star) S_prop (star) S_out`` via :func:`S_layer`. For a
        semi-infinite medium (``thickness is None``) it computes a single
        boundary S-matrix via :func:`S_boundary`.

        Parameters
        ----------
        background : Background
            Shared vacuum reference for the whole stack.
        left : bool, optional
            For semi-infinite media only: ``True`` (default) treats the
            medium as the left (input) semi-infinite region; ``False`` treats
            it as the right (output) semi-infinite region. Has no effect for
            finite layers.
        """
        if self.thickness is None:
            if left:
                return S_boundary(self.W, self.V, background.W0, background.V0)
            else:
                return S_boundary(background.W0, background.V0, self.W, self.V)
        return S_layer(background.W0, background.V0, self.W, self.V,
                        self.lam, self.thickness, background.wvl)

    def transfer(self, background: Background, z: torch.Tensor) -> Block2x2:
        """Propagator psi(0) -> psi(z), via the modal gap-matrix similarity
        transform ``Phi @ diag(e^{lam k0 z}, e^{-lam k0 z}) @ Phi^-1``.

        ``Phi = [[W, W], [V, -V]]`` is the gap matrix (docs/eigenproblem.md).
        Requires ``thickness is not None`` (undefined for a semi-infinite
        medium, which has no interior).

        ``Phi^-1`` is obtained via ``Phi.solve(Phi.eye_like())``: this
        densifies and factorizes the whole matrix at once, which only
        requires ``Phi`` itself (not any individual quadrant) to be
        non-singular — true generically, since its columns are the layer's
        forward/backward eigenmodes. A modal ``V`` (an eigenmode-derived
        H-mode matrix) can have an exactly singular ``d`` sub-block, so
        :meth:`Block2x2.inv`'s Schur-complement-of-``d`` pivot is not safe
        here — the same rank-deficiency :func:`~metarcwa.solver.smatrix.S_boundary`
        is built to avoid.
        """
        Nh = self.lam.shape[-1] // 2
        wvl = torch.as_tensor(background.wvl)
        if wvl.ndim < self.lam.ndim:
            wvl = wvl.reshape(*wvl.shape, *([1] * (self.lam.ndim - wvl.ndim)))
        k0 = 2 * torch.pi / wvl
        xp = torch.exp(self.lam * k0 * z)
        xm = torch.exp(-self.lam * k0 * z)
        kw = dict(device=xp.device, dtype=xp.dtype)
        Dp = Block2x2(Block(Block.DIAG, xp[..., :Nh]), Block.zeros(**kw),
                      Block.zeros(**kw), Block(Block.DIAG, xp[..., Nh:]))
        Dm = Block2x2(Block(Block.DIAG, xm[..., :Nh]), Block.zeros(**kw),
                      Block.zeros(**kw), Block(Block.DIAG, xm[..., Nh:]))
        D = Block2x2(Dp, Dp.zeros_like(), Dp.zeros_like(), Dm)
        Phi = Block2x2(self.W, self.W, self.V, -self.V)
        Phi_inv = Phi.solve(Phi.eye_like())
        return Phi @ D @ Phi_inv
