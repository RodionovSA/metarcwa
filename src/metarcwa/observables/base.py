# metarcwa/observables/base.py
"""
Observables — reflection/transmission in the s/p polarization basis
=====================================================================

:class:`Observables` wraps a :class:`~metarcwa.solver.base.ModalSolution`
(the excitation-independent S-matrix blocks ``S11``/``S21`` in the Cartesian
tangential-field basis (Sx, Sy)) and exposes reflection/transmission as
per-diffraction-order Jones matrices in the s/p (TE/TM) polarization basis,
plus convenience accessors for applying an excitation or reading out named
coefficients (rs, rp, ts, tp).

s/p basis change
-----------------
Each diffraction order (harmonic) has its own local propagation direction
``(kx, ky, kz)``, so the s/p basis is defined *per harmonic*, not once for
the whole field. The basis-change matrix, columns ``[ŝ | p̂_tangential]``::

    R(kz) = (1/ρ) · [ [-ky,  -kz·kx/n],
                       [ kx,  -kz·ky/n] ],   ρ = sqrt(kx² + ky²)

``ŝ`` has no ``kz`` dependence (purely transverse); the tangential
projection of ``p̂`` does. Incident and reflected waves at the same harmonic
share ``(kx, ky)`` but propagate in opposite z-directions, so reflection's
outgoing rotation uses ``-kz`` while its incoming rotation uses ``+kz``;
transmission has no sign flip (the transmitted wave stays forward-
propagating). See ``docs/`` and the design discussion this module
implements for the full derivation.

Given the Cartesian-basis block ``S_xy`` (``S11`` for reflection, ``S21``
for transmission), the s/p Jones operator is ``R_out⁻¹ · S_xy · R_in``,
evaluated as ``R_out.solve(S_xy @ R_in)``.

Basis note
----------
``S11``/``S21`` act directly on tangential-E Fourier coefficients (Sx, Sy)
in the incidence medium (input) and transmission medium (output of S21),
*not* on vacuum-mode coefficients — because both half-spaces are
``IsotropicMediumSpec``, whose mode matrix ``W = I`` (see
``LayerSolver._prepare_vacuum`` / ``homogeneous_modes``). No vacuum-basis
(``W0``) transform is needed.
"""

from typing import Literal, Tuple

import torch

from metarcwa.solver.base import ModalSolution, FieldSolution
from metarcwa.solver.blockmatrix import Block, Block2x2

# Regularization threshold for the kx=ky=0 singularity (rho -> 0), e.g. the
# specular order at/near normal incidence. Same regularization philosophy as
# the Lorentzian/broadening constants elsewhere in the solver
# (homogeneous.py delta=1e-30, eigsolver.py broadening_parameter=1e-10).
_RHO_EPS = 1e-8

Kind = Literal["reflection", "transmission"]
Pol = Literal["s", "p"]


def _broadcast_trailing(x: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
    """Append singleton dims to ``x`` so it right-broadcasts against
    ``like``, preserving ``x``'s existing (leading) axes in place.

    Mirrors the wavelength-broadcast pattern used elsewhere in the solver
    (e.g. ``solver/layersolver/homogeneous.py``): batch tensors carry only
    their own leading axes (typically just wavelength), and must gain
    trailing size-1 axes to broadcast against tensors that also carry
    angle/harmonic axes.
    """
    return x.reshape(x.shape + (1,) * (like.ndim - x.ndim))


def _leaf_element(block: Block, i_out: int, i_in: int) -> torch.Tensor:
    """Extract a single ``(i_out, i_in)`` harmonic-pair scalar from a leaf
    ``Block``, dispatching on SCALAR/DIAG/DENSE.

    Avoids densifying the full ``Nh x Nh`` matrix for a single-element
    query, consistent with the ``Block``/``Block2x2`` design goal of not
    allocating dense harmonic-space arrays until forced.
    """
    if block.kind == Block.DENSE:
        return block.data[..., i_out, i_in]
    if block.kind == Block.DIAG:
        return block.data[..., i_out] if i_out == i_in else torch.zeros_like(block.data[..., 0])
    return block.data if i_out == i_in else torch.zeros_like(block.data)  # SCALAR


class Observables:
    """Reflection/transmission observables in the s/p polarization basis.

    Wraps a :class:`~metarcwa.solver.base.ModalSolution` (excitation-
    independent S-matrix blocks in the Cartesian (Sx, Sy) basis) and exposes
    per-diffraction-order s/p Jones matrices, plus convenience methods for
    applying an excitation or reading out named coefficients (rs, rp, ts,
    tp).

    Diffraction orders are addressed as ``(m, n)`` tuples (the harmonic
    index map, ``LayerSolver.m_flat``/``.n_flat``); both default to
    ``(0, 0)``, the specular order.

    Attributes
    ----------
    modal : ModalSolution
        The excitation-independent S-matrix (S11/S21) + prepared-stack
        handle this instance reads from.
    """

    def __init__(self, modal: ModalSolution):
        self.modal = modal
        self._R_in: Block2x2 | None = None
        self._R_out: dict[Kind, Block2x2] = {}
        self._jones_full: dict[Kind, Block2x2] = {}

    # -- harmonic order lookup ---------------------------------------------

    def _order_index(self, order: Tuple[int, int]) -> int:
        """Resolve a diffraction order ``(m, n)`` to its flat harmonic index."""
        ls = self.modal.prepared.layersolver
        m, n = order
        match = (ls.m_flat == m) & (ls.n_flat == n)
        idx = match.nonzero(as_tuple=True)[0]
        if idx.numel() == 0:
            raise ValueError(
                f"Diffraction order (m={m}, n={n}) not in the truncated harmonic set."
            )
        return int(idx.item())

    # -- s/p basis-change operator (per-harmonic 2x2, DIAG leaves) --------

    def _rotation(self, kz: torch.Tensor, n: torch.Tensor) -> Block2x2:
        """Build ``R(kz) = (1/rho) * [[-ky, -kz*kx/n], [kx, -kz*ky/n]]`` as a
        ``Block2x2`` of DIAG leaves over harmonics.

        Regularizes the ``rho -> 0`` (normal-incidence) singularity by
        falling back to the global azimuth ``phi0 = atan2(ky0, kx0)`` for
        the affected harmonic(s).
        """
        ls = self.modal.prepared.layersolver
        kx, ky = ls.kx, ls.ky
        rho = torch.sqrt(kx**2 + ky**2)
        near_zero = rho < _RHO_EPS
        rho_safe = torch.where(near_zero, torch.ones_like(rho), rho)

        model_spec = self.modal.prepared.model_spec
        phi0 = torch.atan2(model_spec.ky0, model_spec.kx0)[..., None]  # [...,1], broadcasts vs Nh

        cdtype = kz.dtype
        sx = torch.where(near_zero, -torch.sin(phi0), -ky / rho_safe).to(cdtype)
        sy = torch.where(near_zero, torch.cos(phi0), kx / rho_safe).to(cdtype)
        kz_over_n = kz / n
        px = torch.where(
            near_zero,
            -kz_over_n * torch.cos(phi0).to(cdtype),
            -kz_over_n * (kx / rho_safe).to(cdtype),
        )
        py = torch.where(
            near_zero,
            -kz_over_n * torch.sin(phi0).to(cdtype),
            -kz_over_n * (ky / rho_safe).to(cdtype),
        )

        return Block2x2(
            Block(Block.DIAG, sx), Block(Block.DIAG, px),
            Block(Block.DIAG, sy), Block(Block.DIAG, py),
        )

    def _incidence_rotation(self) -> Block2x2:
        """R_in: incidence medium, forward kz (+kz_inc) — shared by
        reflection and transmission (both are excited from the incidence
        side)."""
        if self._R_in is None:
            ls = self.modal.prepared.layersolver
            ops = self.modal.prepared.ops
            Nh = ls.m_flat.shape[0]
            kz_inc = ops[0].lam[..., :Nh] / 1j
            n_inc = self.modal.prepared.model_spec.incidence.refractive_index()
            n_inc = _broadcast_trailing(n_inc, kz_inc)
            self._R_in = self._rotation(kz_inc, n_inc)
        return self._R_in

    def _outgoing_rotation(self, kind: Kind) -> Block2x2:
        """R_out: reflection uses the incidence medium with kz flipped
        (backward-propagating); transmission uses the transmission medium,
        forward kz (no sign flip)."""
        if kind not in self._R_out:
            ls = self.modal.prepared.layersolver
            ops = self.modal.prepared.ops
            Nh = ls.m_flat.shape[0]
            if kind == "reflection":
                kz = -(ops[0].lam[..., :Nh] / 1j)  # incidence medium, backward
                n = self.modal.prepared.model_spec.incidence.refractive_index()
            elif kind == "transmission":
                kz = ops[-1].lam[..., :Nh] / 1j  # transmission medium, forward
                n = self.modal.prepared.model_spec.transmission.refractive_index()
            else:
                raise ValueError(f"kind must be 'reflection' or 'transmission', got {kind!r}")
            n = _broadcast_trailing(n, kz)
            self._R_out[kind] = self._rotation(kz, n)
        return self._R_out[kind]

    def _jones_operator(self, kind: Kind) -> Block2x2:
        """Full harmonic-space s/p Jones operator: ``R_out⁻¹ @ S_xy @ R_in``.

        Lazily cached per ``kind`` (built at most once per instance).
        """
        if kind not in self._jones_full:
            S_xy = self.modal.S11 if kind == "reflection" else self.modal.S21
            R_in = self._incidence_rotation()
            R_out = self._outgoing_rotation(kind)
            self._jones_full[kind] = R_out.solve(S_xy @ R_in)
        return self._jones_full[kind]

    # -- public API ----------------------------------------------------------

    def jones(
        self,
        kind: Kind,
        in_order: Tuple[int, int] = (0, 0),
        out_order: Tuple[int, int] = (0, 0),
    ) -> torch.Tensor:
        """2x2 s/p Jones matrix for one diffraction channel.

        Parameters
        ----------
        kind : {"reflection", "transmission"}
            Which scattering process to evaluate.
        in_order, out_order : tuple of int, default (0, 0)
            Incident / scattered diffraction order ``(m, n)``.

        Returns
        -------
        J : torch.Tensor
            Shape ``[..., 2, 2]``, ``[[J_ss, J_sp], [J_ps, J_pp]]``, such
            that ``(b_s, b_p) = J @ (a_s, a_p)``.
        """
        S_sp = self._jones_operator(kind)
        i_in, i_out = self._order_index(in_order), self._order_index(out_order)
        Jss, Jsp = _leaf_element(S_sp.a, i_out, i_in), _leaf_element(S_sp.b, i_out, i_in)
        Jps, Jpp = _leaf_element(S_sp.c, i_out, i_in), _leaf_element(S_sp.d, i_out, i_in)
        row_s = torch.stack([Jss, Jsp], dim=-1)
        row_p = torch.stack([Jps, Jpp], dim=-1)
        return torch.stack([row_s, row_p], dim=-2)  # [...,2,2]

    def scatter(
        self,
        excitation: torch.Tensor,
        kind: Kind,
        in_order: Tuple[int, int] = (0, 0),
        out_order: Tuple[int, int] = (0, 0),
    ) -> torch.Tensor:
        """Apply the s/p Jones matrix to a batched complex excitation.

        Parameters
        ----------
        excitation : torch.Tensor
            Complex ``(a_s, a_p)`` amplitudes, shape ``[..., 2]`` (or
            broadcastable to ``jones(...)``'s batch shape).
        kind, in_order, out_order
            See :meth:`jones`.

        Returns
        -------
        torch.Tensor
            Scattered ``(b_s, b_p)`` amplitudes, shape ``[..., 2]``.
        """
        J = self.jones(kind, in_order, out_order)
        return torch.einsum("...ij,...j->...i", J, excitation)

    def _unit_excitation(self, pol: Pol, dtype: torch.dtype, device) -> torch.Tensor:
        if pol == "s":
            return torch.tensor([1.0, 0.0], dtype=dtype, device=device)
        if pol == "p":
            return torch.tensor([0.0, 1.0], dtype=dtype, device=device)
        raise ValueError(f"pol must be 's' or 'p', got {pol!r}")

    def reflection(
        self,
        pol: Pol,
        in_order: Tuple[int, int] = (0, 0),
        out_order: Tuple[int, int] = (0, 0),
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Reflection amplitudes ``(rs, rp)`` for a pure-polarization
        incident wave.

        Parameters
        ----------
        pol : {"s", "p"}
            Incident polarization (unit excitation).
        in_order, out_order : tuple of int, default (0, 0)
            Incident / reflected diffraction order ``(m, n)``.

        Returns
        -------
        rs, rp : torch.Tensor
            Both scattered components (co- and cross-polarized).
        """
        J = self.jones("reflection", in_order, out_order)
        exc = self._unit_excitation(pol, J.dtype, J.device)
        rs, rp = torch.einsum("...ij,j->...i", J, exc).unbind(-1)
        return rs, rp

    def transmission(
        self,
        pol: Pol,
        in_order: Tuple[int, int] = (0, 0),
        out_order: Tuple[int, int] = (0, 0),
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Transmission amplitudes ``(ts, tp)`` for a pure-polarization
        incident wave. See :meth:`reflection`."""
        J = self.jones("transmission", in_order, out_order)
        exc = self._unit_excitation(pol, J.dtype, J.device)
        ts, tp = torch.einsum("...ij,j->...i", J, exc).unbind(-1)
        return ts, tp


class FieldObservables(Observables):
    """Field-level observables built from a :class:`FieldSolution`.

    Superset of :class:`Observables`: embeds the excitation-independent
    reflection/transmission API and adds field-reconstruction-dependent
    quantities. Not yet implemented — :class:`FieldSolution` and its field
    pipeline (``solve_fields``) don't exist yet; see
    ``metarcwa.solver.base.FieldSolution``.
    """

    def __init__(self, fields: FieldSolution):
        super().__init__(fields.modal)
        self._fields = fields

    def intensity(self, grid):
        raise NotImplementedError("Field reconstruction (solve_fields) is not implemented yet.")

    def absorption_map(self, grid):
        raise NotImplementedError("Field reconstruction (solve_fields) is not implemented yet.")
