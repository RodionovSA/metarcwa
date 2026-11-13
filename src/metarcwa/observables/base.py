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

With a lateral PML the half-space mode matrix ``W`` is a dense eigenvector
matrix, so ``S11``/``S21`` are mapped to the field basis first
(``W_out · S · W_in⁻¹``). Only the specular order ``(0, 0)`` is available
then: it is the one harmonic that is still an exact plane-wave mode of the
half-spaces. For an isolated structure it holds the specular background plus
the cell-averaged scattered field.

PML scattered-field formulation
-------------------------------
With ``Config.pml_formulation == "scattered"`` (the PML default) the
solution carries the background (incident) field and the scattered waves
``e_L``/``e_R`` leaving the stack (``docs/PML.md``). The scattered field is
purely outgoing and is **not** a set of diffraction orders: it is a field on
the face of the stack, which is reconstructed on the real-space grid. The
observables are then:

- :meth:`Observables.jones` / ``reflection`` / ``transmission`` (specular
  order only): background plus the scattered field projected on the
  incident plane wave ``exp(j k_par . r)`` and averaged over the cell
  (``region="cell"``, default; equals the total-field ``(0, 0)`` order at
  normal incidence) or integrated over the physical region only and divided
  by the cell area (``region="physical"``). For an isolated structure the
  scattered part scales as ``1 / cell area``; ``area * (J - J_bg)`` is the
  cell-independent specular scattering amplitude.
- :meth:`Observables.background_jones`: the background stack alone.
- :meth:`Observables.scattered_power`: power of the scattered field leaving
  through one face, normalized to the incident power on one cell.
- :meth:`Observables.extinction`: power taken from the incident field
  (scattered + absorbed), from the interference term on both faces.
"""

from typing import Literal, Tuple

import torch

from metarcwa.solver.base import ModalSolution, FieldSolution
from metarcwa.solver.polarization import sp_matrix, sp_vectors
from metarcwa.solver.blockmatrix import Block, Block2x2, apply_vec
from metarcwa.solver.layersolver.homogeneous import homogeneous_kz
from metarcwa.solver.layersolver._modes import _regularize_eps
from metarcwa.solver.scattered import _grid_phase

Kind = Literal["reflection", "transmission"]
Pol = Literal["s", "p"]
Region = Literal["cell", "physical"]


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
        the affected harmonic(s). Same basis as the background solve
        (:func:`~metarcwa.solver.polarization.sp_vectors`).
        """
        ls = self.modal.prepared.layersolver
        model_spec = self.modal.prepared.model_spec
        phi0 = torch.atan2(model_spec.ky0, model_spec.kx0)[..., None]  # [...,1], broadcasts vs Nh
        sx, sy, px, py = sp_vectors(ls.kx, ls.ky, kz, n, phi0)
        return Block2x2(
            Block(Block.DIAG, sx), Block(Block.DIAG, px),
            Block(Block.DIAG, sy), Block(Block.DIAG, py),
        )

    def _kz(self, which: int) -> torch.Tensor:
        """Forward kz per harmonic in a half-space, shape ``[..., Nh]``.

        ``which`` is the ``ops`` index: ``0`` incidence, ``-1`` transmission.
        Under a PML the half-space modes come in eigensolver order, not
        harmonic order, so the closed-form kz is used instead; it is exact
        for the specular order, the only one available then.
        """
        prepared = self.modal.prepared
        ls = prepared.layersolver
        Nh = ls.m_flat.shape[0]
        if not ls.stretched:
            return prepared.ops[which].lam[..., :Nh] / 1j
        medium = prepared.model_spec.incidence if which == 0 else prepared.model_spec.transmission
        eps = _regularize_eps(medium.eps, ls.config.grazing_eps_reg)
        return homogeneous_kz(eps, ls.kx, ls.ky)[..., :Nh]

    def _incidence_rotation(self) -> Block2x2:
        """R_in: incidence medium, forward kz (+kz_inc) — shared by
        reflection and transmission (both are excited from the incidence
        side)."""
        if self._R_in is None:
            kz_inc = self._kz(0)
            n_inc = self.modal.prepared.model_spec.incidence.refractive_index()
            n_inc = _broadcast_trailing(n_inc, kz_inc)
            self._R_in = self._rotation(kz_inc, n_inc)
        return self._R_in

    def _outgoing_rotation(self, kind: Kind) -> Block2x2:
        """R_out: reflection uses the incidence medium with kz flipped
        (backward-propagating); transmission uses the transmission medium,
        forward kz (no sign flip)."""
        if kind not in self._R_out:
            if kind == "reflection":
                kz = -self._kz(0)                  # incidence medium, backward
                n = self.modal.prepared.model_spec.incidence.refractive_index()
            elif kind == "transmission":
                kz = self._kz(-1)                  # transmission medium, forward
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
            if self.modal.prepared.layersolver.stretched:
                # S acts on half-space mode amplitudes: fields -> modes -> fields.
                ops = self.modal.prepared.ops
                W_in = ops[0].W
                W_out = ops[0].W if kind == "reflection" else ops[-1].W
                S_R_in = W_out @ (S_xy @ W_in.solve(R_in))
            else:
                S_R_in = S_xy @ R_in
            self._jones_full[kind] = R_out.solve(S_R_in)
        return self._jones_full[kind]

    # -- public API ----------------------------------------------------------

    def jones(
        self,
        kind: Kind,
        in_order: Tuple[int, int] = (0, 0),
        out_order: Tuple[int, int] = (0, 0),
        region: Region = "cell",
    ) -> torch.Tensor:
        """2x2 s/p Jones matrix for one diffraction channel.

        Parameters
        ----------
        kind : {"reflection", "transmission"}
            Which scattering process to evaluate.
        in_order, out_order : tuple of int, default (0, 0)
            Incident / scattered diffraction order ``(m, n)``.
        region : {"cell", "physical"}, default "cell"
            PML scattered-field formulation only: where the scattered field
            is projected on the specular plane wave (see the module
            docstring). Ignored otherwise.

        Returns
        -------
        J : torch.Tensor
            Shape ``[..., 2, 2]``, ``[[J_ss, J_sp], [J_ps, J_pp]]``, such
            that ``(b_s, b_p) = J @ (a_s, a_p)``.
        """
        if self.modal.prepared.layersolver.stretched and (in_order, out_order) != ((0, 0), (0, 0)):
            raise ValueError(
                "With a PML only the specular order (0, 0) is defined; other "
                "harmonics are not plane-wave modes of the half-spaces."
            )
        if self.modal.background is not None:
            bg_t = self._background_face(kind)[..., :2, :]
            sc_t = self._project_specular(kind, region)
            return self._sp_solve(kind, bg_t + sc_t)
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


    # -- PML scattered-field formulation ------------------------------------

    def _require_scattered(self) -> None:
        if self.modal.background is None:
            raise ValueError(
                "Only available for the PML scattered-field formulation "
                "(a Stack with a PML and Config(pml_formulation='scattered'))."
            )

    def _sp_solve(self, kind: Kind, tangential: torch.Tensor) -> torch.Tensor:
        """Map outgoing tangential E ``(Sx, Sy)`` (``[..., 2, ncol]``) of the
        specular plane wave to s/p amplitudes."""
        spec = self.modal.prepared.model_spec
        bg = self.modal.background
        if kind == "reflection":
            i, sign, medium = 0, -1.0, spec.incidence
        elif kind == "transmission":
            i, sign, medium = -1, 1.0, spec.transmission
        else:
            raise ValueError(f"kind must be 'reflection' or 'transmission', got {kind!r}")
        n = torch.sqrt(medium.eps).reshape(-1, 1, 1)
        R = sp_matrix(bg.kx0, bg.ky0, sign * bg.lam[i] / 1j, n, torch.atan2(bg.ky0, bg.kx0))
        return torch.linalg.solve(R, tangential.to(R.dtype))

    def _background_face(self, kind: Kind) -> torch.Tensor:
        """Outgoing background plane wave at the face, ``[..., 4, 2]`` rows
        ``(Sx, Sy, Ux, Uy)``, columns s, p excitation."""
        bg = self.modal.background
        if kind == "reflection":
            c, V, sign = bg.c_minus[0], bg.V[0], -1.0
        elif kind == "transmission":
            c, V, sign = bg.c_plus[-1], bg.V[-1], 1.0
        else:
            raise ValueError(f"kind must be 'reflection' or 'transmission', got {kind!r}")
        return torch.cat([c, sign * (V @ c)], dim=-2)

    def background_jones(self, kind: Kind) -> torch.Tensor:
        """s/p Jones matrix of the background stack alone (specular, ``[..., 2, 2]``).

        The background is the stack with every patterned layer filled with
        its void medium (``metarcwa.solver.background``).
        """
        self._require_scattered()
        return self._sp_solve(kind, self._background_face(kind)[..., :2, :])

    def scattered_coefficients(self, kind: Kind) -> Tuple[torch.Tensor, torch.Tensor]:
        """Harmonic coefficients of the scattered field on one face.

        Returns
        -------
        s, u : torch.Tensor
            Tangential ``(Sx, Sy)`` and ``(Ux, Uy)`` Fourier coefficients,
            shape ``[..., 2Nh, 2]`` (columns: unit s, p incident wave). Along
            PML axes the Bloch phase is zero; along periodic axes the
            coefficients multiply the usual Bloch harmonics.
        """
        self._require_scattered()
        ops = self.modal.prepared.ops
        if kind == "reflection":
            op, e, sign = ops[0], self.modal.e_L, -1.0      # backward waves
        elif kind == "transmission":
            op, e, sign = ops[-1], self.modal.e_R, 1.0
        else:
            raise ValueError(f"kind must be 'reflection' or 'transmission', got {kind!r}")
        return apply_vec(op.W, e), sign * apply_vec(op.V, e)

    def _grid(self, coeffs: torch.Tensor) -> torch.Tensor:
        """Real-space field ``[..., ncol, Ny, Nx]`` from ``[..., Nh, ncol]``
        coefficients (Bloch phase of periodic axes omitted)."""
        ls = self.modal.prepared.layersolver
        nx, ny = ls.config.nx, ls.config.ny
        c = coeffs.transpose(-1, -2)
        grid = torch.zeros(*c.shape[:-1], ny, nx, dtype=c.dtype, device=c.device)
        grid[..., ls.n_flat % ny, ls.m_flat % nx] = c
        return torch.fft.ifft2(grid, norm="forward")

    def scattered_field(self, kind: Kind) -> torch.Tensor:
        """Scattered tangential fields on one face, on the real-space grid.

        Returns
        -------
        torch.Tensor
            Shape ``[..., 4, 2, Ny, Nx]``: components ``(Ex, Ey, Ux, Uy)``
            (``H = j sqrt(eps0/mu0) U``), columns s, p excitation. Same
            pixel indexing as the pattern grid; the solver's phase
            convention puts pixel ``(j, i)`` at ``(y, x) = (j py/Ny, i px/Nx)``
            (``exp(j k_par . r)`` of the source uses it). Along a
            periodic axis the Bloch factor ``exp(j k_par . r)`` is omitted
            (it is common to every field and drops out of fluxes).
        """
        s, u = self.scattered_coefficients(kind)
        Nh = s.shape[-2] // 2
        comps = [s[..., :Nh, :], s[..., Nh:, :], u[..., :Nh, :], u[..., Nh:, :]]
        return torch.stack([self._grid(c) for c in comps], dim=-4)

    def _mask(self, region: Region) -> torch.Tensor:
        ls = self.modal.prepared.layersolver
        nx, ny = ls.config.nx, ls.config.ny
        if region == "cell":
            return torch.ones(ny, nx, dtype=torch.bool, device=ls.m_flat.device)
        if region == "physical":
            return self.modal.prepared.model_spec.pml.physical_mask(ny, nx, ls.m_flat.device)
        raise ValueError(f"region must be 'cell' or 'physical', got {region!r}")

    def _phase(self) -> torch.Tensor:
        """``exp(j k_par . r)`` on the grid along PML axes, ``[..., Ny, Nx]``."""
        ls = self.modal.prepared.layersolver
        spec = self.modal.prepared.model_spec
        bg = self.modal.background
        return _grid_phase(spec.pml, bg.kx0, bg.ky0, 2 * torch.pi / bg.wavelength,
                           spec.a1, spec.a2, ls.config.ny, ls.config.nx)

    def _project_specular(self, kind: Kind, region: Region) -> torch.Tensor:
        """Scattered tangential E projected on the specular plane wave, ``[..., 2, 2]``.

        ``(1/A) ∫_region E_s exp(-j k_par . r) dA`` with ``A`` the cell area
        (grid sum, consistent with the FFT convention of the solver).
        """
        f = self.scattered_field(kind)[..., :2, :, :, :]         # [..., 2, 2, Ny, Nx]
        w = self._mask(region) * self._phase().conj()[..., None, None, :, :]
        return (f * w).mean(dim=(-2, -1))

    @staticmethod
    def _flux(sx, sy, ux, uy, sx2=None, sy2=None, ux2=None, uy2=None) -> torch.Tensor:
        """z-flux density ``(1/2) Im(Sx Uy* - Sy Ux*)`` (units of
        ``sqrt(eps0/mu0)``); with a second field, the symmetric cross term."""
        if sx2 is None:
            return 0.5 * (sx * uy.conj() - sy * ux.conj()).imag
        return 0.5 * (sx * uy2.conj() - sy * ux2.conj()
                      + sx2 * uy.conj() - sy2 * ux.conj()).imag

    def _incident_flux(self) -> torch.Tensor:
        """z-flux density of the unit s and p incident waves, ``[..., 2]``."""
        plus, _ = self.modal.background.waves(0)
        return self._flux(plus[..., 0, :], plus[..., 1, :], plus[..., 3, :], plus[..., 4, :])

    def scattered_power(self, kind: Kind, region: Region = "physical") -> torch.Tensor:
        """Scattered power leaving through one face.

        Real-space Poynting flux of the scattered field through the
        incidence-side (``"reflection"``) or transmission-side face,
        integrated over ``region`` and normalized to the incident power on
        one cell (``I_inc * A``). Multiply by the cell area for a cross
        section. Power that leaves laterally (e.g. guided in a film, or
        crossing the face inside the PML for ``region="physical"``) is not
        counted.

        Returns
        -------
        torch.Tensor
            Shape ``[..., 2]``: unit s and p incident wave.
        """
        self._require_scattered()
        sx, sy, ux, uy = self.scattered_field(kind).unbind(-4)    # [..., 2, Ny, Nx]
        sign = -1.0 if kind == "reflection" else 1.0
        density = sign * self._flux(sx, sy, ux, uy) * self._mask(region)
        return density.mean(dim=(-2, -1)) / self._incident_flux()

    def extinction(self, region: Region = "physical") -> torch.Tensor:
        """Extinguished power (scattered + absorbed by the structure).

        Minus the outward flux of the interference term between the
        incident (background) and scattered fields through both faces,
        integrated over ``region``, normalized as :meth:`scattered_power`.
        Exact when the scattered field leaves only through the two faces;
        the interference flux through the lateral boundaries of the region
        (e.g. a mode guided away in a film) is not included.

        Returns
        -------
        torch.Tensor
            Shape ``[..., 2]``: unit s and p incident wave.
        """
        self._require_scattered()
        bg = self.modal.background
        mask = self._mask(region)
        phase = self._phase()[..., None, :, :]                   # [..., 1, Ny, Nx]
        total = 0.0
        for kind, i, sign in (("reflection", 0, -1.0), ("transmission", -1, 1.0)):
            plus, minus = bg.waves(i)
            w = plus + minus                                     # both referenced to the face
            inc = [w[..., r, :, None, None] * phase for r in (0, 1, 3, 4)]
            sca = self.scattered_field(kind).unbind(-4)
            cross = self._flux(*sca, *inc) * mask
            total = total + sign * cross.mean(dim=(-2, -1))
        return -total / self._incident_flux()

    def scattering_pattern(
        self,
        kind: Kind,
        theta,
        phi=None,
        region: Region = "physical",
        periodic_order: int = 0,
        polarization_resolved: bool = False,
    ) -> torch.Tensor:
        """Angular scattering pattern (differential cross section) of the scattered field.

        Uses the angular spectrum of the scattered tangential E on one face,
        ``Ẽ(k_par) = ∫_region E_s exp(-j k_par . r) dA``. Each propagating
        ``k_par`` is a plane wave leaving the stack into the half-space (index
        ``n``) in the direction ``sin(theta) = |k_par| / (n k0)``; its s/p
        amplitudes come from the same basis as :meth:`jones`. ``theta`` is
        measured from the outward normal of the face (``-z`` for
        ``"reflection"``, ``+z`` for ``"transmission"``). The half-space must
        be lossless (``Re n`` is used).

        - **PML on both axes** (isolated 3D structure): ``theta``, ``phi``
          (polar angle, azimuth) give ``dsigma/dOmega`` in units of
          area/steradian. Integrated over the hemisphere it equals
          ``scattered_power(kind, region) * cell area``.
        - **PML on one axis** (structure uniform or periodic along the other
          axis): ``theta`` is the angle in the plane spanned by the PML axis
          and ``z`` (inside the diffraction cone of ``periodic_order`` along
          the periodic axis when the incidence is tilted along it); ``phi``
          must be ``None``. Returns ``dsigma'/dtheta`` per unit length along
          the periodic axis (units of length/radian). Summed over the
          propagating ``periodic_order`` values and integrated over
          ``theta`` in ``(-pi/2, pi/2)`` it equals
          ``scattered_power(kind, region) * (cell length along the PML axis)``.

        Parameters
        ----------
        kind : {"reflection", "transmission"}
            Face: scattering back into the incidence medium, or into the
            transmission medium.
        theta : float or torch.Tensor
            Polar angle(s) in rad, any shape (flattened to ``N_ang``).
        phi : float or torch.Tensor, optional
            Azimuth(s) in rad, broadcast against ``theta``; both-axes PML
            only (default ``0``).
        region : {"physical", "cell"}, default "physical"
            Where the field is integrated. ``"physical"`` excludes the PML,
            where the field is the stretched (non-physical) continuation; the
            near field must have decayed before the PML.
        periodic_order : int, default 0
            One-axis PML only: diffraction order along the periodic axis.
        polarization_resolved : bool, default False
            If ``True``, return the s and p parts of the scattered wave
            separately.

        Returns
        -------
        torch.Tensor
            Shape ``[..., N_ang, 2]`` (last axis: unit s, p incident wave),
            or ``[..., N_ang, 2, 2]`` (scattered s/p, then incident s/p) with
            ``polarization_resolved``. Normalized to the incident intensity,
            so the values are cross sections, not power fractions.

        Notes
        -----
        The angular resolution is limited by the size of the integration
        region (``Δk ~ 2π / width``): narrow lobes need a large cell.
        """
        self._require_scattered()
        prepared = self.modal.prepared
        spec, ls, bg = prepared.model_spec, prepared.layersolver, self.modal.background
        pml = spec.pml
        if kind == "reflection":
            medium, sign = spec.incidence, -1.0
        elif kind == "transmission":
            medium, sign = spec.transmission, 1.0
        else:
            raise ValueError(f"kind must be 'reflection' or 'transmission', got {kind!r}")

        rdt = bg.kx0.dtype
        dev = bg.kx0.device
        nx, ny = ls.config.nx, ls.config.ny
        px, py = spec.a1[0].abs().to(rdt), spec.a2[1].abs().to(rdt)
        k0 = (2 * torch.pi / bg.wavelength).to(rdt)                   # [N_wl, 1, 1]
        n_c = torch.sqrt(medium.eps)
        if (n_c.imag.abs() > 1e-6 * n_c.abs()).any():
            raise ValueError(
                f"scattering_pattern('{kind}') needs a lossless half-space: "
                "there is no far field in an absorbing medium."
            )
        nr = n_c.real.reshape(-1, 1, 1).to(rdt)                       # [N_wl, 1, 1]
        th = torch.as_tensor(theta, dtype=rdt, device=dev).reshape(-1)
        both = pml.sx is not None and pml.sy is not None

        if both:
            ph = torch.zeros_like(th) if phi is None else torch.as_tensor(phi, dtype=rdt, device=dev).reshape(-1)
            th, ph = torch.broadcast_tensors(th, ph)
            kxn = nr[..., None] * torch.sin(th) * torch.cos(ph)       # [N_wl,1,1,N_ang]
            kyn = nr[..., None] * torch.sin(th) * torch.sin(ph)
            phi0 = ph
            qx, qy = k0[..., None] * kxn, k0[..., None] * kyn
            propagating = torch.ones_like(kxn, dtype=torch.bool)
        else:
            if phi is not None:
                raise ValueError("phi is only defined when the PML acts on both axes.")
            along_x = pml.sx is not None
            G = 2 * torch.pi * periodic_order / (py if along_x else px)
            k_par0 = (bg.ky0 if along_x else bg.kx0) + G / k0         # periodic-axis k, normalized
            kperp2 = nr ** 2 - k_par0 ** 2
            propagating = (kperp2 > 0)[..., None].expand(*kperp2.shape, th.numel())
            kperp = torch.sqrt(kperp2.clamp_min(0))[..., None]       # [*B, 1]
            k_pml = kperp * torch.sin(th)                             # [*B, N_ang]
            k_per = k_par0[..., None].expand_as(k_pml)
            kxn, kyn = (k_pml, k_per) if along_x else (k_per, k_pml)
            phi0 = torch.zeros_like(kxn)
            # The grid omits the Bloch phase of the periodic axis: only G remains there.
            q_pml = k0[..., None] * k_pml
            q_per = torch.full_like(q_pml, float(G))
            qx, qy = (q_pml, q_per) if along_x else (q_per, q_pml)

        kzn = torch.sqrt((nr[..., None] ** 2 - kxn ** 2 - kyn ** 2).clamp_min(0))
        cos_t = kzn / nr[..., None]

        # Angular spectrum: ∫ E exp(-j(qx x + qy y)) dA, separable in x and y.
        f = self.scattered_field(kind)[..., :2, :, :, :] * self._mask(region)  # [...,2c,2e,Ny,Nx]
        x = torch.arange(nx, dtype=rdt, device=dev) * (px / nx)
        y = torch.arange(ny, dtype=rdt, device=dev) * (py / ny)
        Ex = torch.exp(-1j * qx[..., None, :] * x[:, None])            # [*B', Nx, N_ang]
        Ey = torch.exp(-1j * qy[..., None, :] * y[:, None])            # [*B', Ny, N_ang]
        fx = f @ Ex[..., None, None, :, :].to(f.dtype)                 # [..., 2c, 2e, Ny, N_ang]
        Et = (fx * Ey[..., None, None, :, :].to(f.dtype)).sum(-2) * (px * py / (nx * ny))
        Et = Et.movedim(-1, -3)                                        # [..., N_ang, 2c, 2e]

        # s/p amplitudes of each outgoing plane wave. kz is floored so that the
        # p column of the basis stays invertible at grazing (its weight -> 0).
        kz = (sign * kzn.clamp_min(1e-9 * float(nr.min()))).to(Et.dtype)
        R = sp_matrix(kxn, kyn, kz, nr[..., None].to(Et.dtype), phi0)  # [*B', N_ang, 2, 2]
        a2 = torch.linalg.solve(R, Et).abs() ** 2                      # [..., N_ang, 2s/p, 2e]

        flux = 0.5 * nr[..., None, None, None] * cos_t[..., None, None] * a2   # per (2π)^-d d^dk
        if both:
            dens = (nr * k0)[..., None, None, None] ** 2 * cos_t[..., None, None] * flux / (2 * torch.pi) ** 2
        else:
            per_len = py if pml.sx is not None else px
            kperp_phys = (k0[..., None] * kperp)[..., None, None]     # [*B, 1, 1, 1]
            dens = kperp_phys * torch.cos(th)[..., None, None] * flux / (2 * torch.pi * per_len ** 2)
        dens = dens * propagating[..., None, None]
        inc = self._incident_flux()[..., None, None, :]                # [..., 1, 1, 2e]
        dens = dens / inc
        return dens if polarization_resolved else dens.sum(-2)


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
