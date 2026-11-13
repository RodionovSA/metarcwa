# metarcwa/solver/background.py
"""
background — incident field of the background stack (one harmonic)
====================================================================

Under a lateral PML the field is split into a known incident field and a
scattered field (``docs/PML.md`` "Illumination: scattered-field formulation").
The incident field ``E_inc`` is the exact solution of the **background
stack**: the same stack with every patterned layer filled with its
``medium_void``. That stack is planar, so an oblique plane wave stays a plane
wave with the incident in-plane wavevector ``(kx0, ky0)`` in every layer, and
one Fourier harmonic suffices. No PML is involved.

This module solves that one-harmonic problem with the main solver's own
building blocks, so the conventions cannot drift apart:

- field vector ``psi = (Sx, Sy, Ux, Uy)`` (``docs/rcwa_core.md``), modes of
  a uniform isotropic medium ``W = I``, ``V = Q diag(1/lam)``
  (:func:`~metarcwa.solver.layersolver.homogeneous.homogeneous_modes`);
- forward waves vary as ``exp(+lam k0 z)``, ``lam = 1j kz``;
- interfaces, propagation and composition are
  :func:`~metarcwa.solver.smatrix.S_boundary`,
  :func:`~metarcwa.solver.smatrix.S_prop` and :meth:`Block2x2.star` with
  one harmonic (DIAG leaves of size 1);
- excitation columns are the s and p unit amplitudes of
  :func:`~metarcwa.solver.polarization.sp_matrix`, the basis
  :class:`~metarcwa.observables.base.Observables` uses, so ``(a_s, a_p)``
  can be applied downstream exactly as today.

Amplitude reference planes. Inside finite layer ``i`` (local ``z`` in
``[0, d_i]``) the background field is

    psi(z) = [I; V] c+ exp(lam k0 z) + [I; -V] c- exp(-lam k0 (z - d_i)),

i.e. the forward amplitude ``c+`` is referenced to the left face and the
backward amplitude ``c-`` to the right face: each wave is referenced to the
face where it *enters* the layer, so both exponentials are bounded by ``1``
inside the layer (``Re lam <= 0``). In the half-spaces both amplitudes are
referenced to the interface.

What is specific to this module: building the background stack, and the
amplitudes inside every layer from its left and right partial S-matrices.
The calculation is a handful of ``2x2`` (polarization) operations per layer
and batch entry; it depends on the source only and is cheap, so it runs in
:func:`~metarcwa.solver.base.run`.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from metarcwa.model.base import ModelSpec
from metarcwa.model.layer import HomogeneousLayer, PatternedLayer
from metarcwa.model.medium import IsotropicMediumSpec, MediumSpec
from metarcwa.solver.blockmatrix import apply_vec, solve_vec
from metarcwa.solver.layersolver.homogeneous import homogeneous_modes
from metarcwa.solver.layersolver._modes import _regularize_eps
from metarcwa.solver.polarization import sp_matrix
from metarcwa.solver.smatrix import S_boundary, S_prop


@dataclass(frozen=True)
class BackgroundField:
    """Plane-wave solution of the background stack, for s and p excitation.

    Element index ``i`` follows ``PreparedStack.ops``: ``0`` is the incidence
    half-space, ``1..L`` the finite layers, ``L+1`` the transmission
    half-space. ``B`` below is the source batch shape
    ``[N_wl, N_theta, N_phi]``. The last axis of every amplitude is the
    excitation: column ``0`` is a unit s-polarized incident wave, column
    ``1`` a unit p-polarized one.

    Attributes
    ----------
    kx0, ky0 : torch.Tensor
        Incident in-plane wavevector (k0-normalized, **not** zeroed along
        PML axes), shape ``B``.
    wavelength : torch.Tensor
        Free-space wavelength, shape ``[N_wl, 1, 1]``.
    eps : tuple of torch.Tensor
        Regularized background permittivity of every element, broadcastable
        to ``B``.
    lam : tuple of torch.Tensor
        Forward modal exponent ``lam = 1j kz`` of every element, shape ``B``
        (the same for both polarizations of an isotropic medium).
    V : tuple of torch.Tensor
        ``2x2`` H-mode matrix of every element, shape ``[*B, 2, 2]``.
    c_plus, c_minus : tuple of torch.Tensor
        Forward / backward tangential-E amplitudes ``(Sx, Sy)``, shape
        ``[*B, 2, 2]`` (rows: field component, columns: s, p excitation).
        Referenced as described in the module docstring; ``c_minus`` of the
        transmission half-space is zero.
    thickness : tuple
        Layer thickness, ``None`` for the two half-spaces.
    """
    kx0: torch.Tensor
    ky0: torch.Tensor
    wavelength: torch.Tensor
    eps: tuple
    lam: tuple
    V: tuple
    c_plus: tuple
    c_minus: tuple
    thickness: tuple

    @property
    def excitation(self) -> torch.Tensor:
        """Incident tangential E ``(Sx, Sy)`` of the unit s and p waves,
        shape ``[*B, 2, 2]``."""
        return self.c_plus[0]

    def waves(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Full field amplitudes of the forward and backward wave in element ``i``.

        Returns
        -------
        plus, minus : torch.Tensor
            Shape ``[*B, 6, 2]``: rows ``(Ex, Ey, Ez, Ux, Uy, Uz)`` (``U``
            is the normalized magnetic field, ``H = j sqrt(eps0/mu0) U``),
            columns the s and p excitation. ``plus`` multiplies
            ``exp(lam k0 z)`` and ``minus`` multiplies
            ``exp(-lam k0 (z - d))`` (see the module docstring).
        """
        kx = self.kx0[..., None]
        ky = self.ky0[..., None]
        eps = self.eps[i][..., None] if self.eps[i].ndim else self.eps[i]
        out = []
        for c, sign in ((self.c_plus[i], 1.0), (self.c_minus[i], -1.0)):
            u = sign * (self.V[i] @ c)                       # [*B, 2, 2]
            sx, sy = c[..., 0, :], c[..., 1, :]
            ux, uy = u[..., 0, :], u[..., 1, :]
            # docs/rcwa_core.md Eqs. (33)-(34), uniform medium, no source.
            sz = 1j * (ky * ux - kx * uy) / eps
            uz = 1j * (ky * sx - kx * sy)
            out.append(torch.stack([sx, sy, sz, ux, uy, uz], dim=-2))
        return out[0], out[1]


def _medium_eps(element) -> torch.Tensor:
    """Background permittivity of one stack element (void for patterned layers)."""
    if isinstance(element, HomogeneousLayer):
        medium = element.medium
    elif isinstance(element, PatternedLayer):
        medium = element.medium_void
    elif isinstance(element, MediumSpec):
        medium = element
    else:
        raise TypeError(f"unsupported stack element {type(element)}")
    if not isinstance(medium, IsotropicMediumSpec):
        raise NotImplementedError(f"background solve not implemented for {type(medium)}")
    return medium.eps


def solve_background(model_spec: ModelSpec, grazing_eps_reg: float = 0.0,
                     thicknesses=None) -> BackgroundField:
    """Solve the background stack for a unit s and a unit p incident wave.

    Parameters
    ----------
    model_spec : ModelSpec
        Resolved model. Patterned layers contribute their ``medium_void``.
    grazing_eps_reg : float
        ``Config.grazing_eps_reg``, applied to every background permittivity
        exactly as the main solver does, so the background is the exact
        solution of the regularized problem the solver solves.
    thicknesses : sequence of torch.Tensor, optional
        Thickness of every finite layer, overriding ``model_spec.layers``.
        :func:`~metarcwa.solver.base.run` passes the thicknesses held by the
        prepared operators, which are the ones the layer S-matrices use
        (they differ after ``dataclasses.replace(op, thickness=...)``).

    Returns
    -------
    BackgroundField
    """
    elements = (model_spec.incidence, *model_spec.layers, model_spec.transmission)
    kx0, ky0 = model_spec.kx0, model_spec.ky0                  # B
    wvl = model_spec.wavelength

    # One harmonic: modes with DIAG leaves of size 1 (W = I, V = Q diag(1/lam)).
    eps_l, lam_l, W_l, V_l, thick_l = [], [], [], [], []
    for el in elements:
        eps = _regularize_eps(_medium_eps(el), grazing_eps_reg).reshape(-1, 1, 1)
        lam, V = homogeneous_modes(eps, kx0[..., None], ky0[..., None])
        eps_l.append(eps)
        lam_l.append(lam)                                       # [*B, 2]
        W_l.append(V.eye_like())
        V_l.append(V)
        thick_l.append(None if isinstance(el, MediumSpec) else el.thickness)
    if thicknesses is not None:
        if len(thicknesses) != len(model_spec.layers):
            raise ValueError("thicknesses must have one entry per finite layer")
        thick_l[1:-1] = list(thicknesses)

    n_el = len(elements)
    L = n_el - 2
    I = [S_boundary(W_l[i], V_l[i], W_l[i + 1], V_l[i + 1]) for i in range(n_el - 1)]
    P = [None] + [S_prop(lam_l[i], wvl, thick_l[i]) for i in range(1, L + 1)] + [None]

    # SL[i]: everything left of element i's left face; SR[i]: right of its right face.
    SL = [None] * n_el
    SL[1] = I[0]
    for i in range(1, L + 1):
        SL[i + 1] = SL[i].star(P[i]).star(I[i])
    SR = [None] * n_el
    SR[L] = I[L]
    for i in range(L, 0, -1):
        SR[i - 1] = I[i - 1].star(P[i]).star(SR[i])
    total = SL[L + 1]

    kz_inc = lam_l[0][..., 0] / 1j
    n_inc = torch.sqrt(model_spec.incidence.eps).reshape(-1, 1, 1)
    a = sp_matrix(kx0, ky0, kz_inc, n_inc, torch.atan2(ky0, kx0))   # [*B, 2, 2]

    c_plus, c_minus = [a], [apply_vec(total.a, a)]
    for i in range(1, L + 1):
        X = P[i].b                                              # diag exp(lam k0 d)
        RL, RR = SL[i].d, SR[i].a
        M = RL @ X @ RR @ X
        cp = solve_vec(M.eye_like() - M, apply_vec(SL[i].c, a))     # c+ at the left face
        c_plus.append(cp)
        c_minus.append(apply_vec(RR, apply_vec(X, cp)))          # c- at the right face
    c_plus.append(apply_vec(total.c, a))
    c_minus.append(torch.zeros_like(c_plus[-1]))

    return BackgroundField(
        kx0=kx0, ky0=ky0, wavelength=wvl.reshape(-1, 1, 1),
        eps=tuple(eps_l), lam=tuple(l[..., 0] for l in lam_l),
        V=tuple(V.to_dense(1) for V in V_l),
        c_plus=tuple(c_plus), c_minus=tuple(c_minus), thickness=tuple(thick_l),
    )
