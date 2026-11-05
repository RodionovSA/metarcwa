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

This module solves that one-harmonic problem with the same conventions as
the main solver:

- field vector ``psi = (Sx, Sy, Ux, Uy)`` (``docs/rcwa_core.md``), modes of
  a uniform isotropic medium ``W = I``, ``V = Q diag(1/lam)``
  (:func:`~metarcwa.solver.layersolver.homogeneous.homogeneous_modes`);
- forward waves vary as ``exp(+lam k0 z)``, ``lam = 1j kz``;
- excitation columns are the s and p unit amplitudes of the
  :class:`~metarcwa.observables.base.Observables` basis, so ``(a_s, a_p)``
  can be applied downstream exactly as today.

Amplitude reference planes. Inside finite layer ``i`` (local ``z`` in
``[0, d_i]``) the background field is

    psi(z) = [I; V] c+ exp(lam k0 z) + [I; -V] c- exp(-lam k0 (z - d_i)),

i.e. the forward amplitude ``c+`` is referenced to the left face and the
backward amplitude ``c-`` to the right face: each wave is referenced to the
face where it *enters* the layer, so both exponentials are bounded by ``1``
inside the layer (``Re lam <= 0``). In the half-spaces both amplitudes are
referenced to the interface.

The whole calculation is a handful of ``2x2`` (polarization) matrix
operations per layer and batch entry; it depends on the source only and is
cheap, so it runs in :func:`~metarcwa.solver.base.run`.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from metarcwa.model.base import ModelSpec
from metarcwa.model.layer import HomogeneousLayer, PatternedLayer
from metarcwa.model.medium import IsotropicMediumSpec, MediumSpec
from metarcwa.solver.layersolver.homogeneous import homogeneous_modes
from metarcwa.solver.layersolver._modes import _regularize_eps

# Same threshold as observables.base._RHO_EPS (normal-incidence fallback of
# the s/p basis); duplicated to avoid a solver -> observables import.
_RHO_EPS = 1e-8


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


def _sp_basis(kx: torch.Tensor, ky: torch.Tensor, kz: torch.Tensor,
              n: torch.Tensor, phi0: torch.Tensor) -> torch.Tensor:
    """Tangential E of the unit s and p waves, ``[*B, 2, 2]`` (columns s, p).

    Same basis as ``Observables._rotation``: ``s = (-ky, kx)/rho`` and the
    tangential part of ``p`` is ``-(kz/n) (kx, ky)/rho``, with the global
    azimuth ``phi0`` as the fallback direction when ``rho -> 0``.
    """
    rho = torch.sqrt(kx ** 2 + ky ** 2)
    near_zero = rho < _RHO_EPS
    rho_safe = torch.where(near_zero, torch.ones_like(rho), rho)
    ux = torch.where(near_zero, torch.cos(phi0), kx / rho_safe).to(kz.dtype)
    uy = torch.where(near_zero, torch.sin(phi0), ky / rho_safe).to(kz.dtype)
    kz_n = kz / n
    col_s = torch.stack([-uy, ux], dim=-1)
    col_p = torch.stack([-kz_n * ux, -kz_n * uy], dim=-1)
    return torch.stack([col_s, col_p], dim=-1)


# -- 2x2-block S-matrices as (S11, S12, S21, S22) tuples of [*B, 2, 2] ------

def _interface(Va: torch.Tensor, Vb: torch.Tensor) -> tuple:
    """S-matrix of the interface a | b between two uniform media (``W = I``).

    Continuity of ``psi`` gives
    ``[[I, -I], [-Va, -Vb]] (c-_a, c+_b) = [[-I, I], [-Va, -Vb]] (c+_a, c-_b)``
    (the one-harmonic form of :func:`~metarcwa.solver.smatrix.S_boundary`).
    """
    batch = torch.broadcast_shapes(Va.shape[:-2], Vb.shape[:-2])
    Va, Vb = Va.expand(*batch, 2, 2), Vb.expand(*batch, 2, 2)
    eye = torch.eye(2, dtype=Va.dtype, device=Va.device).expand(*batch, 2, 2)
    left = torch.cat([torch.cat([eye, -eye], -1), torch.cat([-Va, -Vb], -1)], -2)
    right = torch.cat([torch.cat([-eye, eye], -1), torch.cat([-Va, -Vb], -1)], -2)
    X = torch.linalg.solve(left, right)
    return X[..., :2, :2], X[..., :2, 2:], X[..., 2:, :2], X[..., 2:, 2:]


def _propagation(x: torch.Tensor) -> tuple:
    """S-matrix ``[[0, X], [X, 0]]`` of a uniform layer, ``x = exp(lam k0 d)`` of shape ``B``."""
    X = x[..., None, None] * torch.eye(2, dtype=x.dtype, device=x.device)
    Z = torch.zeros_like(X)
    return Z, X, X, Z


def _star(A: tuple, B: tuple) -> tuple:
    """Redheffer star product of two ``2x2``-block S-matrices (A left of B)."""
    a11, a12, a21, a22 = A
    b11, b12, b21, b22 = B
    eye = torch.eye(2, dtype=a11.dtype, device=a11.device)
    IP = eye - b11 @ a22
    IQ = eye - a22 @ b11
    Za = torch.linalg.solve(IP, b11 @ a21)
    Zb = torch.linalg.solve(IP, b12)
    Wc = torch.linalg.solve(IQ, a21)
    Wd = torch.linalg.solve(IQ, a22 @ b12)
    return a11 + a12 @ Za, a12 @ Zb, b21 @ Wc, b22 + b21 @ Wd


def solve_background(model_spec: ModelSpec, grazing_eps_reg: float = 0.0) -> BackgroundField:
    """Solve the background stack for a unit s and a unit p incident wave.

    Parameters
    ----------
    model_spec : ModelSpec
        Resolved model. Patterned layers contribute their ``medium_void``.
    grazing_eps_reg : float
        ``Config.grazing_eps_reg``, applied to every background permittivity
        exactly as the main solver does, so the background is the exact
        solution of the regularized problem the solver solves.

    Returns
    -------
    BackgroundField
    """
    elements = (model_spec.incidence, *model_spec.layers, model_spec.transmission)
    kx0, ky0 = model_spec.kx0, model_spec.ky0                  # B
    wvl = model_spec.wavelength.reshape(-1, 1, 1)
    k0 = 2 * torch.pi / wvl

    eps_l, lam_l, V_l, thick_l = [], [], [], []
    for el in elements:
        eps = _regularize_eps(_medium_eps(el), grazing_eps_reg).reshape(-1, 1, 1)
        lam, Vb = homogeneous_modes(eps, kx0[..., None], ky0[..., None])
        eps_l.append(eps)
        lam_l.append(lam[..., 0])                               # B
        V_l.append(Vb.to_dense(1))                              # [*B, 2, 2]
        thick_l.append(getattr(el, "thickness", None) if not isinstance(el, MediumSpec) else None)

    n_el = len(elements)
    L = n_el - 2
    x = [None] + [torch.exp(lam_l[i] * k0 * thick_l[i]) for i in range(1, L + 1)] + [None]
    I = [_interface(V_l[i], V_l[i + 1]) for i in range(n_el - 1)]   # I[i] = i | i+1

    # SL[i]: everything left of element i's left face; SR[i]: right of its right face.
    SL = [None] * n_el
    SL[1] = I[0]
    for i in range(1, L + 1):
        SL[i + 1] = _star(_star(SL[i], _propagation(x[i])), I[i])
    SR = [None] * n_el
    SR[L] = I[L]
    for i in range(L, 0, -1):
        SR[i - 1] = _star(_star(I[i - 1], _propagation(x[i])), SR[i])
    total = SL[L + 1]

    kz_inc = lam_l[0] / 1j
    n_inc = torch.sqrt(model_spec.incidence.eps).reshape(-1, 1, 1)
    phi0 = torch.atan2(ky0, kx0)
    a = _sp_basis(kx0, ky0, kz_inc, n_inc, phi0)               # [*B, 2, 2]

    eye = torch.eye(2, dtype=a.dtype, device=a.device)
    c_plus, c_minus = [a], [total[0] @ a]
    for i in range(1, L + 1):
        X = x[i][..., None, None] * eye
        t = SL[i][2] @ a
        RL, RR = SL[i][3], SR[i][0]
        cp = torch.linalg.solve(eye - RL @ X @ RR @ X, t)     # c+ at the left face
        c_plus.append(cp)
        c_minus.append(RR @ (X @ cp))                          # c- at the right face
    c_plus.append(total[2] @ a)
    c_minus.append(torch.zeros_like(c_plus[-1]))

    return BackgroundField(
        kx0=kx0, ky0=ky0, wavelength=wvl,
        eps=tuple(eps_l), lam=tuple(lam_l), V=tuple(V_l),
        c_plus=tuple(c_plus), c_minus=tuple(c_minus), thickness=tuple(thick_l),
    )
