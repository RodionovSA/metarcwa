# metarcwa/solver/scattered.py
"""
scattered — source vectors of the PML scattered-field formulation
===================================================================

With ``E = E_inc + E_s`` (``docs/PML.md`` "Illumination: scattered-field
formulation") the scattered field obeys the RCWA + PML equations with a
source localized in the structure::

    d psi / d z~ = A psi + f(z),     z~ = k0 z,   A = [[0, P], [Q, 0]],

where ``f`` comes from ``Delta eps E_inc``, ``Delta eps = eps - eps_void``.
Writing ``F_i = [[Delta eps E_inc,i]]`` (a vector over harmonics; Laurent is
exact because ``E_inc`` is continuous), the ``D``-equations gain ``F_x``,
``F_y`` directly and the ``s``-equations gain ``F_z`` through the ``[S_z]``
elimination (``docs/rcwa_core.md`` Eqs. (16)-(21), (33)), with the stretched
``K̃`` and the same ``[[eps]]`` as :func:`compute_P`::

    f = ( -j K̃x [[eps]]^-1 F_z,  -j K̃y [[eps]]^-1 F_z,  -F_y,  F_x ).

Inside a layer ``E_inc`` is one forward and one backward plane wave
(:mod:`metarcwa.solver.background`), so ``F_i(z) = E_i^± g exp(±lam_bg ...)``
with ``g`` the Fourier coefficients of ``Delta eps exp(j k_par . r)`` (the
phase only along PML axes; along a periodic axis it is the Bloch phase
already carried by ``K``). ``f`` therefore has four columns: forward-s,
forward-p, backward-s, backward-p. Forward columns are referenced to the
layer's left face, backward ones to its right face, so every exponential
stays bounded.

This module turns those terms into the layer's affine S-matrix
(:class:`~metarcwa.solver.smatrix.AffineS`):

- :func:`modal_affine` — eigenmode operators (``ModalOperator``): closed-form
  integrals in the layer eigenbasis, evaluated with a stable ``phi1``.
- :func:`transfer_affine` — matrix-exponential operators
  (``TransferOperator``): the exact particular solution of each slice from
  the exponential of the augmented system ``[[A, f], [0, Gamma]]``, then the
  slices are recombined with phase-shifted sources.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import torch

from metarcwa.model.pml import PMLSpec
from metarcwa.solver.background import BackgroundField
from metarcwa.solver.blockmatrix import Block, Block2x2, apply_vec, solve_vec
from metarcwa.solver.layersolver.matexpsolver import (
    TransferOperator, slice_count, star_power, system_matrix,
)
from metarcwa.solver.layersolver.operator import Background, ModalOperator, SourceData
from metarcwa.solver.smatrix import AffineS, S_boundary, S_prop


@dataclass(frozen=True)
class SourceTerms:
    """Source of one patterned layer, as four exponential terms.

    Attributes
    ----------
    f : torch.Tensor
        Source vectors, shape ``[*B, 4Nh, 4]``; columns forward-s,
        forward-p, backward-s, backward-p. The full source is
        ``f[..., :2] exp(lam_bg z~) + f[..., 2:] exp(-lam_bg (z~ - d~))``
        with ``z~ = k0 z`` measured from the layer's left face.
    lam_bg : torch.Tensor
        Forward modal exponent of the background medium of the layer,
        shape ``B``.
    """
    f: torch.Tensor
    lam_bg: torch.Tensor


# -- stable exponential integrals ------------------------------------------

def phi1(x: torch.Tensor) -> torch.Tensor:
    """``(exp(x) - 1) / x``, accurate near ``x = 0`` (removable singularity)."""
    # Taylor series (to x^7) below |x| = 0.1: truncation ~ |x|^8/9! < 3e-14,
    # and the direct form above it loses at most ~eps/0.1 to cancellation.
    small = x.abs() < 0.1
    xs = torch.where(small, torch.ones_like(x), x)
    series = 1 + x / 2 * (1 + x / 3 * (1 + x / 4 * (1 + x / 5 * (1 + x / 6 * (1 + x / 7 * (1 + x / 8))))))
    return torch.where(small, series, (torch.exp(xs) - 1) / xs)


def exp_integral(a: torch.Tensor, b: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
    """``∫_0^d exp(a t) exp(b (d - t)) dt = (e^{ad} - e^{bd}) / (a - b)``.

    Evaluated as ``d e^{h d} phi1((l - h) d)``, ``h`` the argument with the
    larger real part, so it stays bounded whenever ``Re a, Re b <= 0`` and
    is exact at ``a = b`` (a removable singularity, not a resonance).
    """
    a, b = torch.broadcast_tensors(a, b)
    swap = a.real < b.real
    h = torch.where(swap, b, a)
    l = torch.where(swap, a, b)
    return d * torch.exp(h * d) * phi1((l - h) * d)


# -- source vector of a patterned layer ------------------------------------

def _grid_phase(pml: PMLSpec, kx0, ky0, k0, a1, a2, ny: int, nx: int) -> torch.Tensor:
    """``exp(j k0 (kx0 x + ky0 y))`` on the pattern grid, PML axes only.

    Pixel ``i`` sits at ``x = i px / nx``, the position implied by the FFT
    convention of :func:`convolution_matrix`. Shape ``[*B, ny, nx]``.
    """
    dt = a1.real.dtype if a1.is_complex() else a1.dtype
    phase = torch.zeros(*kx0.shape, 1, 1, dtype=dt, device=kx0.device)
    if pml.sx is not None:
        x = torch.arange(nx, dtype=dt, device=kx0.device) * (a1[0].abs() / nx)
        phase = phase + (k0 * kx0)[..., None, None] * x
    if pml.sy is not None:
        y = torch.arange(ny, dtype=dt, device=kx0.device) * (a2[1].abs() / ny)
        phase = phase + (k0 * ky0)[..., None, None] * y[:, None]
    return torch.exp(1j * phase)


def check_structure_in_physical_region(d_eps: torch.Tensor, pml: PMLSpec) -> None:
    """Warn if ``Delta eps`` (``SourceData.d_eps``) reaches the PML: the
    scattered-field source must lie in the physical region (``docs/PML.md``)."""
    d_eps = d_eps.detach().abs()
    ny, nx = d_eps.shape[-2:]
    outside = ~pml.physical_mask(ny, nx, d_eps.device)
    # Relative tolerance: a soft (e.g. sigmoid) pattern is never exactly zero.
    if outside.any() and (d_eps[..., outside] > _DEPS_TOL * d_eps.amax()).any():
        warnings.warn(
            "A patterned layer differs from its medium_void inside the PML. "
            "The scattered-field formulation assumes the structure "
            "(eps - eps_void) lies in the physical region; the result is "
            "not reliable. Move the structure away from the PML or make the "
            "pattern's void medium the material that crosses the PML.",
            RuntimeWarning, stacklevel=4,
        )


#: |Delta eps| in the PML above this fraction of its maximum counts as structure.
_DEPS_TOL = 1e-4


def _bcast(*ts: torch.Tensor) -> list[torch.Tensor]:
    """Broadcast the batch dims (all but the last two) of ``ts`` to a common shape."""
    batch = torch.broadcast_shapes(*(t.shape[:-2] for t in ts))
    return [t.expand(*batch, *t.shape[-2:]) for t in ts]


def layer_source(source: SourceData, i: int, bg: BackgroundField,
                 Kx: Block, Ky: Block, pml: PMLSpec, a1, a2,
                 m_flat, n_flat) -> SourceTerms:
    """Build the :class:`SourceTerms` of patterned layer ``i`` (``ops`` index).

    Parameters
    ----------
    source : SourceData
        The layer's prepared source data (``op.source``): ``eps - eps_void``
        on the grid, ``[[eps]]`` and the TVF correction ``Qfact``. The TVF
        rule describes the ``D``-``E`` relation of the *total* field, so the
        source carries it for ``E_inc`` too:
        ``f_u += Qfact (phi E_x, phi E_y)`` with ``phi`` the Fourier
        coefficients of ``exp(j k_par . r)``. With it the scattered
        formulation reproduces the total-field one exactly at normal
        incidence.
    i : int
        Element index of the layer in ``PreparedStack.ops`` / ``bg``.
    bg : BackgroundField
        Background solution (:func:`~metarcwa.solver.background.solve_background`).
    Kx, Ky : Block
        Stretched wavevector operators of the solver.
    pml : PMLSpec
        Lateral PML of the model.
    a1, a2 : torch.Tensor
        Lattice vectors (rectangular).
    m_flat, n_flat : torch.Tensor
        Harmonic index map.
    """
    qfact = source.qfact
    ny, nx = source.d_eps.shape[-2:]
    d_eps = source.d_eps.reshape(-1, 1, 1, ny, nx)                # [N_wl,1,1,Ny,Nx]
    k0 = 2 * torch.pi / bg.wavelength                             # [N_wl,1,1]
    phase = _grid_phase(pml, bg.kx0, bg.ky0, k0, a1, a2, ny, nx)   # [*B,Ny,Nx]
    g_hat = torch.fft.fft2(d_eps * phase, norm="forward")
    g = g_hat[..., n_flat % ny, m_flat % nx][..., None]          # [*B,Nh,1]

    h = solve_vec(source.eps_conv, g)                             # [[eps]]^-1 g
    kxh, kyh = apply_vec(Kx, h), apply_vec(Ky, h)

    if qfact is not None:
        phi = torch.fft.fft2(phase, norm="forward")[..., n_flat % ny, m_flat % nx][..., None]

    plus, minus = bg.waves(i)                                     # [*B,6,2]
    cols = []
    for w in (plus, minus):
        Ex, Ey, Ez = w[..., 0:1, :], w[..., 1:2, :], w[..., 2:3, :]   # [*B,1,2]
        fs = torch.cat(_bcast(-1j * kxh * Ez, -1j * kyh * Ez), dim=-2)
        fu = torch.cat(_bcast(-g * Ey, g * Ex), dim=-2)
        if qfact is not None:
            fu = fu + apply_vec(qfact, torch.cat(_bcast(phi * Ex, phi * Ey), dim=-2))
        cols.append(torch.cat(_bcast(fs, fu), dim=-2))
    f = torch.cat(_bcast(*cols), dim=-1)                          # [*B,4Nh,4]
    return SourceTerms(f=f, lam_bg=bg.lam[i])


# -- affine S-matrix of a layer --------------------------------------------

def _sandwich(core: AffineS, background: Background, W, V) -> AffineS:
    """``S_boundary(bg, layer) ⋆ core ⋆ S_boundary(layer, bg)`` (mirror trick)."""
    S_in = S_boundary(background.W0, background.V0, W, V)
    S_out = Block2x2(S_in.d, S_in.c, S_in.b, S_in.a)
    return AffineS(S_in).star(core).star(AffineS(S_out))


def modal_affine(op: ModalOperator, background: Background, src: SourceTerms) -> AffineS:
    """Affine S-matrix of an eigenmode layer with source ``src``.

    In the eigenbasis ``Phi = [[W, W], [V, -V]]``, ``Phi^-1 f = (g+, g-)``
    and the modes decouple: ``c+' = lam c+ + g+``, ``c-' = -lam c- + g-``.
    With nothing incoming (``c+(0) = 0``, ``c-(d) = 0``) each term
    integrates in closed form (:func:`exp_integral`)::

        forward  (f e^{mu z}):        e_R = g+ I(mu, lam),   e_L = -g- I(lam + mu, 0)
        backward (f e^{mu (d - z)}):  e_R = g+ I(0, lam + mu), e_L = -g- I(lam, mu)

    with ``mu = lam_bg`` and ``I(a, b) = ∫_0^d e^{at} e^{b(d-t)} dt``. These
    are carried through the two boundary factors by the extended star.
    """
    Nh2 = op.lam.shape[-1]
    k0 = 2 * torch.pi / background.wvl.reshape(-1, *([1] * (op.lam.ndim - 1)))
    d = k0 * op.thickness                                         # [N_wl,1,..,1]
    if op.source is not None and op.source.phi_lu is not None:
        LU, piv = op.source.phi_lu                                # factorized in prepare()
        batch = torch.broadcast_shapes(LU.shape[:-2], src.f.shape[:-2])
        q = torch.linalg.lu_solve(LU.expand(*batch, *LU.shape[-2:]),
                                  piv.expand(*batch, piv.shape[-1]),
                                  src.f.expand(*batch, *src.f.shape[-2:]).to(LU.dtype))
    else:
        q = solve_vec(Block2x2(op.W, op.W, op.V, -op.V), src.f)   # [*B,4Nh,4]
    gp, gm = q[..., :Nh2, :], q[..., Nh2:, :]
    lam = op.lam[..., :, None]                                    # [*B,2Nh,1]
    mu = src.lam_bg[..., None, None]                              # [*B,1,1]
    zero = torch.zeros_like(mu)
    eR = torch.cat([gp[..., :2] * exp_integral(mu, lam, d[..., None]),
                    gp[..., 2:] * exp_integral(zero, lam + mu, d[..., None])], dim=-1)
    eL = torch.cat([-gm[..., :2] * exp_integral(lam + mu, zero, d[..., None]),
                    -gm[..., 2:] * exp_integral(lam, mu, d[..., None])], dim=-1)
    core = AffineS(S_prop(op.lam, background.wvl, op.thickness), eL, eR)
    return _sandwich(core, background, op.W, op.V)


def _affine_slice(op: TransferOperator, src: SourceTerms, k0, h) -> AffineS:
    """Affine S-matrix of one slice (thickness ``h``), against the gap medium.

    ``expm([[A, f], [0, Gamma]] k0 h)`` holds ``T = expm(A k0 h)`` top-left
    and the particular solution ``∫ e^{A(h-t)} f e^{Gamma t} dt`` (zero
    initial field) top-right, ``Gamma = diag(mu, mu, -mu, -mu)``. Backward
    columns are re-referenced to the slice's right face (factor
    ``e^{mu k0 h}``, bounded). Field-basis ``psi(h) = T psi(0) + p`` is then
    converted to gap-medium amplitudes exactly as
    :func:`~metarcwa.solver.layersolver.matexpsolver.transfer_to_smatrix`,
    plus ``e = left^-1 p``.
    """
    Nh = op.Nh
    A_blocks = system_matrix(op.P, op.Q)
    A = A_blocks.to_dense(Nh)                                     # [*B',4Nh,4Nh]
    f = src.f
    batch = torch.broadcast_shapes(A.shape[:-2], f.shape[:-2])
    A, f = A.expand(*batch, *A.shape[-2:]), f.expand(*batch, *f.shape[-2:])
    mu = src.lam_bg.expand(batch)
    kh = (k0 * h).reshape(*k0.shape, *([1] * (len(batch) - k0.ndim))).expand(batch)
    gamma = torch.stack([mu, mu, -mu, -mu], dim=-1)               # [*B,4]
    n4 = 4 * Nh
    aug = torch.zeros(*batch, n4 + 4, n4 + 4, dtype=A.dtype, device=A.device)
    aug[..., :n4, :n4] = A
    aug[..., :n4, n4:] = f.to(A.dtype)
    aug[..., n4:, n4:] = torch.diag_embed(gamma.to(A.dtype))
    E = torch.linalg.matrix_exp(aug * kh[..., None, None])
    T = Block2x2.from_dense(E[..., :n4, :n4], A_blocks)
    p = E[..., :n4, n4:]
    back = torch.exp(mu * kh)[..., None, None]
    p = torch.cat([p[..., :2], p[..., 2:] * back], dim=-1)

    W0, V0 = op.gap_background.W0, op.gap_background.V0
    TW0_s, TW0_u = T.a @ W0, T.c @ W0
    TV0_s, TV0_u = T.b @ V0, T.d @ V0
    left = Block2x2(W0, -(TW0_s - TV0_s), V0, -(TW0_u - TV0_u))
    right = Block2x2(TW0_s + TV0_s, -W0, TW0_u + TV0_u, V0)
    # One factorization of `left` for the S-matrix and the source vector.
    X = torch.linalg.solve(left.to_dense(Nh), torch.cat(_bcast(right.to_dense(Nh), p), dim=-1))
    M = Block2x2.from_dense(X[..., :n4], right)                   # [c_R+; c_L-] = M [c_L+; c_R-]
    e = X[..., n4:]
    S = Block2x2(M.c, M.d, M.a, M.b)
    Nh2 = 2 * Nh
    return AffineS(S, e_L=e[..., Nh2:, :], e_R=e[..., :Nh2, :])


def _shift(a: AffineS, phase: torch.Tensor, forward: bool) -> AffineS:
    """Scale the forward (or backward) source columns of ``a`` by ``phase``."""
    def sc(e):
        if forward:
            return torch.cat([e[..., :2] * phase, e[..., 2:]], dim=-1)
        return torch.cat([e[..., :2], e[..., 2:] * phase], dim=-1)
    return AffineS(a.S, sc(a.e_L), sc(a.e_R))


def transfer_affine(op: TransferOperator, background: Background, src: SourceTerms) -> AffineS:
    """Affine S-matrix of a matrix-exponential layer with source ``src``.

    Slices as :meth:`TransferOperator.smatrix`. Slice ``k`` sees the source
    of slice ``0`` times ``e^{mu k0 k h}`` (forward terms, referenced to the
    left face) or ``e^{mu k0 (n-1-k) h}`` (backward terms, referenced to the
    right face) — the same bounded factor ``phi(L) = e^{mu k0 L}`` per
    length ``L`` for both. Combining a left block ``X`` of length ``Lx``
    with a right block ``Y`` of length ``Ly`` therefore scales ``Y``'s
    forward columns by ``phi(Lx)`` and ``X``'s backward columns by
    ``phi(Ly)``. That rule is the ``join`` passed to :func:`star_power`, so
    the repeated squaring and its resonance guard are shared with the
    source-free path.
    """
    if op.thickness is None:
        raise ValueError("TransferOperator requires a finite thickness.")
    k0 = 2 * torch.pi / torch.as_tensor(background.wvl)
    n = slice_count(op.lam_bound, k0, op.thickness, op.config)
    h = op.thickness / n
    base = _affine_slice(op, src, k0, h)
    k0b = k0.reshape(-1, *([1] * (src.lam_bg.ndim - 1)))
    phi = lambda slices: (torch.exp(src.lam_bg * k0b * h * slices))[..., None, None]

    def join(left, right, left_len, right_len):
        return _shift(left, phi(right_len), forward=False).star(
            _shift(right, phi(left_len), forward=True))

    result = star_power(base, n, join=join, matrix=lambda a: a.S)
    gb = op.gap_background
    return _sandwich(result, background, gb.W0, gb.V0)


def layer_affine(op, background: Background, src: SourceTerms) -> AffineS:
    """Affine S-matrix of a patterned layer, with two source columns (s, p).

    Dispatches on the operator family, then adds the forward- and
    backward-wave contributions of each polarization: inside the layer they
    must stay separate (different reference faces), outside they are just
    two parts of the same emitted wave.
    """
    if isinstance(op, ModalOperator):
        a = modal_affine(op, background, src)
    elif isinstance(op, TransferOperator):
        a = transfer_affine(op, background, src)
    else:
        raise TypeError(f"no source-vector support for {type(op)}")
    return AffineS(a.S, a.e_L[..., :2] + a.e_L[..., 2:], a.e_R[..., :2] + a.e_R[..., 2:])
