# tests/solver/test_matexpsolver.py
# Tests for the matrix-exponential patterned-layer solver
# (Config.modesolver == "matexp"): the T-matrix/S-matrix machinery in
# matexpsolver.py, its slicing-for-stability contract, and the
# ModalOperator/TransferOperator field-parity guarantee.

import warnings

import pytest
import torch
from torch.testing import assert_close

from metarcwa._dtypes import to_complex
from metarcwa.solver.config import Config
from metarcwa.solver.blockmatrix import Block2x2
from metarcwa.solver.harmonics import harmonic_index_map, compute_kxy
from metarcwa.solver.smatrix import S_prop
from metarcwa.solver.layersolver.homogeneous import homogeneous_modes
from metarcwa.solver.layersolver.isotropic import compute_isotropic
from metarcwa.solver.layersolver.eigsolver import eigsolver
from metarcwa.solver.layersolver.operator import Background, ModalOperator
from metarcwa.solver.layersolver.matexpsolver import (
    TransferOperator,
    system_matrix,
    transfer_matrix,
    transfer_to_smatrix,
    star_power,
    slice_count,
)


Nh_half = 1   # (2*1+1)**2 = 9 harmonics -- small for speed


# ---------------------------------------------------------------------------
# Fixtures & helpers
# ---------------------------------------------------------------------------

@pytest.fixture(
    params=[
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="CUDA not available"
            ),
        ),
    ]
)
def device(request):
    return request.param


def _harmonic_context(device: str):
    """Oblique-incidence harmonic context on a square unit cell (kx*ky != 0
    for every harmonic, matching test_layersolver.py's convention, so V.d
    stays non-singular)."""
    a1 = torch.tensor([1.0, 0.0], dtype=torch.float64, device=device)
    a2 = torch.tensor([0.0, 1.0], dtype=torch.float64, device=device)
    kx0 = torch.tensor([0.2], dtype=torch.float64, device=device)
    ky0 = torch.tensor([0.1], dtype=torch.float64, device=device)
    m_flat, n_flat = harmonic_index_map(Nh_half, Nh_half, device=device)
    wvl = torch.tensor([0.3], dtype=torch.float64, device=device)
    kx, ky = compute_kxy(kx0, ky0, a1, a2, m_flat, n_flat, k0=2 * torch.pi / wvl)
    Nh = m_flat.shape[0]
    return kx, ky, m_flat, n_flat, wvl, Nh


def _checkerboard(device: str) -> torch.Tensor:
    pat = torch.zeros(8, 8, dtype=torch.float64)
    pat[::2, ::2] = 1.0
    pat[1::2, 1::2] = 1.0
    return pat.to(device)


def _eps_grid(eps_val: float, pattern: torch.Tensor, device: str) -> torch.Tensor:
    """Permittivity grid that is uniform in *value* (eps_solid == eps_void
    == eps_val) but shaped like `pattern` -- same construction
    LayerSolver._patterned uses, kept local here so this file doesn't reach
    into solver internals."""
    eps = torch.full((1,), eps_val, dtype=torch.complex128, device=device)
    return eps[..., None, None] * pattern[None, ...] + (1 - pattern[None, ...]) * eps


def _vacuum_background(kx, ky, wvl, device):
    eps_vac = torch.ones(1, dtype=torch.complex128, device=device)
    _, V0 = homogeneous_modes(eps_vac, kx, ky)
    return Background(V0.eye_like(), V0, wvl)


def _gap_background(eps_grid, kx, ky, wvl, device):
    """Mean-permittivity gap-medium Background, mirroring the "mean" branch
    of LayerSolver._patterned's matexp gap construction (kept local here so
    this file doesn't reach into solver internals -- see _eps_grid)."""
    eps_gap = eps_grid.mean(dim=(-2, -1))
    _, Vg = homogeneous_modes(eps_gap, kx, ky)
    return Background(Vg.eye_like(), Vg, wvl)


def _eps_grid_patterned(eps_solid: complex, eps_void: complex,
                        pattern: torch.Tensor, device: str) -> torch.Tensor:
    """Permittivity grid genuinely patterned in *value* (unlike _eps_grid,
    which is uniform-in-value but pattern-shaped) -- for tests that need
    real spatial contrast, e.g. exercising the fictitious-sub-slab
    resonance mechanism gap embedding is meant to shrink."""
    solid = torch.full((1,), eps_solid, dtype=torch.complex128, device=device)
    void = torch.full((1,), eps_void, dtype=torch.complex128, device=device)
    return solid[..., None, None] * pattern[None, ...] + void[..., None, None] * (1 - pattern[None, ...])


def _lam_bound(kx, ky, eps_grid):
    return torch.sqrt((kx.abs() ** 2 + ky.abs() ** 2).amax() + eps_grid.abs().amax())


# ---------------------------------------------------------------------------
# system_matrix / transfer_matrix / transfer_to_smatrix
# ---------------------------------------------------------------------------

class TestTransferToSmatrix:

    def test_vacuum_slab_matches_s_prop(self, device):
        """Highest-risk correctness check: pins the Phi0 basis-change and
        exponential sign convention. For a vacuum layer, Phi0 diagonalizes A
        exactly, so transfer_to_smatrix must reduce to S_prop."""
        kx, ky, m_flat, n_flat, wvl, Nh = _harmonic_context(device)
        pattern = _checkerboard(device)
        eps_grid = _eps_grid(1.0, pattern, device)   # vacuum

        P, Q = compute_isotropic(eps_grid, m_flat, n_flat, kx, ky, tvf_fields=None)
        eps_vac = torch.ones(1, dtype=torch.complex128, device=device)
        lam_vac, _ = homogeneous_modes(eps_vac, kx, ky)
        background = _vacuum_background(kx, ky, wvl, device)

        d = torch.tensor([0.4], dtype=torch.float64, device=device)
        k0 = 2 * torch.pi / wvl
        A = system_matrix(P, Q)
        T = transfer_matrix(A, Nh, k0, d)
        S = transfer_to_smatrix(T, background)

        S_ref = S_prop(lam_vac, wvl, d)

        assert_close(S.to_dense(Nh), S_ref.to_dense(Nh), atol=1e-8, rtol=1e-6)

    def test_system_matrix_block_placement(self, device):
        """A = [[0, P], [Q, 0]] -- verify against a hand-built dense
        reference, not just via downstream behaviour."""
        kx, ky, m_flat, n_flat, wvl, Nh = _harmonic_context(device)
        pattern = _checkerboard(device)
        eps_grid = _eps_grid(2.5, pattern, device)
        P, Q = compute_isotropic(eps_grid, m_flat, n_flat, kx, ky, tvf_fields=None)

        A_dense = system_matrix(P, Q).to_dense(Nh)
        P_dense = P.to_dense(Nh)
        Q_dense = Q.to_dense(Nh)
        N2 = P_dense.shape[-1]

        assert_close(A_dense[..., :N2, :N2], torch.zeros_like(P_dense))
        assert_close(A_dense[..., :N2, N2:], P_dense)
        assert_close(A_dense[..., N2:, :N2], Q_dense)
        assert_close(A_dense[..., N2:, N2:], torch.zeros_like(Q_dense))


# ---------------------------------------------------------------------------
# Slicing: invariance, star_power, slice_count
# ---------------------------------------------------------------------------

class TestSlicing:

    def test_slicing_invariance_for_thin_layer(self, device):
        """For a thin layer the per-slice exponent is already small, so
        slicing must not change the result beyond numerical error -- n=1 and
        n=8 must agree tightly. Confirms recombination is exact, not
        approximate."""
        kx, ky, m_flat, n_flat, wvl, Nh = _harmonic_context(device)
        pattern = _checkerboard(device)
        eps_grid = _eps_grid(2.5, pattern, device)
        P, Q = compute_isotropic(eps_grid, m_flat, n_flat, kx, ky, tvf_fields=None)
        background = _vacuum_background(kx, ky, wvl, device)

        d = torch.tensor([0.05], dtype=torch.float64, device=device)
        k0 = 2 * torch.pi / wvl
        A = system_matrix(P, Q)

        def s_for_n(n):
            T_slice = transfer_matrix(A, Nh, k0, d / n)
            S_slice = transfer_to_smatrix(T_slice, background)
            return star_power(S_slice, n)

        S1, S8 = s_for_n(1), s_for_n(8)
        assert_close(S1.to_dense(Nh), S8.to_dense(Nh), atol=1e-10, rtol=1e-8)

    def test_star_power_matches_explicit_stars(self, device):
        kx, ky, m_flat, n_flat, wvl, Nh = _harmonic_context(device)
        pattern = _checkerboard(device)
        eps_grid = _eps_grid(2.5, pattern, device)
        P, Q = compute_isotropic(eps_grid, m_flat, n_flat, kx, ky, tvf_fields=None)
        background = _vacuum_background(kx, ky, wvl, device)

        d = torch.tensor([0.03], dtype=torch.float64, device=device)
        k0 = 2 * torch.pi / wvl
        T_slice = transfer_matrix(system_matrix(P, Q), Nh, k0, d)
        S_slice = transfer_to_smatrix(T_slice, background)

        n = 5
        explicit = S_slice
        for _ in range(n - 1):
            explicit = explicit.star(S_slice)

        result = star_power(S_slice, n)
        assert_close(result.to_dense(Nh), explicit.to_dense(Nh), atol=1e-10, rtol=1e-8)

    def test_star_power_rejects_zero(self):
        with pytest.raises(ValueError):
            star_power(Block2x2.star_identity(), 0)

    def test_slice_count_monotonic_in_thickness(self):
        cfg = Config(dtype=torch.float64)
        k0 = torch.tensor([2 * torch.pi / 0.3])
        lam_bound = torch.tensor(5.0)
        n_thin  = slice_count(lam_bound, k0, torch.tensor([0.01]), cfg)
        n_thick = slice_count(lam_bound, k0, torch.tensor([10.0]), cfg)
        assert n_thick > n_thin

    def test_slice_count_respects_explicit_override(self):
        cfg = Config(dtype=torch.float64, matexp_slices=7)
        k0 = torch.tensor([2 * torch.pi / 0.3])
        n = slice_count(torch.tensor(100.0), k0, torch.tensor([10.0]), cfg)
        assert n == 7

    def test_slice_count_disabled_returns_one(self):
        cfg = Config(dtype=torch.float64, matexp_slicing=False)
        k0 = torch.tensor([2 * torch.pi / 0.3])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            n = slice_count(torch.tensor(100.0), k0, torch.tensor([10.0]), cfg)
        assert n == 1

    def test_slice_count_disabled_warns_when_budget_exceeded(self):
        cfg = Config(dtype=torch.float64, matexp_slicing=False)
        k0 = torch.tensor([2 * torch.pi / 0.3])
        with pytest.warns(RuntimeWarning):
            slice_count(torch.tensor(100.0), k0, torch.tensor([10.0]), cfg)

    def test_slice_count_clamped_to_max_slices_warns(self):
        cfg = Config(dtype=torch.float64, matexp_max_slices=4, matexp_max_exponent=0.01)
        k0 = torch.tensor([2 * torch.pi / 0.3])
        with pytest.warns(RuntimeWarning):
            n = slice_count(torch.tensor(100.0), k0, torch.tensor([10.0]), cfg)
        assert n == 4


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------

def test_config_rejects_unknown_modesolver():
    with pytest.raises(ValueError):
        Config(modesolver="bogus")


def test_config_rejects_unknown_matexp_gap():
    with pytest.raises(ValueError):
        Config(matexp_gap="bogus")


# ---------------------------------------------------------------------------
# ModalOperator / TransferOperator field parity
# ---------------------------------------------------------------------------

class TestFieldParity:

    def test_modal_and_transfer_operator_agree_on_interior_field(self, device):
        """The strongest single correctness check, and the field-parity
        guarantee the operator-as-Protocol design exists for: for the same
        patterned layer, ModalOperator.transfer (gap-matrix similarity
        transform) and TransferOperator.transfer (direct expm) must agree at
        an interior depth."""
        kx, ky, m_flat, n_flat, wvl, Nh = _harmonic_context(device)
        pattern = _checkerboard(device)
        eps_grid = _eps_grid(2.5, pattern, device)
        P, Q = compute_isotropic(eps_grid, m_flat, n_flat, kx, ky, tvf_fields=None)
        background = _vacuum_background(kx, ky, wvl, device)

        d = torch.tensor([0.3], dtype=torch.float64, device=device)
        lam, W, V = eigsolver(P, Q, stable_eig_grad=True)
        modal_op = ModalOperator(lam, W, V, thickness=d)

        cfg = Config(dtype=torch.float64)
        gap_background = _gap_background(eps_grid, kx, ky, wvl, device)
        transfer_op = TransferOperator(P, Q, Nh, _lam_bound(kx, ky, eps_grid), cfg,
                                        gap_background=gap_background, thickness=d)

        z = torch.tensor([0.15], dtype=torch.float64, device=device)
        T_modal = modal_op.transfer(background, z)
        T_transfer = transfer_op.transfer(background, z)

        assert_close(T_modal.to_dense(Nh), T_transfer.to_dense(Nh), atol=1e-8, rtol=1e-6)


# ---------------------------------------------------------------------------
# Gap-medium embedding (Config.matexp_gap): the sandwich in
# TransferOperator.smatrix() references each slice to a per-layer gap
# medium instead of plain vacuum, then transitions back to the stack's
# vacuum background via two boundary S-matrices. This is exact for *any*
# gap medium -- only conditioning changes, never the answer.
# ---------------------------------------------------------------------------

def _oblique_harmonic_context(device: str):
    """A more oblique context than _harmonic_context: kx0/ky0 large enough
    that some harmonics have kt^2 = kx^2+ky^2 > 1 -- i.e. evanescent in
    vacuum -- which is the regime the fictitious-sub-slab resonance
    mechanism needs (a harmonic evanescent in the reference medium but
    propagating inside a high-index sub-slab). _harmonic_context's own
    kx0=0.2/ky0=0.1 never reaches kt^2>1 with Nh_half=1, so it can't
    reproduce the mechanism; this fixture is for tests that specifically
    need to."""
    a1 = torch.tensor([1.0, 0.0], dtype=torch.float64, device=device)
    a2 = torch.tensor([0.0, 1.0], dtype=torch.float64, device=device)
    kx0 = torch.tensor([0.8], dtype=torch.float64, device=device)
    ky0 = torch.tensor([0.5], dtype=torch.float64, device=device)
    m_flat, n_flat = harmonic_index_map(Nh_half, Nh_half, device=device)
    wvl = torch.tensor([0.3], dtype=torch.float64, device=device)
    kx, ky = compute_kxy(kx0, ky0, a1, a2, m_flat, n_flat, k0=2 * torch.pi / wvl)
    Nh = m_flat.shape[0]
    return kx, ky, m_flat, n_flat, wvl, Nh


class TestGapMediumInvariance:
    """The gap-medium sandwich is algebraically exact for any choice of
    homogeneous gap medium -- the direct correctness proof underlying
    Config.matexp_gap's whole premise (that the setting only affects
    conditioning, never the answer)."""

    def test_smatrix_independent_of_gap_choice(self, device):
        kx, ky, m_flat, n_flat, wvl, Nh = _harmonic_context(device)
        pattern = _checkerboard(device)
        eps_grid = _eps_grid(2.5, pattern, device)
        P, Q = compute_isotropic(eps_grid, m_flat, n_flat, kx, ky, tvf_fields=None)
        background = _vacuum_background(kx, ky, wvl, device)
        d = torch.tensor([0.05], dtype=torch.float64, device=device)   # thin, non-resonant
        lam_bound = _lam_bound(kx, ky, eps_grid)
        # matexp_slices fixed explicitly: slice_count depends only on
        # (lam_bound, k0, thickness, config), none of which differ between
        # the operators below, so all three already get the same n -- fixing
        # it here just makes that shared-n precondition explicit/robust.
        cfg = Config(dtype=torch.float64, modesolver="matexp", matexp_slices=3)

        eps_arbitrary = torch.full((1,), 9.0, dtype=torch.complex128, device=device)
        _, V_arb = homogeneous_modes(eps_arbitrary, kx, ky)
        gaps = {
            "vacuum": _vacuum_background(kx, ky, wvl, device),
            "mean":   _gap_background(eps_grid, kx, ky, wvl, device),
            "arbitrary (eps=9)": Background(V_arb.eye_like(), V_arb, wvl),
        }

        results = {
            name: TransferOperator(P, Q, Nh, lam_bound, cfg, gap_background=gap,
                                    thickness=d).smatrix(background).to_dense(Nh)
            for name, gap in gaps.items()
        }
        ref_name, ref = next(iter(results.items()))
        for name, S in list(results.items())[1:]:
            assert_close(S, ref, atol=1e-9, rtol=1e-7,
                         msg=f"gap={name!r} disagrees with gap={ref_name!r}")

    def test_transfer_independent_of_gap_choice(self, device):
        """TransferOperator.transfer() is a raw field-basis propagator with
        no gap-medium embedding at all (unlike smatrix()) -- pin that it was
        correctly left alone by this change (see matexpsolver.py module
        docstring / TransferOperator.smatrix docstring)."""
        kx, ky, m_flat, n_flat, wvl, Nh = _harmonic_context(device)
        pattern = _checkerboard(device)
        eps_grid = _eps_grid(2.5, pattern, device)
        P, Q = compute_isotropic(eps_grid, m_flat, n_flat, kx, ky, tvf_fields=None)
        background = _vacuum_background(kx, ky, wvl, device)
        d = torch.tensor([0.3], dtype=torch.float64, device=device)
        lam_bound = _lam_bound(kx, ky, eps_grid)
        cfg = Config(dtype=torch.float64, modesolver="matexp")

        op_vac = TransferOperator(P, Q, Nh, lam_bound, cfg,
                                   gap_background=_vacuum_background(kx, ky, wvl, device),
                                   thickness=d)
        op_mean = TransferOperator(P, Q, Nh, lam_bound, cfg,
                                    gap_background=_gap_background(eps_grid, kx, ky, wvl, device),
                                    thickness=d)

        z = torch.tensor([0.15], dtype=torch.float64, device=device)
        assert_close(op_vac.transfer(background, z).to_dense(Nh),
                     op_mean.transfer(background, z).to_dense(Nh),
                     atol=1e-12, rtol=1e-10)


class TestGapMediumMechanism:
    """Regression for the reason Config.matexp_gap exists: on a high-contrast
    structure with at least one harmonic evanescent in vacuum but
    propagating inside the material (kt^2 > 1, see
    _oblique_harmonic_context), vacuum-embedded sub-slabs develop a sharp
    S-matrix resonance at some thickness; gap-embedded ones should not (or
    much less so), since the gap medium's own kt^2 threshold sits above the
    vacuum one."""

    @pytest.mark.parametrize("gap_mode", ["mean", "max"])
    def test_gap_embedding_suppresses_sub_slab_resonance(self, device, gap_mode):
        kx, ky, m_flat, n_flat, wvl, Nh = _oblique_harmonic_context(device)
        pattern = _checkerboard(device)
        eps_grid = _eps_grid_patterned(12.0 + 0j, 1.0 + 0j, pattern, device)
        P, Q = compute_isotropic(eps_grid, m_flat, n_flat, kx, ky, tvf_fields=None)
        vacuum = _vacuum_background(kx, ky, wvl, device)
        if gap_mode == "mean":
            gap = _gap_background(eps_grid, kx, ky, wvl, device)
        else:   # "max" -- componentwise max, mirrors LayerSolver._patterned
            eps_gap = (eps_grid.real.amax(dim=(-2, -1))
                       + 1j * eps_grid.imag.amax(dim=(-2, -1)))
            _, Vg = homogeneous_modes(eps_gap, kx, ky)
            gap = Background(Vg.eye_like(), Vg, wvl)

        A = system_matrix(P, Q)
        k0 = 2 * torch.pi / wvl
        ts = torch.linspace(0.05, 50.0, 200, dtype=torch.float64, device=device)

        def max_mag(background):
            mags = []
            for t in ts:
                T = transfer_matrix(A, Nh, k0, t.reshape(1))
                S = transfer_to_smatrix(T, background)
                mags.append(float(S.to_dense().abs().max()))
            return max(mags)

        mag_vacuum = max_mag(vacuum)
        mag_gap = max_mag(gap)

        # Vacuum embedding must show the mechanism at all (else the test
        # fixture itself isn't exercising it) ...
        assert mag_vacuum > 10.0, (
            f"expected a vacuum-embedded sub-slab resonance on this "
            f"high-contrast fixture, got max|S|={mag_vacuum:.3g}; fixture "
            "may no longer reach the evanescent-in-vacuum regime"
        )
        # ... and gap embedding (either mode) must suppress it, staying near
        # the physically-bounded O(1) a lossless passive S-matrix should have.
        assert mag_gap < 1.5, (
            f"gap_mode={gap_mode!r} did not suppress the resonance: "
            f"max|S|={mag_gap:.3g}"
        )


# ---------------------------------------------------------------------------
# Thick-layer stability: the whole reason slicing exists
# ---------------------------------------------------------------------------

class TestThickLayerStability:

    def test_auto_slicing_stays_finite_and_matches_eig(self, device):
        """A layer thick enough that an unsliced expm overflows must still
        be handled correctly under default (auto) slicing, and must match
        the eig path.

        eps=0.01 is well below every harmonic's kx^2+ky^2 (~0.02-0.41 here),
        making every mode evanescent (Re(lam) != 0, up to ~0.63) rather than
        propagating -- unlike a propagating mode (|exp(lam k0 d)| == 1
        exactly, no growth regardless of d), only a genuinely evanescent
        mode's exponential blows up for large d, which is what this test
        needs to actually exercise the instability slicing exists for.
        """
        kx, ky, m_flat, n_flat, wvl, Nh = _harmonic_context(device)
        pattern = _checkerboard(device)
        eps_grid = _eps_grid(0.01, pattern, device)
        P, Q = compute_isotropic(eps_grid, m_flat, n_flat, kx, ky, tvf_fields=None)
        background = _vacuum_background(kx, ky, wvl, device)
        d = torch.tensor([200.0], dtype=torch.float64, device=device)   # thick

        lam, W, V = eigsolver(P, Q, stable_eig_grad=True)
        S_eig = ModalOperator(lam, W, V, thickness=d).smatrix(background)

        cfg = Config(dtype=torch.float64)   # matexp_slicing=True by default
        # Real (mean-eps) gap medium here: eps_grid is uniform-in-value (see
        # _eps_grid), so the gap exactly equals the layer -- gap embedding
        # only makes this more accurate, not less; tolerances stay valid.
        gap_background = _gap_background(eps_grid, kx, ky, wvl, device)
        S_matexp = TransferOperator(
            P, Q, Nh, _lam_bound(kx, ky, eps_grid), cfg,
            gap_background=gap_background, thickness=d,
        ).smatrix(background)

        S_matexp_dense = S_matexp.to_dense(Nh)
        assert not torch.isnan(S_matexp_dense).any()
        assert not torch.isinf(S_matexp_dense).any()
        assert_close(S_matexp_dense, S_eig.to_dense(Nh), atol=1e-6, rtol=1e-5)

    def test_unsliced_matexp_degrades_for_thick_layer(self, device):
        """Disabling slicing for the same thick layer must not silently give
        the same accurate answer -- otherwise slicing would be unmotivated.
        Expect NaN/Inf, a large deviation from the eig reference, or (seen on
        some LAPACK/cuSOLVER backends, which raise on an exactly-singular
        matrix rather than returning Inf/NaN) a linear-algebra error -- all
        are valid manifestations of the same overflowed-T instability."""
        kx, ky, m_flat, n_flat, wvl, Nh = _harmonic_context(device)
        pattern = _checkerboard(device)
        eps_grid = _eps_grid(0.01, pattern, device)   # evanescent, see sibling test
        P, Q = compute_isotropic(eps_grid, m_flat, n_flat, kx, ky, tvf_fields=None)
        background = _vacuum_background(kx, ky, wvl, device)
        d = torch.tensor([200.0], dtype=torch.float64, device=device)

        lam, W, V = eigsolver(P, Q, stable_eig_grad=True)
        S_eig = ModalOperator(lam, W, V, thickness=d).smatrix(background).to_dense(Nh)

        cfg = Config(dtype=torch.float64, matexp_slicing=False)
        # Deliberately vacuum here, not a real (mean-eps) gap: eps_grid is
        # uniform-in-value (see _eps_grid), so a real gap would exactly equal
        # the layer and transfer_to_smatrix would degenerate to pure
        # propagation -- stable even unsliced, which would cancel the very
        # reference/layer-mismatch instability this test exists to exercise.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                S_bad = TransferOperator(
                    P, Q, Nh, _lam_bound(kx, ky, eps_grid), cfg,
                    gap_background=background, thickness=d,
                ).smatrix(background).to_dense(Nh)
            except torch._C._LinAlgError:
                return   # a hard singular-matrix error is itself the failure

        degraded = (
            torch.isnan(S_bad).any()
            or torch.isinf(S_bad).any()
            or (S_bad - S_eig).abs().max().item() > 1e-3
        )
        assert degraded


# ---------------------------------------------------------------------------
# Gradients: matexp must not just match eig's values, but produce equally
# usable gradients -- in fact cleaner ones, since torch.linalg.matrix_exp has
# an exact autograd formula and needs none of eigsolver's Eig
# Lorentzian-broadening machinery for near-degenerate eigenvalues.
# ---------------------------------------------------------------------------

class TestGradients:

    def _loss_and_leaves(self, device, modesolver, eps_val, d_val):
        """Build a single patterned layer, run it through `modesolver`, and
        return (loss, eps_param, d) where loss = sum(|S11|^2) and
        eps_param/d are real leaf tensors the S-matrix was built from."""
        kx, ky, m_flat, n_flat, wvl, Nh = _harmonic_context(device)
        pattern = _checkerboard(device)
        eps_param = torch.tensor(eps_val, dtype=torch.float64, device=device,
                                  requires_grad=True)
        d = torch.tensor([d_val], dtype=torch.float64, device=device,
                          requires_grad=True)
        eps_solid = to_complex(eps_param).reshape(1)
        eps_grid = (eps_solid[..., None, None] * pattern[None, ...]
                    + (1 - pattern[None, ...]) * eps_solid[..., None, None])

        P, Q = compute_isotropic(eps_grid, m_flat, n_flat, kx, ky, tvf_fields=None)
        background = _vacuum_background(kx, ky, wvl, device)

        if modesolver == "eig":
            lam, W, V = eigsolver(P, Q, stable_eig_grad=True)
            op = ModalOperator(lam, W, V, thickness=d)
        elif modesolver == "matexp":
            cfg = Config(dtype=torch.float64, modesolver="matexp")
            # Real (differentiable) gap medium: this is the path that must
            # exercise eps_gap's gradient contribution (analytically it
            # cancels, per TestGapMediumInvariance, but the implemented
            # smatrix() still differentiates through it).
            gap_background = _gap_background(eps_grid, kx, ky, wvl, device)
            op = TransferOperator(P, Q, Nh, _lam_bound(kx, ky, eps_grid), cfg,
                                   gap_background=gap_background, thickness=d)
        else:
            raise ValueError(modesolver)

        S_dense = op.smatrix(background).to_dense(Nh)
        N = S_dense.shape[-1] // 2
        S11 = S_dense[..., :N, :N]
        loss = S11.abs().pow(2).sum()
        return loss, eps_param, d

    def test_matexp_gradients_match_eig(self, device):
        loss_e, eps_e, d_e = self._loss_and_leaves(device, "eig", 2.5, 0.3)
        loss_e.backward()
        loss_m, eps_m, d_m = self._loss_and_leaves(device, "matexp", 2.5, 0.3)
        loss_m.backward()

        assert_close(eps_e.grad, eps_m.grad, atol=1e-6, rtol=1e-4)
        assert_close(d_e.grad, d_m.grad, atol=1e-6, rtol=1e-4)

    def test_matexp_gradient_matches_finite_difference(self, device):
        eps_val, d_val, h = 2.5, 0.3, 1e-5

        loss0, eps0, d0 = self._loss_and_leaves(device, "matexp", eps_val, d_val)
        loss0.backward()

        loss_ep, *_ = self._loss_and_leaves(device, "matexp", eps_val + h, d_val)
        loss_em, *_ = self._loss_and_leaves(device, "matexp", eps_val - h, d_val)
        fd_eps = (loss_ep.item() - loss_em.item()) / (2 * h)

        loss_dp, *_ = self._loss_and_leaves(device, "matexp", eps_val, d_val + h)
        loss_dm, *_ = self._loss_and_leaves(device, "matexp", eps_val, d_val - h)
        fd_d = (loss_dp.item() - loss_dm.item()) / (2 * h)

        assert abs(eps0.grad.item() - fd_eps) < 1e-3 * max(1.0, abs(fd_eps))
        assert abs(d0.grad.item() - fd_d) < 1e-3 * max(1.0, abs(fd_d))
