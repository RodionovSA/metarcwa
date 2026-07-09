# tests/block_structure_bench.py
"""
Empirical benchmark: does the Block/Block2x2 structured-matrix design
(src/metarcwa/solver/blockmatrix.py -- SCALAR/DIAG/DENSE storage, promoted
only when unavoidable) give a real memory/speed win on a full RCWA solve?

Controlled A/B design: every layer in this benchmark has the *same*
physical medium (uniform eps=4.0). The `n_patterned` layers are still built
via Layer(shape_fn=..., medium_void=...) -- the mask is just literally
uniform (all ones), so medium_void never contributes -- which forces them
through the PatternedLayer / compute_isotropic / eigsolver code path (DENSE
mode matrices, real eigendecomposition even though Q/P happen to be
diagonal for a uniform grid). The remaining layers are real
HomogeneousLayer specs, going through the closed-form homogeneous_modes
path (DIAG/SCALAR, no eigendecomposition at all). So sweeping n_patterned
compares two code paths solving the *identical* physics problem -- any
speed/memory difference is a pure structural-representation effect, not a
confound from different materials or genuine spatial patterning. TVF is
off by default: for a uniform mask the Li correction Delta = [[eps]] -
[[1/eps]]^-1 is exactly zero (see test_uniform_pattern_equals_homogeneous
in tests/solver/test_layersolver.py), so it cannot change the result here,
only add unrelated cost.

Since both code paths solve the same physics, their S-matrices must agree
-- this is checked explicitly at the end (not just assumed): every
n_patterned configuration's S-matrix is compared against the n_patterned=0
(fully homogeneous) reference.

Every DIAG/SCALAR-mode boundary (i.e. every boundary in the n_patterned=0
row) hits S_boundary's O(Nh) per-harmonic fast path (ANALYSIS.md A4/C4);
any boundary touching a patterned-path layer falls to the O(Nh^3) dense
solve, and that layer's own prepare() pays for a real eigendecomposition
instead of a closed-form solve. Both of these are the same underlying
"exploit the Block structure where it exists" idea, just applied at two
different pipeline stages (mode-solving vs. boundary-matching) -- this
benchmark's construct/solve split reports them separately.

Not in scope here: gradient/backward cost (see the C3 gradient-checkpointing
benchmark in ANALYSIS.md for that) -- this measures pure forward S-matrix
computation, per the request that motivated this script.

Usage:
    python tests/block_structure_bench.py
    python tests/block_structure_bench.py --nh-half 8 --repeats 3
    python tests/block_structure_bench.py --device cpu --n-wvl 5
"""

import argparse
import resource
import time

import torch

from metarcwa.model.base import Model
from metarcwa.model.stack import Stack
from metarcwa.model.layer import Layer
from metarcwa.model.medium import IsotropicMedium
from metarcwa.model.lattice import Lattice
from metarcwa.model.source import Source
from metarcwa.model.nn_helpers import CallableModule
from metarcwa.solver.base import Solver
from metarcwa.solver.config import Config, Factorization


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--nh-half", type=int, default=6,
                   help="Harmonic half-order per axis; Nh = (2*nh_half+1)^2.")
    p.add_argument("--n-wvl", type=int, default=1,
                   help="Number of wavelengths in the batch.")
    p.add_argument("--repeats", type=int, default=5, help="Timed repeats per configuration.")
    p.add_argument("--warmup", type=int, default=2, help="Untimed warmup repeats.")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--tvf", action="store_true",
                   help="Enable TVF factorization (off by default -- isolates the "
                        "boundary/eig cost this benchmark targets from TVF's own cost).")
    return p.parse_args()


def _const_eps(val: complex):
    """CallableModule returning a constant complex eps matching wvl's shape
    (mirrors tests/solver/test_solver.py::_const_eps)."""
    return CallableModule(lambda wvl: torch.full_like(wvl, val, dtype=torch.complex128))


def _uniform_map(lattice, nx: int, ny: int) -> torch.Tensor:
    """shape_fn(lattice, nx, ny) -> mask, all-ones -- i.e. no real spatial
    pattern at all. This deliberately forces the "patterned" layers through
    the PatternedLayer/eigsolver code path while keeping the physical
    medium identical to the homogeneous layers (see module docstring): a
    controlled A/B test of the two code paths on the same physics, not a
    comparison confounded by different materials or genuine patterning.
    Built on `lattice`'s device/dtype since shape_fn receives no explicit
    device argument (Layer.spec() does not move the returned mask itself --
    CallableModule only moves registered buffers/parameters, not ad hoc
    tensors built inside the wrapped call)."""
    return torch.ones(ny, nx, dtype=lattice.dtype, device=lattice.device)


def _make_model(n_patterned: int, n_wvl: int, device: str) -> Model:
    """3-layer stack, every layer physically eps=4.0 (see module docstring
    for why): the first `n_patterned` layers go through the patterned code
    path (shape_fn=_uniform_map, medium_solid=eps4 -- medium_void=eps1 is
    passed only because Layer requires it when shape_fn is given, but it
    never contributes since the mask is all-ones), the rest are real
    HomogeneousLayer specs (medium_solid=eps4, no shape_fn). Vacuum
    incidence/transmission. Oblique incidence (theta, phi != 0) -- matching
    tests/solver/test_layersolver.py::_make_solver's convention -- so no
    configuration accidentally hits the degenerate kx*ky=0 grazing case;
    that isn't what this benchmark is about.
    """
    incidence    = IsotropicMedium(_const_eps(1.0 + 0j))
    transmission = IsotropicMedium(_const_eps(1.0 + 0j))

    layers = []
    for i in range(3):
        if i < n_patterned:
            layer = Layer(
                medium_solid=IsotropicMedium(_const_eps(4.0 + 0j)),
                medium_void=IsotropicMedium(_const_eps(1.0 + 0j)),
                shape_fn=CallableModule(_uniform_map),
                thickness=0.3,
            )
        else:
            layer = Layer(
                medium_solid=IsotropicMedium(_const_eps(4.0 + 0j)),
                thickness=0.3,
            )
        layers.append(layer)

    lattice = Lattice.rectangular(1.0, 1.0)
    stack   = Stack(incidence, layers, transmission, lattice)
    source  = Source(
        wavelength=torch.linspace(1.0, 1.2, n_wvl),
        theta=0.2, phi=0.1,     # oblique: avoids kx*ky=0 grazing harmonics
    )
    return Model(stack, source).to(dtype=torch.float64, device=device)


def _sync(device: str):
    if device == "cuda":
        torch.cuda.synchronize()


def _peak_memory_mb(device: str, before_reset: bool = False) -> tuple[float | None, bool]:
    """Returns (peak_mb, is_approximate).

    CUDA: exact -- torch.cuda.max_memory_allocated() after a
    reset_peak_memory_stats()/empty_cache() taken right before the timed
    block, so it isolates this configuration's peak tensor allocation.

    CPU: approximate -- resource.ru_maxrss is a whole-process high-water
    mark that can't be reset mid-process, so it's only a reasonable
    per-configuration reading because this script always sweeps
    n_patterned in increasing order (increasing expected memory); it is
    NOT a true isolated peak like the CUDA reading.
    """
    if device == "cuda":
        if before_reset:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            return None, False
        return torch.cuda.max_memory_allocated() / (1024 ** 2), False
    if before_reset:
        return None, True
    # ru_maxrss is KB on Linux, bytes on macOS -- Linux is this project's target.
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, True


def run_one(n_patterned: int, config: Config, args, Nh: int) -> dict:
    device = args.device

    def _one_pass():
        model = _make_model(n_patterned, args.n_wvl, device)
        _sync(device)
        t0 = time.perf_counter()
        solver = Solver(model, config)
        _sync(device)
        t1 = time.perf_counter()
        S = solver.solve()
        _sync(device)
        t2 = time.perf_counter()
        return (t1 - t0) * 1e3, (t2 - t1) * 1e3, S

    for _ in range(args.warmup):
        _one_pass()

    _peak_memory_mb(device, before_reset=True)

    construct_ms, solve_ms = [], []
    S_last = None
    for _ in range(args.repeats):
        c_ms, s_ms, S_last = _one_pass()
        construct_ms.append(c_ms)
        solve_ms.append(s_ms)

    peak_mb, peak_approx = _peak_memory_mb(device)

    return {
        "n_patterned": n_patterned,
        "construct_mean": sum(construct_ms) / len(construct_ms),
        "construct_min": min(construct_ms),
        "solve_mean": sum(solve_ms) / len(solve_ms),
        "solve_min": min(solve_ms),
        "total_mean": sum(construct_ms) / len(construct_ms) + sum(solve_ms) / len(solve_ms),
        "peak_mb": peak_mb,
        "peak_approx": peak_approx,
        # detached/CPU snapshot for the end-of-run correctness check (S_last
        # is whichever repeat happened to run last; the model/config are
        # identical across repeats so any repeat's S is equally valid).
        "S_dense": S_last.to_dense(Nh).detach().to("cpu"),
    }


def main():
    args = parse_args()
    Nh = (2 * args.nh_half + 1) ** 2

    config = Config(
        dtype=torch.float64,
        device=args.device,
        m=args.nh_half, n=args.nh_half,
        factorization=Factorization() if args.tvf else None,
    )

    print("=" * 92)
    print(f"torch        : {torch.__version__}")
    print(f"device       : {args.device}"
          + (f" ({torch.cuda.get_device_name(0)})" if args.device == "cuda" else ""))
    print(f"Nh           : {Nh}  (nh_half={args.nh_half})")
    print(f"N_wvl        : {args.n_wvl}")
    print(f"TVF          : {'on' if args.tvf else 'off'}")
    print(f"repeats/warmup: {args.repeats}/{args.warmup}")
    print("=" * 92)

    peak_label = "peak MB*" if args.device != "cuda" else "peak MB"
    header = (f"{'n_patterned':>11} | {'construct ms':>12} (min) | {'solve ms':>10} (min) | "
              f"{'total ms':>10} | {peak_label:>10}")
    print(header)
    print("-" * len(header))

    rows = []
    for n_patterned in (0, 1, 2, 3):
        r = run_one(n_patterned, config, args, Nh)
        rows.append(r)
        peak_str = f"{r['peak_mb']:10.1f}" if r["peak_mb"] is not None else "n/a"
        print(f"{r['n_patterned']:>11} | "
              f"{r['construct_mean']:9.3f} ({r['construct_min']:7.3f}) | "
              f"{r['solve_mean']:7.3f} ({r['solve_min']:6.3f}) | "
              f"{r['total_mean']:10.3f} | {peak_str}")

    print("-" * len(header))
    if rows and rows[0]["peak_approx"]:
        print("* CPU peak memory is an approximate whole-process high-water mark "
              "(resource.ru_maxrss, cannot be reset mid-process) -- not a true "
              "per-configuration isolate like the CUDA reading.")
    first, last = rows[0], rows[-1]
    total_ratio = last["total_mean"] / first["total_mean"] if first["total_mean"] > 0 else float("nan")
    solve_ratio = last["solve_mean"] / first["solve_mean"] if first["solve_mean"] > 0 else float("nan")
    print(f"\nn_patterned=0 -> 3: total time {total_ratio:.1f}x, solve()-only time {solve_ratio:.1f}x")
    if first["peak_mb"] is not None and last["peak_mb"] is not None and first["peak_mb"] > 0:
        mem_ratio = last["peak_mb"] / first["peak_mb"]
        print(f"                    peak memory {mem_ratio:.1f}x "
              f"({first['peak_mb']:.1f} MB -> {last['peak_mb']:.1f} MB)")
    print("\n(Every layer is physically eps=4.0 uniform (see module docstring) -- the\n"
          " n_patterned=0 row solves it via closed-form homogeneous_modes (DIAG/SCALAR,\n"
          " no eigendecomposition) with every S_boundary call hitting the O(Nh) DIAG fast\n"
          " path (A4/C4); the n_patterned=3 row solves the *same physics* via eigsolver\n"
          " (DENSE) with every S_boundary call falling to the O(Nh^3) dense solve. The gap\n"
          " between the two rows is therefore a pure structural-representation effect --\n"
          " confirmed below by comparing their S-matrices, not assumed.)")

    # --- Correctness check: same physics -> same S-matrix ---------------------
    print("\n" + "=" * 92)
    print("Correctness check: S-matrix agreement vs. n_patterned=0 (all rows solve the")
    print("same eps=4.0 uniform 3-layer stack; differences should be at most numerical)")
    print("-" * 92)
    tol_atol, tol_rtol = 1e-6, 1e-6
    ref = rows[0]["S_dense"]
    all_ok = True
    for r in rows:
        diff = (r["S_dense"] - ref)
        max_abs = diff.abs().max().item()
        denom = ref.abs()
        max_rel = (diff.abs() / denom.clamp_min(1e-300)).max().item()
        ok = torch.allclose(r["S_dense"], ref, atol=tol_atol, rtol=tol_rtol)
        all_ok &= ok
        print(f"  n_patterned={r['n_patterned']}: max|diff|={max_abs:.3e}  "
              f"max relative={max_rel:.3e}  -> {'PASS' if ok else 'FAIL'}")
    print("-" * 92)
    print(f"Overall: {'ALL S-matrices agree' if all_ok else 'MISMATCH DETECTED'} "
          f"(atol={tol_atol:g}, rtol={tol_rtol:g})")


if __name__ == "__main__":
    main()
