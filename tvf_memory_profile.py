# tvf_memory_profile.py
"""
Compare peak / residual memory of an RCWA solve on a single patterned layer,
with TVF (Li/FFF factorization) enabled vs. disabled.

Background
----------
Exploration of the solver traced the TVF-only memory cost to two distinct
sinks, both absent when ``Config.factorization=None`` (the plain-Laurent
``Q0``-only branch, ``solver/layersolver/isotropic.py:288``):

  1. Transient peak — the Newton TVF optimizer builds a dense
     ``torch.eye(flat)`` basis and ``vmap``s a full-grid ``ifft2`` Hessian-
     vector product over it, scaling as ``(2m+1)(2n+1) x Ny x Nx``, plus a
     dense ``[B, flat, flat]`` Hessian solve.
     (``solver/tvf/optimizers.py:196-217``)
  2. Retained residual — the FFF Q-path allocates ~a dozen extra batched
     ``[..., Nh, Nh]`` dense matrices (``epsilon_inv_conv``, 4 A-blocks, 4
     solves, 8 matmuls) that depend on the differentiable pattern grid, so
     they are kept alive by the autograd graph whenever grad is enabled.
     (``solver/layersolver/isotropic.py:290-295`` + ``compute_Qfact``)

This script runs four variants (TVF on/off x grad on/off), each in its own
subprocess for a clean per-variant peak, and reports baseline / peak /
residual-with-graph / residual-after-cleanup memory for each, so the two
sinks can be told apart:
  - a large delta-peak gap present in BOTH grad modes -> sink #1 (transient).
  - a large residual-with-graph gap present ONLY with grad on -> sink #2
    (retained FFF matrices).

Usage
-----
    uv run python tvf_memory_profile.py
    uv run python tvf_memory_profile.py --device cpu --m 6 --n 6
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import json
import resource
import subprocess
import sys
import time

import torch

PERIOD = 400.0
THICKNESS = 200.0
RECT_SIZE = (200.0, 150.0)
RECT_ANGLE = 20.0
SOFTNESS = 2.0  # boundary smoothing width (grid pixels) used when requires_grad

_DTYPES = {"float32": torch.float32, "float64": torch.float64}


# --------------------------------------------------------------------------
# Model construction (single patterned layer, mirrors examples/convergence_test.ipynb)
# --------------------------------------------------------------------------

def build_model(*, requires_grad: bool, wl_count: int):
    from dispertorch import ConstantEps
    from metashapes.shape import Rectangle

    from metarcwa import IsotropicMedium, Layer, Lattice, Model, Source, Stack
    from metarcwa.model.adapters import from_dispertorch, from_metashapes

    eps_inc = ConstantEps(1.0)
    eps_trans = ConstantEps(1.46 ** 2)
    eps_void = ConstantEps(1.0)
    eps_solid = ConstantEps(2.1 ** 2, eps_im=0.0)

    medium_inc = IsotropicMedium(from_dispertorch(eps_inc))
    medium_trans = IsotropicMedium(from_dispertorch(eps_trans))
    medium_solid = IsotropicMedium(from_dispertorch(eps_solid))
    medium_void = IsotropicMedium(from_dispertorch(eps_void))

    xc = yc = PERIOD / 2
    if requires_grad:
        # A learnable size makes the rasterized pattern (eps_grid) depend on
        # a differentiable parameter, so the FFF Q-path (sink #2) actually
        # retains an autograd graph. Needs a soft (non-hard-step) mask.
        size = torch.nn.Parameter(torch.tensor(RECT_SIZE))
    else:
        size = torch.tensor(RECT_SIZE)
    shape = Rectangle((xc, yc), size, angle=RECT_ANGLE, corner_radius=0.0)
    shape_fn = from_metashapes(shape, soft=requires_grad, softness=SOFTNESS if requires_grad else 0.0)

    lattice = Lattice.rectangular(px=PERIOD, py=PERIOD)
    layer = Layer(medium_solid, THICKNESS, medium_void, shape_fn)
    stack = Stack(medium_inc, [layer], medium_trans, lattice)

    wl = torch.linspace(400.0, 700.0, wl_count)
    theta = torch.tensor([0.0])
    phi = torch.tensor([0.0])
    src = Source(wl, theta, phi)

    return Model(stack, src)


def build_config(*, with_tvf: bool, dtype, device, m, n, nx, ny, truncation,
                  newton_chunk_size):
    from metarcwa import Config, Factorization

    factorization = (
        Factorization(method="Jones", beta=0.05, gamma=0.05,
                      newton_chunk_size=newton_chunk_size)
        if with_tvf else None
    )
    return Config(
        dtype, device,
        m=m, n=n, nx=nx, ny=ny,
        truncation=truncation,
        factorization=factorization,
    )


# --------------------------------------------------------------------------
# Memory measurement
# --------------------------------------------------------------------------

def _cuda_mib(nbytes: int) -> float:
    return nbytes / (1024 ** 2)


def _rss_mib() -> float:
    with open("/proc/self/statm") as f:
        resident_pages = int(f.read().split()[1])
    return resident_pages * resource.getpagesize() / (1024 ** 2)


def _peak_rss_mib() -> float:
    # ru_maxrss is already in KiB on Linux.
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def measure_variant(*, with_tvf: bool, requires_grad: bool, device_str: str,
                     dtype_str: str, m: int, n: int, nx: int, ny: int,
                     truncation: str, wl_count: int, newton_chunk_size) -> dict:
    dtype = _DTYPES[dtype_str]
    device = torch.device(device_str)
    is_cuda = device.type == "cuda"

    model = build_model(requires_grad=requires_grad, wl_count=wl_count)
    config = build_config(with_tvf=with_tvf, dtype=dtype, device=device,
                           m=m, n=n, nx=nx, ny=ny, truncation=truncation,
                           newton_chunk_size=newton_chunk_size)

    from metarcwa import Solver

    grad_ctx = contextlib.nullcontext() if requires_grad else torch.no_grad()

    t0 = time.perf_counter()

    if is_cuda:
        # Warm up the CUDA context/allocator before measuring, so context
        # init doesn't pollute the peak-memory stats we care about.
        torch.zeros(1, device=device)
        torch.cuda.synchronize(device)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        baseline_mib = _cuda_mib(torch.cuda.memory_allocated(device))

        with grad_ctx:
            solver = Solver(model, config)
            solution = solver.run()
        torch.cuda.synchronize(device)

        peak_mib = _cuda_mib(torch.cuda.max_memory_allocated(device))
        residual_with_graph_mib = _cuda_mib(torch.cuda.memory_allocated(device))

        del solver, solution
        gc.collect()
        torch.cuda.empty_cache()
        residual_after_cleanup_mib = _cuda_mib(torch.cuda.memory_allocated(device))
    else:
        baseline_mib = _rss_mib()

        with grad_ctx:
            solver = Solver(model, config)
            solution = solver.run()

        residual_with_graph_mib = _rss_mib()
        peak_mib = _peak_rss_mib()  # whole-process high-water mark (subprocess-isolated)

        del solver, solution
        gc.collect()
        residual_after_cleanup_mib = _rss_mib()

    elapsed_s = time.perf_counter() - t0

    return {
        "with_tvf": with_tvf,
        "requires_grad": requires_grad,
        "device": device_str,
        "baseline_mib": baseline_mib,
        "peak_mib": peak_mib,
        "delta_peak_mib": peak_mib - baseline_mib,
        "residual_with_graph_mib": residual_with_graph_mib,
        "residual_after_cleanup_mib": residual_after_cleanup_mib,
        "elapsed_s": elapsed_s,
    }


# --------------------------------------------------------------------------
# Subprocess isolation (one variant per process -> clean per-variant peak,
# no caching-allocator cross-contamination between variants)
# --------------------------------------------------------------------------

def run_child(args: argparse.Namespace) -> None:
    result = measure_variant(
        with_tvf=bool(args._tvf),
        requires_grad=bool(args._grad),
        device_str=args.device,
        dtype_str=args.dtype,
        m=args.m, n=args.n, nx=args.nx, ny=args.ny,
        truncation=args.truncation,
        wl_count=args.wl_count,
        newton_chunk_size=args.newton_chunk_size,
    )
    # Single JSON line on stdout; everything else (imports, warnings) goes
    # to stderr so the parent can parse stdout cleanly.
    print(json.dumps(result))


def run_parent(args: argparse.Namespace) -> None:
    grad_modes = {"on": [True], "off": [False], "both": [False, True]}[args.grad]
    variants = [(tvf, grad) for tvf in (False, True) for grad in grad_modes]

    results = []
    for with_tvf, requires_grad in variants:
        label = f"tvf={'on' if with_tvf else 'off':3s} grad={'on' if requires_grad else 'off':3s}"
        print(f"[running] {label} ...", file=sys.stderr)
        cmd = [
            sys.executable, __file__,
            "--device", args.device,
            "--dtype", args.dtype,
            "--m", str(args.m), "--n", str(args.n),
            "--nx", str(args.nx), "--ny", str(args.ny),
            "--truncation", args.truncation,
            "--wl-count", str(args.wl_count),
            "--newton-chunk-size",
            "none" if args.newton_chunk_size is None else str(args.newton_chunk_size),
            "--_tvf", "1" if with_tvf else "0",
            "--_grad", "1" if requires_grad else "0",
            "--_child",
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            print(f"[FAILED] {label}\n{proc.stderr}", file=sys.stderr)
            results.append({
                "with_tvf": with_tvf, "requires_grad": requires_grad,
                "device": args.device, "error": proc.stderr.strip()[-2000:],
            })
            continue
        # stdout should be exactly one JSON line; be tolerant of stray output.
        line = ""
        for candidate in proc.stdout.splitlines()[::-1]:
            if candidate.strip().startswith("{"):
                line = candidate
                break
        try:
            results.append(json.loads(line))
        except json.JSONDecodeError:
            print(f"[FAILED] {label}: could not parse child output\n{proc.stdout}\n{proc.stderr}",
                  file=sys.stderr)
            results.append({
                "with_tvf": with_tvf, "requires_grad": requires_grad,
                "device": args.device, "error": "unparsable child output",
            })

    print_report(results, args)


def print_report(results: list[dict], args: argparse.Namespace) -> None:
    print()
    print(f"metarcwa TVF memory profile  "
          f"(device={args.device}, dtype={args.dtype}, m={args.m}, n={args.n}, "
          f"nx={args.nx}, ny={args.ny}, truncation={args.truncation}, "
          f"wl_count={args.wl_count}, newton_chunk_size={args.newton_chunk_size})")
    print("=" * 116)
    header = (f"{'variant':<16}{'baseline':>12}{'peak':>12}{'d_peak':>12}"
              f"{'resid(graph)':>14}{'resid(clean)':>14}{'time(s)':>10}")
    print(header)
    print("-" * 116)

    by_key = {}
    for r in results:
        label = f"tvf={'on' if r['with_tvf'] else 'off'} grad={'on' if r['requires_grad'] else 'off'}"
        by_key[(r["with_tvf"], r["requires_grad"])] = r
        if "error" in r:
            print(f"{label:<16}{'ERROR':>12}  {r['error'].splitlines()[-1] if r['error'] else ''}")
            continue
        print(f"{label:<16}"
              f"{r['baseline_mib']:>10.1f}M"
              f"{r['peak_mib']:>10.1f}M"
              f"{r['delta_peak_mib']:>10.1f}M"
              f"{r['residual_with_graph_mib']:>12.1f}M"
              f"{r['residual_after_cleanup_mib']:>12.1f}M"
              f"{r['elapsed_s']:>10.2f}")
    print("=" * 116)

    def gap(grad: bool, field: str) -> str:
        on = by_key.get((True, grad))
        off = by_key.get((False, grad))
        if not on or not off or "error" in on or "error" in off:
            return "n/a"
        a, b = on[field], off[field]
        ratio = f"{a / b:.2f}x" if b > 1e-9 else "n/a"
        return f"{a - b:+.1f} MiB ({ratio})"

    print("\nTVF-on vs TVF-off gaps (attribution):")
    for grad in (False, True):
        if (True, grad) not in by_key or (False, grad) not in by_key:
            continue
        print(f"  grad={'on ' if grad else 'off'}: "
              f"delta_peak gap = {gap(grad, 'delta_peak_mib')}, "
              f"residual(graph) gap = {gap(grad, 'residual_with_graph_mib')}, "
              f"residual(clean) gap = {gap(grad, 'residual_after_cleanup_mib')}")

    print(
        "\nReading the gaps:\n"
        "  - delta_peak gap present at grad=off (and grad=on) -> sink #1, the Newton TVF\n"
        "    optimizer's transient vmap/ifft2 Hessian assembly (optimizers.py:196-217).\n"
        "  - residual(graph) gap present ONLY at grad=on -> sink #2, the FFF Q-path's extra\n"
        "    dense [Nh,Nh] matrices retained by the autograd graph (isotropic.py:290-295)."
    )
    if args.device == "cpu" or args.device.startswith("cpu"):
        print(
            "\nNote (CPU only): 'residual(clean)' uses process RSS, which glibc malloc often\n"
            "does not return to the OS after free()/gc.collect() -- a nonzero clean-residual\n"
            "gap here is not reliable evidence of a real leak. Trust the CUDA allocator numbers\n"
            "(torch.cuda.memory_allocated, exact) over CPU RSS for that distinction."
        )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    default_device = "cuda:0" if torch.cuda.is_available() else "cpu"
    p.add_argument("--device", default=default_device,
                    help=f"'cuda:0' or 'cpu' (default: {default_device})")
    p.add_argument("--dtype", default="float32", choices=sorted(_DTYPES))
    p.add_argument("--m", type=int, default=12)
    p.add_argument("--n", type=int, default=12)
    p.add_argument("--nx", type=int, default=128)
    p.add_argument("--ny", type=int, default=128)
    p.add_argument("--truncation", default="circular", choices=["circular", "rectangular"])
    p.add_argument("--wl-count", type=int, default=3)
    p.add_argument("--grad", default="both", choices=["on", "off", "both"],
                    help="which autograd modes to test (default: both)")
    p.add_argument("--newton-chunk-size", type=_chunk_size_type, default=64,
                    help="Factorization.newton_chunk_size for the exact-Newton TVF "
                         "optimizer's Hessian assembly (int, or 'none' to disable "
                         "chunking and restore the original single-shot behavior). "
                         "Default: 64.")
    # Internal flags used to re-invoke this script as an isolated child process.
    p.add_argument("--_child", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--_tvf", type=int, default=0, help=argparse.SUPPRESS)
    p.add_argument("--_grad", type=int, default=0, help=argparse.SUPPRESS)
    return p.parse_args()


def _chunk_size_type(value: str):
    return None if value.lower() == "none" else int(value)


def main() -> None:
    args = parse_args()
    if args._child:
        run_child(args)
    else:
        run_parent(args)


if __name__ == "__main__":
    main()
