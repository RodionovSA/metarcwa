"""
Benchmark torch.linalg.eig on GPU and confirm whether it synchronizes with the CPU.

Two independent sync checks:
  1) torch.cuda.set_sync_debug_mode("warn") -- authoritative *if* the backend's
     sync is instrumented through the dispatcher (MAGMA's internal D2H copy may
     or may not trip it, hence check #2).
  2) Timing test -- launch eig WITHOUT a trailing synchronize and measure how long
     the Python call itself blocks. If the call blocks for ~the full GPU compute
     time, it synced internally. If it returns in ~launch-overhead time, it's async.
     Calibrated against a known-async op (matmul) and a known-sync op (.item()).

Usage:
    python torch_eig_gpu_bench.py
    python torch_eig_gpu_bench.py --sizes 242 450 882 2048 4096 --dtype complex128
    python torch_eig_gpu_bench.py --backend cusolver     # force Xgeev path (torch>=2.10, CUDA>=12.8)
    python torch_eig_gpu_bench.py --backend magma        # force legacy path for comparison
"""

import argparse
import time
import warnings

import torch


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--sizes", type=int, nargs="+",
                   default=[242, 450, 882, 2048, 4096],
                   help="Matrix dimensions n (RCWA: n = 2*M*N harmonics).")
    p.add_argument("--dtype", choices=["complex64", "complex128"],
                   default="complex128")
    p.add_argument("--iters", type=int, default=10, help="Timed iterations per size.")
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--backend", choices=["default", "cusolver", "magma"],
                   default="default",
                   help="Preferred CUDA linalg backend. 'cusolver' targets Xgeev.")
    p.add_argument("--batch-n", type=int, default=512,
                   help="Matrix size for the batched-vs-loop test.")
    p.add_argument("--batch-b", type=int, default=8,
                   help="Batch count for the batched-vs-loop test.")
    return p.parse_args()


def make_matrix(n, batch=None, dtype=torch.complex128, device="cuda"):
    # General (non-Hermitian) complex matrix -> exercises torch.linalg.eig, not eigh.
    shape = (batch, n, n) if batch else (n, n)
    return torch.randn(*shape, dtype=dtype, device=device)


def gpu_time_ms(fn, iters):
    """Average GPU time per call via CUDA events (excludes Python overhead)."""
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def block_test(fn):
    """
    Returns (cpu_block_ms, gpu_ms, ratio).
    cpu_block_ms: wall time the Python call took to return (no trailing sync).
    gpu_ms:       actual GPU compute time for the op (CUDA events).
    ratio:        cpu_block_ms / gpu_ms.  ~1 => call blocked == internal sync.
                                          ~0 => call returned async == no sync.
    """
    torch.cuda.synchronize()
    ev0 = torch.cuda.Event(enable_timing=True)
    ev1 = torch.cuda.Event(enable_timing=True)
    ev0.record()
    t0 = time.perf_counter()
    out = fn()
    t_ret = time.perf_counter()          # CPU time at which the call returned
    ev1.record()
    torch.cuda.synchronize()
    gpu_ms = ev0.elapsed_time(ev1)
    cpu_block_ms = (t_ret - t0) * 1e3
    ratio = cpu_block_ms / gpu_ms if gpu_ms > 0 else float("nan")
    return cpu_block_ms, gpu_ms, ratio, out


def sync_debug_probe(fn):
    """Run fn under sync-debug 'warn' mode; return list of captured sync warnings."""
    captured = []
    try:
        torch.cuda.set_sync_debug_mode("warn")
    except Exception as e:
        return [f"(sync_debug_mode unavailable: {e})"]
    try:
        with warnings.catch_warnings(record=True) as wlist:
            warnings.simplefilter("always")
            fn()
            torch.cuda.synchronize()
        for w in wlist:
            msg = str(w.message).lower()
            if "sync" in msg or "synchron" in msg:
                captured.append(str(w.message))
    finally:
        torch.cuda.set_sync_debug_mode("default")
    return captured


def main():
    args = parse_args()

    if not torch.cuda.is_available():
        print("CUDA not available -- this benchmark requires a GPU. Aborting.")
        return

    dtype = getattr(torch, args.dtype)
    dev = torch.device("cuda")

    if args.backend != "default":
        torch.backends.cuda.preferred_linalg_library(args.backend)

    print("=" * 64)
    print(f"torch            : {torch.__version__}")
    print(f"CUDA (torch)     : {torch.version.cuda}")
    print(f"device           : {torch.cuda.get_device_name(0)}")
    try:
        print(f"linalg backend   : {torch.backends.cuda.preferred_linalg_library()}")
    except Exception:
        print("linalg backend   : (query unavailable)")
    print(f"dtype            : {args.dtype}")
    print("=" * 64)

    # --- Calibration references for the timing-based sync test ----------------
    cal_n = max(args.sizes[len(args.sizes) // 2], 1024)
    A_cal = make_matrix(cal_n, dtype=dtype, device=dev)
    for _ in range(args.warmup):
        A_cal @ A_cal
    torch.cuda.synchronize()

    async_block, async_gpu, async_ratio, _ = block_test(lambda: A_cal @ A_cal)
    sync_block, sync_gpu, sync_ratio, _ = block_test(
        lambda: (A_cal @ A_cal).abs().sum().item()
    )
    print("\nCalibration (ratio = cpu_block / gpu_time):")
    print(f"  matmul        (async ref) : ratio={async_ratio:6.3f}  "
          f"(cpu {async_block:7.2f} ms / gpu {async_gpu:7.2f} ms)")
    print(f"  matmul+.item()(sync  ref) : ratio={sync_ratio:6.3f}  "
          f"(cpu {sync_block:7.2f} ms / gpu {sync_gpu:7.2f} ms)")
    threshold = (async_ratio + sync_ratio) / 2.0
    print(f"  -> sync if eig ratio > {threshold:.3f}\n")

    # --- Per-size eig benchmark + sync verdict --------------------------------
    print(f"{'n':>6} | {'gpu ms':>9} | {'eig/s':>8} | {'cpu_block ms':>12} | "
          f"{'ratio':>6} | verdict")
    print("-" * 70)

    sync_debug_hits = None
    for n in args.sizes:
        A = make_matrix(n, dtype=dtype, device=dev)
        eig = lambda: torch.linalg.eig(A)

        for _ in range(args.warmup):
            eig()
        torch.cuda.synchronize()

        g = gpu_time_ms(eig, args.iters)
        cpu_block, gpu_ms, ratio, (evals, evecs) = block_test(eig)

        # sanity: outputs actually live on GPU, complex (non-Hermitian path)
        assert evals.is_cuda and evecs.is_cuda, "eig output not on CUDA!"
        assert evals.is_complex(), "expected complex eigenvalues"

        verdict = "SYNCS (cpu blocked)" if ratio > threshold else "async (no sync)"
        print(f"{n:>6} | {g:9.3f} | {1000.0/g:8.1f} | {cpu_block:12.2f} | "
              f"{ratio:6.3f} | {verdict}")

        # run the dispatcher-level probe once, on a mid-size matrix
        if sync_debug_hits is None and n >= 512:
            sync_debug_hits = sync_debug_probe(eig)

    # --- sync_debug_mode result -----------------------------------------------
    print("\nsync_debug_mode probe:")
    if not sync_debug_hits:
        print("  no sync warnings captured (either truly async, or backend sync")
        print("  is not surfaced through the dispatcher -- trust the timing test).")
    else:
        for h in sync_debug_hits:
            print(f"  WARNING: {h}")

    # --- Batched vs loop (does batched eig parallelize?) ----------------------
    bn, bb = args.batch_n, args.batch_b
    Abatch = make_matrix(bn, batch=bb, dtype=dtype, device=dev)
    Asingle = make_matrix(bn, dtype=dtype, device=dev)
    for _ in range(args.warmup):
        torch.linalg.eig(Abatch)
        torch.linalg.eig(Asingle)
    torch.cuda.synchronize()

    t_batched = gpu_time_ms(lambda: torch.linalg.eig(Abatch), args.iters)
    t_single = gpu_time_ms(lambda: torch.linalg.eig(Asingle), args.iters)
    speedup = (bb * t_single) / t_batched if t_batched > 0 else float("nan")

    print("\n" + "=" * 64)
    print(f"Batched test (n={bn}, batch={bb}):")
    print(f"  single eig        : {t_single:8.3f} ms")
    print(f"  batched eig       : {t_batched:8.3f} ms  ({t_batched/bb:.3f} ms/matrix)")
    print(f"  loop-equiv (b*1)  : {bb*t_single:8.3f} ms")
    print(f"  batched speedup   : {speedup:5.2f}x  "
          f"({'parallelized' if speedup > 1.3 else 'serialized (no parallel gain)'})")
    print("=" * 64)


if __name__ == "__main__":
    main()
