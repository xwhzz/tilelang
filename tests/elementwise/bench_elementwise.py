"""Benchmark the ElementWise schedule template across many shapes.

Usage:
    python tests/elementwise/bench_elementwise.py
    python tests/elementwise/bench_elementwise.py --arch sm_80
    python tests/elementwise/bench_elementwise.py --check-only
"""

from __future__ import annotations

import argparse

import tilelang
import torch
from tilelang import tvm
from tilelang.profiler import do_bench
from tvm import te

from tilelang.schedule.gpu.element_wise import ElementWise

# ---------------------------------------------------------------------------
# Workload shapes: (M, N, K) tuples covering diverse scenarios
# ---------------------------------------------------------------------------
SHAPES = [
    # --- Power-of-2, fragment-friendly ---
    (1, 1, 8192),
    (1, 1, 65536),
    (64, 128, 512),
    (128, 128, 128),
    (1, 1024, 1024),
    (16, 256, 256),
    # --- Large, power-of-2 ---
    (64, 128, 1024),
    (32, 256, 2048),
    (1, 1, 1048576),
    # --- Non-power-of-2 but divisible (fragment OK) ---
    (64, 128, 16384),
    (1, 100, 8192),
    (10, 10, 8192),
    # --- Non-divisible suffix (fragment disabled) ---
    (64, 128, 16383),
    (1, 127, 8191),
    (1, 1, 1000003),    # prime
    (7, 13, 17),        # small primes
    (1, 1, 999),
    # --- Edge cases ---
    (1, 1, 1),
    (1, 1, 33),
    (1, 1, 1023),
    # --- Practical shapes ---
    (1, 4096, 4096),    # transformer hidden dim
    (8, 1024, 1024),
    (32, 512, 512),
    (1, 32000, 1),      # vocab-size-like
    (1, 1, 131072),     # 128K sequence
]


def _build_mod(m: int, n: int, k: int, arch: str):
    """Build and lower a scheduled element-wise module. Returns (mod, has_fragment)."""
    a = te.placeholder((m, n, k), name="a")
    b = te.placeholder((m, n, k), name="b")
    scale = te.placeholder((m, n, k), name="scale")
    tmp = te.compute(
        (m, n, k),
        lambda i, j, kk: a[i, j, kk] + b[i, j, kk] * scale[i, j, kk],
        name="tmp",
    )
    c = te.compute(
        (m, n, k),
        lambda i, j, kk: te.max(tmp[i, j, kk], 0.0),
        name="c",
    )
    func = te.create_prim_func([a, b, scale, c])

    target = tvm.target.cuda(arch=arch)
    sch = ElementWise().apply(func, target, False)
    if sch is None:
        return None, False

    mod = sch.mod
    mod = tvm.tir.transform.Simplify()(mod)
    mod = tvm.tir.transform.ConvertBlocksToOpaque()(mod)
    mod = tilelang.transform.ReserveRootBlock()(mod)

    lowered_ir = str(mod)
    has_fragment = "T.copy(" in lowered_ir and "local.fragment" in lowered_ir
    return mod, has_fragment


def _run_one(m: int, n: int, k: int, arch: str, bench_backend: str, check_only: bool):
    """Build, verify correctness, and benchmark one shape. Returns a result dict."""
    shape_str = f"({m}, {n}, {k})"
    total_elems = m * n * k
    total_bytes = (3 + 1) * total_elems * 4  # 3 reads + 1 write, fp32

    try:
        mod, has_fragment = _build_mod(m, n, k, arch)
    except Exception as e:
        return {"shape": shape_str, "elems": total_elems, "fragment": "?",
                "status": f"BUILD FAIL: {e}", "tl_ms": None, "torch_ms": None,
                "tl_gbps": None, "torch_gbps": None, "speedup": None}

    if mod is None:
        return {"shape": shape_str, "elems": total_elems, "fragment": "-",
                "status": "SKIP (rule returned None)", "tl_ms": None, "torch_ms": None,
                "tl_gbps": None, "torch_gbps": None, "speedup": None}

    if check_only:
        return {"shape": shape_str, "elems": total_elems, "fragment": "Y" if has_fragment else "N",
                "status": "OK", "tl_ms": None, "torch_ms": None,
                "tl_gbps": None, "torch_gbps": None, "speedup": None}

    try:
        kernel = tilelang.compile(mod["main"])
    except Exception as e:
        return {"shape": shape_str, "elems": total_elems, "fragment": "Y" if has_fragment else "N",
                "status": f"COMPILE FAIL: {e}", "tl_ms": None, "torch_ms": None,
                "tl_gbps": None, "torch_gbps": None, "speedup": None}

    # Correctness
    torch.manual_seed(0)
    a_t = torch.randn((m, n, k), device="cuda", dtype=torch.float32)
    b_t = torch.randn((m, n, k), device="cuda", dtype=torch.float32)
    s_t = torch.randn((m, n, k), device="cuda", dtype=torch.float32)
    c_t = torch.empty((m, n, k), device="cuda", dtype=torch.float32)

    try:
        kernel(a_t, b_t, s_t, c_t)
        ref = torch.relu(a_t + b_t * s_t)
        torch.testing.assert_close(c_t, ref, rtol=1e-4, atol=1e-4)
    except Exception as e:
        return {"shape": shape_str, "elems": total_elems, "fragment": "Y" if has_fragment else "N",
                "status": f"WRONG: {e}", "tl_ms": None, "torch_ms": None,
                "tl_gbps": None, "torch_gbps": None, "speedup": None}

    # Benchmark
    tl_ms = do_bench(lambda: kernel(a_t, b_t, s_t, c_t), backend=bench_backend)
    torch_fn = torch.compile(lambda a, b, s: torch.relu(a + b * s))
    torch_fn(a_t, b_t, s_t)  # warmup compile
    torch_ms = do_bench(lambda: torch_fn(a_t, b_t, s_t), backend=bench_backend)

    tl_gbps = total_bytes / (tl_ms * 1e-3) / 1e9
    torch_gbps = total_bytes / (torch_ms * 1e-3) / 1e9
    speedup = torch_ms / tl_ms if tl_ms > 0 else float("inf")

    return {"shape": shape_str, "elems": total_elems, "fragment": "Y" if has_fragment else "N",
            "status": "PASS", "tl_ms": tl_ms, "torch_ms": torch_ms,
            "tl_gbps": tl_gbps, "torch_gbps": torch_gbps, "speedup": speedup}


def main():
    parser = argparse.ArgumentParser(description="Benchmark ElementWise schedule template.")
    parser.add_argument("--arch", type=str, default="sm_90a", help='CUDA arch, e.g. "sm_90a".')
    parser.add_argument("--check-only", action="store_true", help="Only check build/lowering, skip GPU run.")
    parser.add_argument("--bench-backend", type=str, default="event", choices=["event", "cupti"])
    args = parser.parse_args()

    if not args.check_only and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to run benchmarks.")

    tilelang.disable_cache()

    results = []
    for m, n, k in SHAPES:
        result = _run_one(m, n, k, args.arch, args.bench_backend, args.check_only)
        results.append(result)
        # Print progress
        r = result
        if r["tl_ms"] is not None:
            print(f"  {r['shape']:>22s}  frag={r['fragment']}  "
                  f"tl={r['tl_ms']:8.4f}ms ({r['tl_gbps']:7.1f} GB/s)  "
                  f"torch={r['torch_ms']:8.4f}ms ({r['torch_gbps']:7.1f} GB/s)  "
                  f"speedup={r['speedup']:.3f}x  {r['status']}")
        else:
            print(f"  {r['shape']:>22s}  frag={r['fragment']}  {r['status']}")

    # Summary table
    print("\n" + "=" * 110)
    print(f"{'Shape':>22s}  {'Frag':>4s}  {'TL (ms)':>10s}  {'TL GB/s':>9s}  "
          f"{'Torch (ms)':>10s}  {'Torch GB/s':>10s}  {'Speedup':>8s}  {'Status':>8s}")
    print("-" * 110)
    for r in results:
        tl_str = f"{r['tl_ms']:10.4f}" if r["tl_ms"] is not None else "       N/A"
        tl_bw = f"{r['tl_gbps']:9.1f}" if r["tl_gbps"] is not None else "      N/A"
        to_str = f"{r['torch_ms']:10.4f}" if r["torch_ms"] is not None else "       N/A"
        to_bw = f"{r['torch_gbps']:10.1f}" if r["torch_gbps"] is not None else "       N/A"
        sp_str = f"{r['speedup']:8.3f}" if r["speedup"] is not None else "     N/A"
        print(f"{r['shape']:>22s}  {r['fragment']:>4s}  {tl_str}  {tl_bw}  {to_str}  {to_bw}  {sp_str}  {r['status']:>8s}")
    print("=" * 110)

    # Stats
    passed = [r for r in results if r["status"] == "PASS"]
    failed = [r for r in results if r["status"] not in ("PASS", "OK") and "SKIP" not in r["status"]]
    skipped = [r for r in results if "SKIP" in r["status"]]
    print(f"\nTotal: {len(results)}  Passed: {len(passed)}  Failed: {len(failed)}  Skipped: {len(skipped)}")
    if passed:
        avg_speedup = sum(r["speedup"] for r in passed) / len(passed)
        wins = sum(1 for r in passed if r["speedup"] >= 1.0)
        print(f"Average speedup: {avg_speedup:.3f}x  Wins: {wins}/{len(passed)}")


if __name__ == "__main__":
    main()
