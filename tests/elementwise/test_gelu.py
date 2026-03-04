"""Element-wise template correctness + performance check.

This mirrors the reduction/transpose prototype scripts and validates:
1. The `ElementWise` schedule template can be applied.
2. Lowered IR contains tile primitives (`T.copy`) and fragment buffers.
3. Numerical correctness against PyTorch.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
import types
from pathlib import Path
from typing import Tuple

import tilelang
import torch
from tilelang import tvm
from tilelang.profiler import do_bench
from tvm import te
from tilelang.schedule.gpu.element_wise import ElementWise

def _build_mod(m: int, n: int, k: int, arch: str):
    """Build and lower a scheduled element-wise module."""
    a = te.placeholder((m, n,), name="a")

    ## implement gelu activation
    c = te.compute(
        (m, n, ),
        lambda i, j: a[i, j] * 0.5 * (1.0 + te.tanh(0.7978845608028654 * (a[i, j] + 0.044715 * a[i, j] * a[i, j] * a[i, j]))),
        name="c",
    )

    func = te.create_prim_func([a, c])

    target = tvm.target.cuda(arch=arch)
    sch = ElementWise().apply(func, target, False)
    if sch is None:
        raise RuntimeError("ElementWise schedule rule returned None for this workload.")

    print("=== Scheduled IR ===")
    print(sch.mod)

    mod = sch.mod
    mod = tvm.tir.transform.Simplify()(mod)
    mod = tvm.tir.transform.ConvertBlocksToOpaque()(mod)
    mod = tilelang.transform.ReserveRootBlock()(mod)

    print("=== Lowered IR ===")
    print(mod)

    lowered_ir = str(mod)
    if "T.copy(" not in lowered_ir:
        raise RuntimeError("Expected T.copy in lowered IR, but it was not found.")
    if "local.fragment" not in lowered_ir:
        raise RuntimeError("Expected local.fragment buffers in lowered IR, but none were found.")
    if 'launch_thread("threadIdx.x"' not in lowered_ir:
        raise RuntimeError("Expected launch_thread(threadIdx.x) in lowered IR, but it was not found.")

    return mod


def build_and_run(m: int, n: int, k: int, arch: str, bench_backend: str) -> Tuple[float, float]:
    """Build an element-wise kernel, run correctness check, and benchmark."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to run this script.")

    mod = _build_mod(m, n, k, arch)
    kernel = tilelang.compile(mod["main"])

    torch.manual_seed(0)
    a_torch = torch.randn((m, n), device="cuda", dtype=torch.float32)
    c_torch = torch.empty((m, n), device="cuda", dtype=torch.float32)

    kernel(a_torch, c_torch)

    @torch.compile()
    def fn_torch(a):
        return torch.nn.functional.gelu(a, approximate='tanh')

    torch.testing.assert_close(c_torch, fn_torch(a_torch), rtol=1e-4, atol=1e-4)
    print("\033[92mCorrectness check passed.\033[0m")

    tilelang_time = do_bench(lambda: kernel(a_torch, c_torch), backend=bench_backend)
    torch_time = do_bench(lambda: fn_torch(a_torch), backend=bench_backend)

    total_bytes = (2 * m * n) * 4  # 2 reads in fp32
    tilelang_gbps = total_bytes / (tilelang_time * 1e-3) / 1e9
    torch_gbps = total_bytes / (torch_time * 1e-3) / 1e9

    print(f"TileLang time: {tilelang_time:.6f} ms, BW: {tilelang_gbps:.2f} GB/s")
    print(f"Torch   time: {torch_time:.6f} ms, BW: {torch_gbps:.2f} GB/s")
    print(f"Speedup (torch/tilelang): {torch_time / tilelang_time:.4f}x")
    return tilelang_time, torch_time


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check element-wise template correctness and performance.")
    parser.add_argument("--m", type=int, default=128, help="First input dimension.")
    parser.add_argument("--n", type=int, default=16384, help="Second input dimension.")
    parser.add_argument("--k", type=int, default=512, help="Third input dimension.")
    parser.add_argument("--arch", type=str, default="sm_90a", help='CUDA arch string, e.g. "sm_90a".')
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Only check schedule + lowering IR; skip CUDA compile/run/benchmark.",
    )
    parser.add_argument(
        "--bench-backend",
        type=str,
        default="cupti",
        choices=["cupti", "torch"],
        help="Profiler backend used by tilelang.profiler.do_bench.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.check_only:
        _build_mod(args.m, args.n, args.k, args.arch)
        print("\033[92mSchedule/lowering check passed.\033[0m")
    else:
        build_and_run(args.m, args.n, args.k, args.arch, args.bench_backend)
