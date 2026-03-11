"""GEMV template correctness + performance check.

Validates:
1. The `GEMV` schedule template lowers GEMV TE expressions to tile-style IR.
2. Numerical correctness against PyTorch.
3. Performance comparison against torch.compile.
"""

from __future__ import annotations

import argparse
from typing import Tuple

import tilelang
import torch
from tilelang import tvm
from tilelang.profiler import do_bench
from tvm import te, tir

from tilelang.schedule.gpu.gemv import GEMV


def _lower_mod(mod):
    """Apply the lowering passes for tile-primitive GEMV."""
    mod = tir.transform.Simplify()(mod)
    mod = tir.transform.LowerInitBlock()(mod)
    mod = tir.transform.ConvertBlocksToOpaque()(mod)
    mod = tilelang.transform.ReserveRootBlock()(mod)
    return mod


def _build_mod(M: int, N: int, arch: str):
    """Build a scheduled GEMV module: C[i] = sum_k A[i, k] * B[k]."""
    A = te.placeholder((M, N), name="A", dtype="float32")
    B = te.placeholder((N,), name="B", dtype="float32")
    k = te.reduce_axis((0, N), name="k")
    C = te.compute((M,), lambda i: te.sum(A[i, k] * B[k], axis=k), name="C")

    func = te.create_prim_func([A, B, C])

    target = tvm.target.cuda(arch=arch)
    sch = GEMV().apply(func, target, False)
    if sch is None:
        raise RuntimeError("GEMV schedule rule returned None for this workload.")

    print("=== Scheduled IR ===")
    print(sch.mod)

    mod = _lower_mod(sch.mod)

    print("=== Lowered IR ===")
    print(mod)

    lowered_ir = str(mod)
    if "T.copy(" not in lowered_ir:
        raise RuntimeError("Expected tiled T.copy staging in lowered IR, but none was found.")
    if "shared.dyn" not in lowered_ir:
        raise RuntimeError("Expected shared.dyn staging in lowered IR, but none was found.")
    if "local.fragment" not in lowered_ir:
        raise RuntimeError("Expected local.fragment buffers in lowered IR, but none were found.")
    if "T.parallel(" not in lowered_ir:
        raise RuntimeError("Expected tile-level T.parallel loops in lowered IR, but none were found.")
    if "tvm_thread_allreduce" in lowered_ir:
        raise RuntimeError("Unexpected cross-thread allreduce in tile-style GEMV lowering.")

    return mod


def _build_mod_transposed(M: int, N: int, arch: str):
    """Build GEMV with transposed weight: C[i] = sum_k A[k] * B[i, k]."""
    A = te.placeholder((N,), name="A", dtype="float32")
    B = te.placeholder((M, N), name="B", dtype="float32")
    k = te.reduce_axis((0, N), name="k")
    C = te.compute((M,), lambda i: te.sum(A[k] * B[i, k], axis=k), name="C")

    func = te.create_prim_func([A, B, C])

    target = tvm.target.cuda(arch=arch)
    sch = GEMV().apply(func, target, False)
    if sch is None:
        raise RuntimeError("GEMV schedule rule returned None for transposed workload.")

    print("=== Scheduled IR (transposed) ===")
    print(sch.mod)

    mod = _lower_mod(sch.mod)

    print("=== Lowered IR (transposed) ===")
    print(mod)

    lowered_ir = str(mod)
    if "T.copy(" not in lowered_ir:
        raise RuntimeError("Expected tiled T.copy staging in transposed lowered IR, but none was found.")
    if "shared.dyn" not in lowered_ir:
        raise RuntimeError("Expected shared.dyn staging in transposed lowered IR, but none was found.")
    if "local.fragment" not in lowered_ir:
        raise RuntimeError("Expected local.fragment buffers in transposed lowered IR, but none were found.")
    if "tvm_thread_allreduce" in lowered_ir:
        raise RuntimeError("Unexpected cross-thread allreduce in transposed tile-style GEMV lowering.")

    return mod


def _build_mod_batched(B: int, M: int, N: int, arch: str):
    """Build batched GEMV: C[b, i] = sum_k A[b, i, k] * X[b, k]."""
    A_t = te.placeholder((B, M, N), name="A", dtype="float32")
    X_t = te.placeholder((B, N), name="X", dtype="float32")
    k = te.reduce_axis((0, N), name="k")
    C_t = te.compute((B, M), lambda b, i: te.sum(A_t[b, i, k] * X_t[b, k], axis=k), name="C")

    func = te.create_prim_func([A_t, X_t, C_t])

    target = tvm.target.cuda(arch=arch)
    sch = GEMV().apply(func, target, False)
    if sch is None:
        raise RuntimeError("GEMV schedule rule returned None for batched workload.")

    print("=== Scheduled IR (batched) ===")
    print(sch.mod)

    mod = _lower_mod(sch.mod)

    print("=== Lowered IR (batched) ===")
    print(mod)

    lowered_ir = str(mod)
    if "T.copy(" not in lowered_ir:
        raise RuntimeError("Expected tiled T.copy staging in batched lowered IR, but none was found.")
    if "shared.dyn" not in lowered_ir:
        raise RuntimeError("Expected shared.dyn staging in batched lowered IR, but none was found.")
    if "local.fragment" not in lowered_ir:
        raise RuntimeError("Expected local.fragment buffers in batched lowered IR, but none were found.")
    if "tvm_thread_allreduce" in lowered_ir:
        raise RuntimeError("Unexpected cross-thread allreduce in batched tile-style GEMV lowering.")

    return mod


def build_and_run(M: int, N: int, arch: str, bench_backend: str) -> Tuple[float, float]:
    """Build a GEMV kernel, run correctness check, and benchmark."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to run this script.")

    mod = _build_mod(M, N, arch)
    kernel = tilelang.compile(mod["main"])

    torch.manual_seed(0)
    A_torch = torch.randn((M, N), device="cuda", dtype=torch.float32)
    B_torch = torch.randn((N,), device="cuda", dtype=torch.float32)
    C_torch = torch.empty((M,), device="cuda", dtype=torch.float32)

    kernel(A_torch, B_torch, C_torch)

    @torch.compile()
    def fn_torch(A, B):
        return A @ B

    ref = fn_torch(A_torch, B_torch)
    torch.testing.assert_close(C_torch, ref, rtol=1e-3, atol=1e-3)
    print("\033[92mCorrectness check passed (standard GEMV).\033[0m")

    # Benchmark
    tilelang_time = do_bench(lambda: kernel(A_torch, B_torch, C_torch), backend=bench_backend)
    torch_time = do_bench(lambda: fn_torch(A_torch, B_torch), backend=bench_backend)

    total_bytes = (M * N + N + M) * 4  # fp32 read A, read B, write C
    tilelang_gbps = total_bytes / (tilelang_time * 1e-3) / 1e9
    torch_gbps = total_bytes / (torch_time * 1e-3) / 1e9

    print(f"TileLang time: {tilelang_time:.6f} ms, BW: {tilelang_gbps:.2f} GB/s")
    print(f"Torch   time: {torch_time:.6f} ms, BW: {torch_gbps:.2f} GB/s")
    print(f"Speedup (torch/tilelang): {torch_time / tilelang_time:.4f}x")
    return tilelang_time, torch_time


def build_and_run_transposed(M: int, N: int, arch: str, bench_backend: str) -> Tuple[float, float]:
    """Build transposed GEMV: y = W @ x where W is [M, N], x is [N]."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to run this script.")

    mod = _build_mod_transposed(M, N, arch)
    kernel = tilelang.compile(mod["main"])

    torch.manual_seed(42)
    x_torch = torch.randn((N,), device="cuda", dtype=torch.float32)
    W_torch = torch.randn((M, N), device="cuda", dtype=torch.float32)
    y_torch = torch.empty((M,), device="cuda", dtype=torch.float32)

    kernel(x_torch, W_torch, y_torch)

    @torch.compile()
    def fn_torch(x, W):
        return W @ x

    ref = fn_torch(x_torch, W_torch)
    torch.testing.assert_close(y_torch, ref, rtol=1e-3, atol=1e-3)
    print("\033[92mCorrectness check passed (transposed GEMV).\033[0m")

    tilelang_time = do_bench(lambda: kernel(x_torch, W_torch, y_torch), backend=bench_backend)
    torch_time = do_bench(lambda: fn_torch(x_torch, W_torch), backend=bench_backend)

    total_bytes = (M * N + N + M) * 4
    tilelang_gbps = total_bytes / (tilelang_time * 1e-3) / 1e9
    torch_gbps = total_bytes / (torch_time * 1e-3) / 1e9

    print(f"TileLang time: {tilelang_time:.6f} ms, BW: {tilelang_gbps:.2f} GB/s")
    print(f"Torch   time: {torch_time:.6f} ms, BW: {torch_gbps:.2f} GB/s")
    print(f"Speedup (torch/tilelang): {torch_time / tilelang_time:.4f}x")
    return tilelang_time, torch_time


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check GEMV template correctness and performance.")
    parser.add_argument("--m", type=int, default=4096, help="Number of output elements (rows of weight matrix).")
    parser.add_argument("--n", type=int, default=4096, help="Reduction dimension (cols of weight matrix).")
    parser.add_argument("--arch", type=str, default="sm_90a", help='CUDA arch string, e.g. "sm_90a".')
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Only check schedule + lowering IR; skip CUDA compile/run/benchmark.",
    )
    parser.add_argument(
        "--bench-backend",
        type=str,
        default="event",
        choices=["event", "cupti"],
        help="Profiler backend used by tilelang.profiler.do_bench.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.check_only:
        print("--- Standard GEMV ---")
        _build_mod(args.m, args.n, args.arch)
        print("\033[92mSchedule/lowering check passed (standard).\033[0m\n")

        print("--- Transposed GEMV ---")
        _build_mod_transposed(args.m, args.n, args.arch)
        print("\033[92mSchedule/lowering check passed (transposed).\033[0m\n")

        print("--- Batched GEMV ---")
        _build_mod_batched(2, args.m, args.n, args.arch)
        print("\033[92mSchedule/lowering check passed (batched).\033[0m\n")
    else:
        print("=== Standard GEMV ===")
        build_and_run(args.m, args.n, args.arch, args.bench_backend)
        print()
        print("=== Transposed GEMV ===")
        build_and_run_transposed(args.m, args.n, args.arch, args.bench_backend)
