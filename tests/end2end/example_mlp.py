"""End-to-end MLP example using tilelang.torch_compile (VM first, JIT fallback)."""

from __future__ import annotations

import argparse

import torch
import torch.nn.functional as F

from tilelang.profiler import do_bench
from tilelang.torch_compile import torch_compile


def _dtype_tolerance(dtype: str, accum_dtype: str) -> tuple[float, float]:
    if dtype == "float16" and accum_dtype == "float32":
        return 2e-2, 2e-2
    if accum_dtype == "float32":
        return 2e-3, 2e-3
    if accum_dtype == "float16":
        return 1e-1, 1e-1
    return 1e-3, 1e-3


def _build_reference(dtype: str, accum_dtype: str):
    @torch.compile()
    def _reference(x: torch.Tensor, w1: torch.Tensor, w2: torch.Tensor) -> torch.Tensor:
        if accum_dtype != dtype:
            x_acc = x.to(getattr(torch, accum_dtype))
            w1_acc = w1.to(getattr(torch, accum_dtype))
            w2_acc = w2.to(getattr(torch, accum_dtype))
        else:
            x_acc = x
            w1_acc = w1
            w2_acc = w2
        return w2_acc @ torch.relu(w1_acc @ x_acc)

    return _reference


def _build_compiled_mlp(
    arch: str | None,
    executor: str,
    dump_vm_code: bool,
    dtype: str,
    accum_dtype: str,
):
    torch_dtype = getattr(torch, dtype)
    torch_accum_dtype = getattr(torch, accum_dtype)

    @torch_compile(arch=arch, executor=executor, dump_vm_code=dump_vm_code)
    def mlp(x: torch.Tensor, w1: torch.Tensor, w2: torch.Tensor) -> torch.Tensor:
        if torch_accum_dtype != torch_dtype:
            x_acc = x.to(torch_dtype)
            w1_acc = w1.to(torch_dtype)
            w2_acc = w2.to(torch_accum_dtype)
        else:
            x_acc, w1_acc, w2_acc = x, w1, w2
        lv0 = torch.matmul(w1_acc, x_acc)
        lv1 = F.relu(lv0)
        lv2 = torch.matmul(w2_acc, lv1)
        return lv2

    return mlp


def check_only(dim: int, arch: str | None, dtype: str, accum_dtype: str, executor: str, dump_vm_code: bool) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to run this script.")

    torch_dtype = getattr(torch, dtype)
    torch.manual_seed(0)
    x = torch.randn((dim,), device="cuda", dtype=torch_dtype)
    w1 = torch.randn((dim, dim), device="cuda", dtype=torch_dtype)
    w2 = torch.randn((dim, dim), device="cuda", dtype=torch_dtype)

    mlp = _build_compiled_mlp(arch, executor, dump_vm_code, dtype, accum_dtype)
    runner = mlp.compile(x, w1, w2)

    print("=== Scheduled Relax Module ===")
    print(runner.scheduled_mod.script())
    for call in runner.call_sequence:
        print(f"=== Scheduled TIR ({call.gv_name}) ===")
        print(runner.scheduled_mod[call.gv_name].script())


def build_and_run(
    dim: int,
    arch: str | None,
    dtype: str,
    accum_dtype: str,
    bench_backend: str,
    executor: str,
    dump_vm_code: bool,
) -> tuple[float, float]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to run this script.")

    torch_dtype = getattr(torch, dtype)
    torch.manual_seed(0)
    x = torch.randn((dim,), device="cuda", dtype=torch_dtype)
    w1 = torch.randn((dim, dim), device="cuda", dtype=torch_dtype)
    w2 = torch.randn((dim, dim), device="cuda", dtype=torch_dtype)

    mlp = _build_compiled_mlp(arch, executor, dump_vm_code, dtype, accum_dtype)
    runner = mlp.compile(x, w1, w2)
    tilelang_out = runner(x, w1, w2)

    ref_fn = _build_reference(dtype, accum_dtype)
    ref_out = ref_fn(x, w1, w2)

    rtol, atol = _dtype_tolerance(dtype, accum_dtype)
    torch.testing.assert_close(tilelang_out, ref_out, rtol=rtol, atol=atol)
    print(f"\033[92mCorrectness check passed ({runner.active_executor}).\033[0m")

    if runner.active_executor == "vm":
        tilelang_time = do_bench(lambda: runner(x, w1, w2), backend=bench_backend)
    else:
        tilelang_time = do_bench(lambda: runner(x, w1, w2), backend=bench_backend)
    torch_time = do_bench(lambda: ref_fn(x, w1, w2), backend=bench_backend)
    print(f"tilelang.torch_compile time: {tilelang_time:.6f} ms ({runner.active_executor}), torch.compile time: {torch_time:.6f} ms")
    return tilelang_time, torch_time


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="tilelang.torch_compile MLP example.")
    parser.add_argument("--arch", type=str, default=None, help='CUDA arch string, e.g. "sm_90a".')
    parser.add_argument("--dim", type=int, default=4096, help="Hidden/input/output size.")
    parser.add_argument(
        "--dtype",
        type=str,
        default="float32",
        choices=["float16", "float32"],
        help="Input and weight storage dtype.",
    )
    parser.add_argument(
        "--accum-dtype",
        type=str,
        default="float32",
        choices=["float16", "float32"],
        help="Internal matmul accumulation / output dtype in the graph.",
    )
    parser.add_argument("--bench-backend", type=str, default="event", choices=["event", "cuda"])
    parser.add_argument(
        "--executor",
        type=str,
        default="auto",
        choices=["auto", "vm", "jit"],
        help="Execution path: Relax VM, JIT fallback, or auto-select (VM first).",
    )
    parser.add_argument(
        "--dump-vm-code",
        action="store_true",
        help="Dump VM text/python plus generated host/device source from VM build.",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Only compile and print scheduled Relax/TIR, skip execution and benchmark.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.check_only:
        check_only(args.dim, args.arch, args.dtype, args.accum_dtype, args.executor, args.dump_vm_code)
        print("\033[92mCompilation check passed.\033[0m")
    else:
        build_and_run(
            args.dim,
            args.arch,
            args.dtype,
            args.accum_dtype,
            args.bench_backend,
            args.executor,
            args.dump_vm_code,
        )
