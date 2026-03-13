"""End-to-end Relax GEMV MLP test using TileLang schedule rules.

This test treats Relax graph IR as the frontend input, applies the default
TileLang GPU schedule stack, and prefers launching the scheduled module through
the Relax VM. A Python per-kernel fallback remains available for scheduled TIR
that still uses TileLang-specific scopes unsupported by the generic VM build
path.

It validates:
1. Schedule selection on a fused Relax graph.
2. Numerical correctness against `torch.compile`.
3. End-to-end latency comparison against `torch.compile`.
"""

from __future__ import annotations

import argparse
from typing import Dict, List, Sequence, Tuple

import tilelang
import tilelang.relax as tl_relax
import torch
from tilelang import tvm
from tilelang.profiler import do_bench
from tilelang.schedule.gpu import default_schedule_rules
from tvm import relax, tir
from tvm.ir import GlobalVar
from tvm.script import relax as R
from tvm.script.ir_builder import IRBuilder
from tvm.script.ir_builder import relax as I


def _dump_runtime_module_sources(
    rt_mod: tvm.runtime.Module, prefix: str = "root", seen: set[int] | None = None
) -> None:
    if seen is None:
        seen = set()

    module_id = id(rt_mod)
    if module_id in seen:
        return
    seen.add(module_id)

    try:
        source = rt_mod.inspect_source()
    except Exception:
        source = None

    if source:
        print(f"=== Runtime Module Source ({prefix}, kind={rt_mod.kind}) ===")
        print(str(source))

    for index, child in enumerate(rt_mod.imports):
        _dump_runtime_module_sources(child, f"{prefix}.imports[{index}]", seen)


def _dump_vm_executable(executable) -> None:
    print("=== VM Text ===")
    print(executable.as_text())
    print("=== VM Python ===")
    print(executable.as_python())
    _dump_runtime_module_sources(executable.mod)


def _build_relax_gemv_mlp(dim: int, dtype: str, accum_dtype: str) -> tvm.IRModule:
    with IRBuilder() as builder:
        with I.function():
            R.func_name("main")
            x = R.arg("x", R.Tensor((dim,), dtype))
            w1 = R.arg("w1", R.Tensor((dim, dim), dtype))
            w2 = R.arg("w2", R.Tensor((dim, dim), dtype))
            with R.dataflow() as frame:
                if accum_dtype != dtype:
                    x_acc = R.emit(R.astype(x, accum_dtype))
                    w1_acc = R.emit(R.astype(w1, accum_dtype))
                    w2_acc = R.emit(R.astype(w2, accum_dtype))
                else:
                    x_acc, w1_acc, w2_acc = x, w1, w2

                lv0 = R.emit(R.matmul(w1_acc, x_acc, out_dtype=accum_dtype))
                lv1 = R.emit(R.nn.relu(lv0))
                lv2 = R.emit(R.matmul(w2_acc, lv1, out_dtype=accum_dtype))
                R.output(lv2)
            R.func_ret_value(frame.output_vars[0])
    return tvm.IRModule({"main": builder.get()})


def _schedule_relax_module(mod: tvm.IRModule, arch: str) -> tvm.IRModule:
    target = tvm.target.cuda(arch=arch)
    mod = relax.transform.LegalizeOps()(mod)
    mod = relax.transform.AnnotateTIROpPattern()(mod)
    mod = relax.transform.FoldConstant()(mod)
    mod = relax.transform.FuseOps()(mod)
    mod = relax.transform.FuseTIR()(mod)
    with target:
        mod = tvm.dlight.ApplyDefaultSchedule(*default_schedule_rules())(mod)
    return mod


def _extract_call_sequence(main_func: relax.Function) -> List[Tuple[str, str, List[str]]]:
    if not isinstance(main_func.body, relax.SeqExpr):
        raise RuntimeError("Expected Relax main body to be a SeqExpr.")
    if len(main_func.body.blocks) != 1 or not isinstance(main_func.body.blocks[0], relax.DataflowBlock):
        raise RuntimeError("Expected Relax main body to contain one DataflowBlock.")

    calls: List[Tuple[str, str, List[str]]] = []
    for binding in main_func.body.blocks[0].bindings:
        value = binding.value
        if not isinstance(value, relax.Call) or value.op.name != "relax.call_tir":
            continue
        callee = value.args[0]
        if not isinstance(callee, GlobalVar):
            raise RuntimeError("Expected call_tir callee to be a GlobalVar.")
        arg_names: List[str] = []
        for arg in value.args[1]:
            if not isinstance(arg, relax.Var):
                raise RuntimeError(f"Unsupported call_tir arg type: {type(arg)}")
            arg_names.append(arg.name_hint)
        calls.append((binding.var.name_hint, callee.name_hint, arg_names))

    if not calls:
        raise RuntimeError("No call_tir kernels were found in the scheduled Relax graph.")
    return calls


def _lower_primfunc_for_tilelang(func: tir.PrimFunc, name: str) -> tir.PrimFunc:
    func = func.with_attr("global_symbol", name)
    if "tvm_thread_allreduce" in func.script():
        return func

    mod = tvm.IRModule({name: func})
    mod = tir.transform.Simplify()(mod)
    mod = tir.transform.ForceNarrowIndexToInt32()(mod)
    mod = tir.transform.LowerInitBlock()(mod)
    mod = tir.transform.ConvertBlocksToOpaque()(mod)
    mod = tilelang.transform.ReserveRootBlock()(mod)
    return mod[name]


def _output_buffer_info(func: tir.PrimFunc) -> Tuple[Tuple[int, ...], str]:
    output_buffer = func.buffer_map[list(func.params)[-1]]
    return tuple(int(extent) for extent in output_buffer.shape), output_buffer.dtype


def _dtype_tolerance(dtype: str, accum_dtype: str) -> Tuple[float, float]:
    if dtype == "float16" and accum_dtype == "float32":
        return 2e-2, 2e-2
    if accum_dtype == "float32":
        return 2e-3, 2e-3
    if accum_dtype == "float16":
        return 1e-1, 1e-1
    return 1e-3, 1e-3


def _assert_scheduled_kernels(mod: tvm.IRModule, calls: Sequence[Tuple[str, str, List[str]]]) -> None:
    has_cross_thread_reduction = False
    for _, gv_name, _ in calls:
        text = mod[gv_name].script()
        has_tile_staging = "T.copy(" in text and "local.fragment" in text
        has_thread_reduction = "tvm_thread_allreduce" in text
        has_parallelism = (
            'launch_thread("threadIdx.x"' in text
            or "thread_binding" in text
            or "T.parallel(" in text
        )
        if not (has_tile_staging or has_thread_reduction):
            raise RuntimeError(
                f"Expected either tile staging or cross-thread reduction in {gv_name}, but found neither."
            )
        if not has_parallelism:
            raise RuntimeError(f"Expected explicit GPU parallelism in {gv_name}, but found none.")
        has_cross_thread_reduction = has_cross_thread_reduction or has_thread_reduction

    if not has_cross_thread_reduction:
        raise RuntimeError("Expected at least one GEMV-style cross-thread reduction kernel.")


def _compile_calls(
    mod: tvm.IRModule, calls: Sequence[Tuple[str, str, List[str]]]
) -> Dict[str, tilelang.jit.JITKernel]:
    kernels = {}
    for _, gv_name, _ in calls:
        kernels[gv_name] = tilelang.compile(_lower_primfunc_for_tilelang(mod[gv_name], gv_name))
    return kernels


def _run_tilelang_graph(
    mod: tvm.IRModule,
    calls: Sequence[Tuple[str, str, List[str]]],
    kernels: Dict[str, tilelang.jit.JITKernel],
    inputs: Dict[str, torch.Tensor],
) -> torch.Tensor:
    env = dict(inputs)
    for out_name, gv_name, arg_names in calls:
        shape, out_dtype = _output_buffer_info(mod[gv_name])
        out = torch.empty(shape, device="cuda", dtype=getattr(torch, out_dtype))
        kernels[gv_name](*[env[arg_name] for arg_name in arg_names], out)
        env[out_name] = out
    return env[calls[-1][0]]


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


def _normalize_output_to_torch(value: torch.Tensor, dtype: str) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value
    if hasattr(value, "numpy"):
        return torch.from_numpy(value.numpy()).to(device="cuda", dtype=getattr(torch, dtype))
    raise TypeError(f"Unsupported VM output type: {type(value)}")


def _build_vm_runner(mod: tvm.IRModule, arch: str, dump_vm_code: bool):
    target = tvm.target.cuda(arch=arch)
    executable = tl_relax.build(mod, target=target)
    if dump_vm_code:
        _dump_vm_executable(executable)
    vm = relax.VirtualMachine(executable, tvm.cuda())
    return vm["main"]


def build_and_run(
    dim: int,
    arch: str,
    dtype: str,
    accum_dtype: str,
    bench_backend: str,
    executor: str,
    dump_vm_code: bool,
) -> Tuple[float, float]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to run this script.")

    mod = _schedule_relax_module(_build_relax_gemv_mlp(dim, dtype, accum_dtype), arch)
    calls = _extract_call_sequence(mod["main"])
    _assert_scheduled_kernels(mod, calls)

    print("=== Scheduled Relax Module ===")
    print(mod.script())
    for _, gv_name, _ in calls:
        print(f"=== Scheduled TIR ({gv_name}) ===")
        print(mod[gv_name].script())

    torch_dtype = getattr(torch, dtype)
    torch.manual_seed(0)
    x = torch.randn((dim,), device="cuda", dtype=torch_dtype)
    w1 = torch.randn((dim, dim), device="cuda", dtype=torch_dtype)
    w2 = torch.randn((dim, dim), device="cuda", dtype=torch_dtype)
    inputs = {"x": x, "w1": w1, "w2": w2}

    ref_fn = _build_reference(dtype, accum_dtype)
    used_executor = executor
    runner = None
    if executor != "python":
        try:
            runner = _build_vm_runner(mod, arch, dump_vm_code)
            used_executor = "vm"
        except Exception as err:
            if executor == "vm":
                raise
            used_executor = "python"
            print(f"Relax VM build fallback: {err}")
    if used_executor == "python":
        kernels = _compile_calls(mod, calls)
        runner = lambda a, b, c: _run_tilelang_graph(mod, calls, kernels, {"x": a, "w1": b, "w2": c})

    try:
        tilelang_out = _normalize_output_to_torch(runner(x, w1, w2), accum_dtype)
    except Exception as err:
        if used_executor == "vm" and executor == "vm":
            raise
        if used_executor == "vm":
            used_executor = "python"
            print(f"Relax VM run fallback: {err}")
            kernels = _compile_calls(mod, calls)
            runner = lambda a, b, c: _run_tilelang_graph(mod, calls, kernels, {"x": a, "w1": b, "w2": c})
            tilelang_out = _normalize_output_to_torch(runner(x, w1, w2), accum_dtype)
        else:
            raise

    ref_out = ref_fn(x, w1, w2)

    rtol, atol = _dtype_tolerance(dtype, accum_dtype)
    torch.testing.assert_close(tilelang_out, ref_out, rtol=rtol, atol=atol)
    print(f"\033[92mCorrectness check passed ({used_executor}).\033[0m")

    tilelang_time = do_bench(
        lambda: runner(x, w1, w2),
        backend=bench_backend,
    )
    torch_time = do_bench(lambda: ref_fn(x, w1, w2), backend=bench_backend)
    print(
        f"TileLang time: {tilelang_time:.6f} ms ({used_executor}), "
        f"torch.compile time: {torch_time:.6f} ms"
    )
    return tilelang_time, torch_time


def check_only(dim: int, arch: str, dtype: str, accum_dtype: str, executor: str, dump_vm_code: bool) -> None:
    mod = _schedule_relax_module(_build_relax_gemv_mlp(dim, dtype, accum_dtype), arch)
    calls = _extract_call_sequence(mod["main"])
    _assert_scheduled_kernels(mod, calls)
    print("=== Scheduled Relax Module ===")
    print(mod.script())
    for _, gv_name, _ in calls:
        print(f"=== Scheduled TIR ({gv_name}) ===")
        print(mod[gv_name].script())
    if dump_vm_code and executor != "python":
        _build_vm_runner(mod, arch, dump_vm_code=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Relax end-to-end GEMV MLP test.")
    parser.add_argument("--arch", type=str, default="sm_90a", help='CUDA arch string, e.g. "sm_90a".')
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
        help="Internal matmul accumulation / output dtype in the Relax graph.",
    )
    parser.add_argument("--bench-backend", type=str, default="event", choices=["event", "cuda"])
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Only check Relax scheduling and generated TIR; skip compile/run.",
    )
    parser.add_argument(
        "--executor",
        type=str,
        default="auto",
        choices=["auto", "vm", "python"],
        help="Execution path: Relax VM, Python per-kernel fallback, or auto-select.",
    )
    parser.add_argument(
        "--dump-vm-code",
        action="store_true",
        help="Dump VM text/python plus any available generated host/device source from the VM build.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.check_only:
        check_only(args.dim, args.arch, args.dtype, args.accum_dtype, args.executor, args.dump_vm_code)
        print("\033[92mRelax scheduling check passed.\033[0m")
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
