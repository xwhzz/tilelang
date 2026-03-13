from __future__ import annotations

from collections.abc import Sequence
import torch
from torch.nn import Module
from torch.export import export
import torch.nn.functional as F
import tilelang
import tilelang.relax as tl_relax
from tilelang import tvm
from tilelang.schedule.gpu import default_schedule_rules
from tvm import relax, tir
from tvm.ir import GlobalVar
from tvm.relax.frontend.torch import from_exported_program


def _build_relax_gemv_mlp(dim: int, inputs: dict[str, torch.Tensor], dtype: str, accum_dtype: str) -> tvm.IRModule:
    torch_dtype = getattr(torch, dtype)
    torch_accum_dtype = getattr(torch, accum_dtype)

    class MLP(Module):
        def __init__(self, torch_dtype, torch_accum_dtype):
            super().__init__()
            self.torch_dtype = torch_dtype
            self.torch_accum_dtype = torch_accum_dtype

        def forward(self, x: torch.Tensor, w1: torch.Tensor, w2: torch.Tensor) -> torch.Tensor:
            if self.torch_accum_dtype != self.torch_dtype:
                x_acc = x.to(self.torch_dtype)
                w1_acc = w1.to(self.torch_dtype)
                w2_acc = w2.to(self.torch_accum_dtype)
            else:
                x_acc, w1_acc, w2_acc = x, w1, w2
            lv0 = torch.matmul(w1_acc, x_acc)
            lv1 = F.relu(lv0)
            lv2 = torch.matmul(w2_acc, lv1)
            return lv2

    exported_program = export(MLP(torch_dtype, torch_accum_dtype), args=(inputs["x"], inputs["w1"], inputs["w2"]), dynamic_shapes={})
    mod = from_exported_program(
        exported_program,
        run_ep_decomposition=None,
        keep_params_as_input=None,
        unwrap_unit_return_tuple=None,
        no_bind_return_tuple=None,
        custom_convert_map=None,
    )
    return mod


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


def _extract_call_sequence(main_func: relax.Function) -> list[tuple[str, str, list[str]]]:
    if not isinstance(main_func.body, relax.SeqExpr):
        raise RuntimeError("Expected Relax main body to be a SeqExpr.")
    if len(main_func.body.blocks) != 1 or not isinstance(main_func.body.blocks[0], relax.DataflowBlock):
        raise RuntimeError("Expected Relax main body to contain one DataflowBlock.")

    calls: list[tuple[str, str, list[str]]] = []
    for binding in main_func.body.blocks[0].bindings:
        value = binding.value
        if not isinstance(value, relax.Call) or value.op.name != "relax.call_tir":
            continue
        callee = value.args[0]
        if not isinstance(callee, GlobalVar):
            raise RuntimeError("Expected call_tir callee to be a GlobalVar.")
        arg_names: list[str] = []
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


def _output_buffer_info(func: tir.PrimFunc) -> tuple[tuple[int, ...], str]:
    output_buffer = func.buffer_map[list(func.params)[-1]]
    return tuple(int(extent) for extent in output_buffer.shape), output_buffer.dtype


def _assert_scheduled_kernels(mod: tvm.IRModule, calls: Sequence[tuple[str, str, list[str]]]) -> None:
    has_cross_thread_reduction = False
    for _, gv_name, _ in calls:
        text = mod[gv_name].script()
        has_tile_staging = "T.copy(" in text and "local.fragment" in text
        has_thread_reduction = "tvm_thread_allreduce" in text
        has_parallelism = 'launch_thread("threadIdx.x"' in text or "thread_binding" in text or "T.parallel(" in text
        if not (has_tile_staging or has_thread_reduction):
            raise RuntimeError(f"Expected either tile staging or cross-thread reduction in {gv_name}, but found neither.")
        if not has_parallelism:
            raise RuntimeError(f"Expected explicit GPU parallelism in {gv_name}, but found none.")
        has_cross_thread_reduction = has_cross_thread_reduction or has_thread_reduction

    if not has_cross_thread_reduction:
        raise RuntimeError("Expected at least one GEMV-style cross-thread reduction kernel.")


def _build_vm_runner(mod: tvm.IRModule, arch: str):
    target = tvm.target.cuda(arch=arch)
    executable = tl_relax.build(mod, target=target)
    vm = relax.VirtualMachine(executable, tvm.cuda())
    return vm["main"]


def _compile_calls(mod: tvm.IRModule, calls: Sequence[tuple[str, str, list[str]]]) -> dict[str, tilelang.jit.JITKernel]:
    kernels = {}
    for _, gv_name, _ in calls:
        kernels[gv_name] = tilelang.compile(_lower_primfunc_for_tilelang(mod[gv_name], gv_name))
    return kernels


def _run_tilelang_graph(
    mod: tvm.IRModule,
    calls: Sequence[tuple[str, str, list[str]]],
    kernels: dict[str, tilelang.jit.JITKernel],
    inputs: dict[str, torch.Tensor],
) -> torch.Tensor:
    env = dict(inputs)
    for out_name, gv_name, arg_names in calls:
        shape, out_dtype = _output_buffer_info(mod[gv_name])
        out = torch.empty(shape, device="cuda", dtype=getattr(torch, out_dtype))
        kernels[gv_name](*[env[arg_name] for arg_name in arg_names], out)
        env[out_name] = out
    return env[calls[-1][0]]


def build_and_run(
    dim: int,
    arch: str,
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
    inputs = {"x": x, "w1": w1, "w2": w2}

    mod = _schedule_relax_module(_build_relax_gemv_mlp(dim, inputs, dtype, accum_dtype), arch)
    calls = _extract_call_sequence(mod["main"])
    _assert_scheduled_kernels(mod, calls)

    used_executor = executor
    runner = None
    if executor != "python":
        try:
            runner = _build_vm_runner(mod, arch)
            used_executor = "vm"
        except Exception as err:
            if executor == "vm":
                raise
            used_executor = "python"
            print(f"Relax VM build fallback: {err}")
    if used_executor == "python":
        kernels = _compile_calls(mod, calls)

        def runner(a, b, c):
            return _run_tilelang_graph(mod, calls, kernels, {"x": a, "w1": b, "w2": c})

    return runner


# @my_torch_compile
# def mlp(torch_dtype, torch_accum_dtype):

#     def kernel( x: torch.Tensor, w1: torch.Tensor, w2: torch.Tensor) -> torch.Tensor:
#         if torch_accum_dtype != torch_dtype:
#             x_acc = x.to(torch_dtype)
#             w1_acc = w1.to(torch_dtype)
#             w2_acc = w2.to(torch_accum_dtype)
#         else:
#             x_acc, w1_acc, w2_acc = x, w1, w2
#         lv0 = torch.matmul(w1_acc, x_acc)
#         lv1 = F.relu(lv0)
#         lv2 = torch.matmul(w2_acc, lv1)
#         return lv2
