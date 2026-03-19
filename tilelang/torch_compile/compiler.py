"""Torch frontend compiler with Relax VM first and TileLang JIT fallback."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Literal

import torch
import tvm_ffi
from torch.export import export
from torch.nn import Module
from torch.utils import dlpack as torch_dlpack

import tilelang
import tilelang.relax as tl_relax
from tilelang import tvm
from tilelang.schedule.gpu import default_schedule_rules
from tvm import relax, tir
from tvm.ir import GlobalVar
from tvm.relax.frontend.torch import from_exported_program

ExecutorMode = Literal["auto", "vm", "jit"]


@dataclass(frozen=True)
class CallRecord:
    out_name: str
    gv_name: str
    arg_names: tuple[str, ...]


def _resolve_arch(arch: str | None) -> str:
    if arch is not None:
        return arch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to auto-detect CUDA arch.")
    major, minor = torch.cuda.get_device_capability()
    return f"sm_{major}{minor}"


def _dump_runtime_module_sources(rt_mod: tvm.runtime.Module, prefix: str = "root", seen: set[int] | None = None) -> None:
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


def _build_relax_module(func: Callable[..., torch.Tensor], example_args: tuple[torch.Tensor, ...]) -> tvm.IRModule:
    class _WrappedModule(Module):
        def __init__(self, inner: Callable[..., torch.Tensor]):
            super().__init__()
            self.inner = inner

        def forward(self, *args: torch.Tensor) -> torch.Tensor:
            return self.inner(*args)

    wrapped = _WrappedModule(func)
    exported_program = export(
        wrapped,
        args=example_args,
        dynamic_shapes={},
    )
    return from_exported_program(
        exported_program,
        run_ep_decomposition=None,
        keep_params_as_input=None,
        unwrap_unit_return_tuple=None,
        no_bind_return_tuple=None,
        custom_convert_map=None,
    )


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


def _extract_call_sequence(main_func: relax.Function) -> tuple[CallRecord, ...]:
    if not isinstance(main_func.body, relax.SeqExpr):
        raise RuntimeError("Expected Relax main body to be a SeqExpr.")
    if len(main_func.body.blocks) != 1 or not isinstance(main_func.body.blocks[0], relax.DataflowBlock):
        raise RuntimeError("Expected Relax main body to contain one DataflowBlock.")

    calls = []
    for binding in main_func.body.blocks[0].bindings:
        value = binding.value
        if not isinstance(value, relax.Call) or value.op.name != "relax.call_tir":
            continue
        callee = value.args[0]
        if not isinstance(callee, GlobalVar):
            raise RuntimeError("Expected call_tir callee to be a GlobalVar.")
        arg_names = []
        for arg in value.args[1]:
            if not isinstance(arg, relax.Var):
                raise RuntimeError(f"Unsupported call_tir arg type: {type(arg)}")
            arg_names.append(arg.name_hint)
        calls.append(CallRecord(binding.var.name_hint, callee.name_hint, tuple(arg_names)))

    if not calls:
        raise RuntimeError("No call_tir kernels were found in the scheduled Relax graph.")
    return tuple(calls)


def _extract_input_names(main_func: relax.Function) -> tuple[str, ...]:
    names = []
    for param in main_func.params:
        if not isinstance(param, relax.Var):
            raise RuntimeError(f"Unsupported Relax parameter type: {type(param)}")
        names.append(param.name_hint)
    return tuple(names)


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


def _normalize_vm_output_to_torch(value: Any, device: torch.device) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value
    if hasattr(value, "__dlpack__"):
        try:
            return torch_dlpack.from_dlpack(value)
        except Exception:
            pass
    if hasattr(value, "to_dlpack"):
        try:
            return torch_dlpack.from_dlpack(value.to_dlpack())
        except Exception:
            pass
    if hasattr(value, "numpy"):
        return torch.from_numpy(value.numpy()).to(device=device)
    if isinstance(value, tvm_ffi.container.Array):
        return _normalize_vm_output_to_torch(value[0], device)
    raise TypeError(f"Unsupported VM output type: {type(value)}")


def _build_vm_runner(mod: tvm.IRModule, arch: str, dump_vm_code: bool):
    target = tvm.target.cuda(arch=arch)
    executable = tl_relax.build(mod, target=target)
    if dump_vm_code:
        _dump_vm_executable(executable)
    vm = relax.VirtualMachine(executable, tvm.cuda())
    return vm["main"]


def _build_jit_kernels(mod: tvm.IRModule, calls: tuple[CallRecord, ...]) -> dict[str, tilelang.jit.JITKernel]:
    kernels = {}
    for call in calls:
        kernels[call.gv_name] = tilelang.compile(_lower_primfunc_for_tilelang(mod[call.gv_name], call.gv_name))
    return kernels


class _JITGraphRunner:
    def __init__(
        self,
        mod: tvm.IRModule,
        calls: tuple[CallRecord, ...],
        input_names: tuple[str, ...],
        device: torch.device,
    ) -> None:
        self.mod = mod
        self.calls = calls
        self.input_names = input_names
        self.device = device
        self.kernels = _build_jit_kernels(mod, calls)

    def __call__(self, *args: torch.Tensor) -> torch.Tensor:
        if len(args) != len(self.input_names):
            raise ValueError(f"Expected {len(self.input_names)} inputs, got {len(args)}.")
        env = dict(zip(self.input_names, args))
        for call in self.calls:
            shape, out_dtype = _output_buffer_info(self.mod[call.gv_name])
            out = torch.empty(shape, device=self.device, dtype=getattr(torch, out_dtype))
            self.kernels[call.gv_name](*[env[name] for name in call.arg_names], out)
            env[call.out_name] = out
        return env[self.calls[-1].out_name]


def _validate_tensor_args(args: tuple[Any, ...]) -> tuple[torch.Tensor, ...]:
    if not args:
        raise ValueError("Expected at least one tensor input.")
    tensors = []
    for idx, arg in enumerate(args):
        if not isinstance(arg, torch.Tensor):
            raise TypeError(f"Only tensor inputs are supported now. Arg#{idx} has type {type(arg)}.")
        if arg.device.type != "cuda":
            raise RuntimeError("Only CUDA tensors are supported now.")
        if not arg.is_contiguous():
            raise ValueError(
                f"Only contiguous CUDA tensors are supported now. "
                f"Arg#{idx} has shape {tuple(arg.shape)} and stride {tuple(arg.stride())}."
            )
        tensors.append(arg)
    return tuple(tensors)


def _signature_from_inputs(args: tuple[torch.Tensor, ...]) -> tuple[Any, ...]:
    return tuple(
        (
            tuple(int(v) for v in arg.shape),
            str(arg.dtype),
            arg.device.type,
            arg.device.index,
        )
        for arg in args
    )


class CompiledTorchRunner:
    """Compiled callable with VM-first execution and JIT fallback."""

    def __init__(
        self,
        mod: tvm.IRModule,
        calls: tuple[CallRecord, ...],
        input_names: tuple[str, ...],
        arch: str,
        requested_executor: ExecutorMode,
        dump_vm_code: bool,
        device: torch.device,
    ) -> None:
        self.scheduled_mod = mod
        self.call_sequence = calls
        self.input_names = input_names
        self.arch = arch
        self.requested_executor = requested_executor
        self.dump_vm_code = dump_vm_code
        self.device = device

        self._vm_runner = None
        self._jit_runner: _JITGraphRunner | None = None
        self.active_executor: ExecutorMode = requested_executor

        self._build_initial_executor()

    def _build_initial_executor(self) -> None:
        if self.requested_executor in ("auto", "vm"):
            try:
                self._vm_runner = _build_vm_runner(self.scheduled_mod, self.arch, self.dump_vm_code)
                self.active_executor = "vm"
                return
            except Exception:
                if self.requested_executor == "vm":
                    raise
        self._jit_runner = _JITGraphRunner(
            mod=self.scheduled_mod,
            calls=self.call_sequence,
            input_names=self.input_names,
            device=self.device,
        )
        self.active_executor = "jit"

    def _ensure_jit_runner(self) -> _JITGraphRunner:
        if self._jit_runner is None:
            self._jit_runner = _JITGraphRunner(
                mod=self.scheduled_mod,
                calls=self.call_sequence,
                input_names=self.input_names,
                device=self.device,
            )
        return self._jit_runner

    def _run_vm_raw_with_fallback(self, *args: torch.Tensor):
        assert self._vm_runner is not None
        try:
            return self._vm_runner(*args)
        except Exception:
            if self.requested_executor == "vm":
                raise
            jit_runner = self._ensure_jit_runner()
            self.active_executor = "jit"
            return jit_runner(*args)

    def __call__(self, *args: torch.Tensor) -> torch.Tensor:
        tensors = _validate_tensor_args(args)
        if self.active_executor == "vm":
            out = self._run_vm_raw_with_fallback(*tensors)
            if self.active_executor == "vm":
                return _normalize_vm_output_to_torch(out, self.device)
            return out

        jit_runner = self._ensure_jit_runner()
        return jit_runner(*tensors)


class TorchCompileImpl:
    """Decorator wrapper for compiling PyTorch callables through Relax+TileLang."""

    def __init__(
        self,
        func: Callable[..., torch.Tensor],
        *,
        arch: str | None = None,
        executor: ExecutorMode = "auto",
        dump_vm_code: bool = False,
    ) -> None:
        self.func = func
        self.arch = arch
        self.executor = executor
        self.dump_vm_code = dump_vm_code
        self._cache: dict[tuple[Any, ...], CompiledTorchRunner] = {}

    def compile(self, *example_args: torch.Tensor) -> CompiledTorchRunner:
        tensors = _validate_tensor_args(example_args)
        signature = _signature_from_inputs(tensors)
        cached = self._cache.get(signature)
        if cached is not None:
            return cached

        arch = _resolve_arch(self.arch)
        mod = _build_relax_module(self.func, tensors)
        mod = _schedule_relax_module(mod, arch)
        main_func = mod["main"]
        calls = _extract_call_sequence(main_func)
        input_names = _extract_input_names(main_func)

        compiled = CompiledTorchRunner(
            mod=mod,
            calls=calls,
            input_names=input_names,
            arch=arch,
            requested_executor=self.executor,
            dump_vm_code=self.dump_vm_code,
            device=tensors[0].device,
        )
        self._cache[signature] = compiled
        return compiled

    def __call__(self, *args: torch.Tensor) -> torch.Tensor:
        compiled = self.compile(*args)
        return compiled(*args)


def torch_compile(
    func: Callable[..., torch.Tensor] | None = None,
    *,
    arch: str | None = None,
    executor: ExecutorMode = "auto",
    dump_vm_code: bool = False,
) -> Callable[[Callable[..., torch.Tensor]], TorchCompileImpl] | TorchCompileImpl:
    """Decorate a PyTorch function and compile with VM-first, JIT-fallback execution."""

    if executor not in ("auto", "vm", "jit"):
        raise ValueError(f"Unsupported executor: {executor}")

    def decorator(inner: Callable[..., torch.Tensor]) -> TorchCompileImpl:
        return TorchCompileImpl(
            inner,
            arch=arch,
            executor=executor,
            dump_vm_code=dump_vm_code,
        )

    return decorator(func) if func is not None else decorator
