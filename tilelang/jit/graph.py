"""Graph-mode compilation utilities for TileLang.

Shared infrastructure used by ``tilelang.jit.backend`` (the
``torch.compile(backend="tilelang")`` backend) and related tools:

* **Trace classes** — ``KernelTrace``, ``GraphCompileTrace``
* **Schedule helpers** — ``_schedule_relax_module``, ``_apply_schedule_rules_traced``
* **TIR normalization** — ``_lower_primfunc_for_tilelang``
* **VM build** — ``tilelang_relax_build``

Usage (via ``torch.compile``)::

    compiled = torch.compile(model, backend="tilelang")
    result = compiled(x_cuda)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import torch
from tvm import relax, tir

from tilelang import tvm
from tilelang.schedule.gpu import default_schedule_rules

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Graph compilation trace
# ---------------------------------------------------------------------------

class KernelTrace:
    """Trace record for a single kernel in the graph.

    TIR snapshots (``unscheduled_tir`` / ``scheduled_tir``) are serialized
    lazily on first access to avoid the cost of TVM Script printing during
    compilation when the user never inspects the trace.
    """

    __slots__ = (
        "name", "schedule_rule", "input_shapes", "output_shape",
        "output_dtype", "is_dynamic", "is_torch_op", "compile_time_ms",
        "_unscheduled_func", "_scheduled_func",
        "_unscheduled_tir", "_scheduled_tir",
    )

    def __init__(self, name: str):
        self.name = name
        self.schedule_rule: str = ""
        self.input_shapes: list[tuple[int | str, ...]] = []
        self.output_shape: tuple[int | str, ...] = ()
        self.output_dtype: str = ""
        self.is_dynamic: bool = False
        self.is_torch_op: bool = False
        self.compile_time_ms: float = 0.0
        # Lazy TIR snapshots: store PrimFunc refs, serialize on access.
        self._unscheduled_func: Any = None
        self._scheduled_func: Any = None
        self._unscheduled_tir: str | None = None
        self._scheduled_tir: str | None = None

    @property
    def unscheduled_tir(self) -> str:
        if self._unscheduled_tir is None:
            self._unscheduled_tir = str(self._unscheduled_func) if self._unscheduled_func is not None else ""
        return self._unscheduled_tir

    @property
    def scheduled_tir(self) -> str:
        if self._scheduled_tir is None:
            self._scheduled_tir = str(self._scheduled_func) if self._scheduled_func is not None else ""
        return self._scheduled_tir

    def set_tir_funcs(self, unscheduled: Any, scheduled: Any) -> None:
        """Attach PrimFunc references for lazy serialization."""
        self._unscheduled_func = unscheduled
        self._scheduled_func = scheduled


@dataclass
class GraphCompileTrace:
    """Structured trace of the entire graph compilation pipeline."""
    compilation_path: str = ""  # "export", "dynamo", or "export+dynamo_fallback"
    arch: str = ""
    dynamic: bool = False
    # Relax IR pass timings (ms)
    trace_time_ms: float = 0.0     # torch.export tracing
    schedule_time_ms: float = 0.0  # Relax transforms + schedule rules
    compile_time_ms: float = 0.0   # total tilelang.compile time
    total_time_ms: float = 0.0
    # Per-kernel details
    kernels: list[KernelTrace] = field(default_factory=list)
    # Relax IR functions before scheduling (name → op summary)
    relax_functions: dict[str, str] = field(default_factory=dict)
    # Schedule rule matching (function_name → rule_name)
    schedule_matches: dict[str, str] = field(default_factory=dict)

    def summary(self) -> str:
        """Return a human-readable summary of the compilation trace."""
        lines = []
        lines.append(f"Graph Compilation Trace")
        lines.append(f"  path: {self.compilation_path}, arch: {self.arch}, "
                      f"dynamic: {self.dynamic}")
        lines.append(f"  timing: trace={self.trace_time_ms:.1f}ms, "
                      f"schedule={self.schedule_time_ms:.1f}ms, "
                      f"compile={self.compile_time_ms:.1f}ms, "
                      f"total={self.total_time_ms:.1f}ms")
        lines.append(f"  kernels ({len(self.kernels)}):")
        for kt in self.kernels:
            tag = "[torch_op]" if kt.is_torch_op else f"[{kt.schedule_rule}]"
            dyn = " (dynamic)" if kt.is_dynamic else ""
            lines.append(f"    {kt.name}: {tag}{dyn} "
                          f"→ {kt.output_dtype}{list(kt.output_shape)} "
                          f"({kt.compile_time_ms:.1f}ms)")
        return "\n".join(lines)

    def show_tir(self, kernel_name: str | None = None) -> None:
        """Print unscheduled and scheduled TIR for each kernel.

        If *kernel_name* is given, only that kernel is printed.
        """
        targets = self.kernels
        if kernel_name is not None:
            targets = [kt for kt in self.kernels if kt.name == kernel_name]
            if not targets:
                print(f"No kernel named '{kernel_name}' in trace.")
                return
        for kt in targets:
            if kt.is_torch_op:
                print(f"--- {kt.name} [torch_op, no TIR] ---")
                continue
            print(f"{'=' * 60}")
            print(f"Kernel: {kt.name}  (rule: {kt.schedule_rule})")
            print(f"{'=' * 60}")
            print(f"--- Unscheduled TIR ---")
            print(kt.unscheduled_tir or "(not captured)")
            print()
            print(f"--- Scheduled TIR ---")
            print(kt.scheduled_tir or "(not captured)")
            print()


# ---------------------------------------------------------------------------
# Shared utilities (tracing, scheduling, extraction)
# ---------------------------------------------------------------------------

def _resolve_arch(arch: str | None) -> str:
    if arch is not None:
        return arch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to auto-detect CUDA arch.")
    major, minor = torch.cuda.get_device_capability()
    return f"sm_{major}{minor}"



@dataclass
class _ScheduleResult:
    """Result of traced schedule rule application."""
    schedule_matches: dict[str, str] = field(default_factory=dict)
    # Per-function PrimFunc refs: {func_name: (unscheduled_func, scheduled_func)}.
    # Stored as references, not strings — serialization is deferred to KernelTrace.
    tir_snapshots: dict[str, tuple[Any, Any]] = field(default_factory=dict)


def _apply_schedule_rules_traced(
    mod: tvm.IRModule,
    rules: list[Any],
    target: tvm.target.Target,
) -> tuple[tvm.IRModule, _ScheduleResult]:
    """Apply schedule rules with tracing.

    Returns ``(mod, result)`` where *result* contains which rule matched
    each TIR function and per-function before/after TIR snapshots.
    """
    result = _ScheduleResult()
    updated_functions = {}
    for g_var, func in mod.functions_items():
        if not isinstance(func, tir.PrimFunc):
            continue
        if func.attrs and func.attrs.get("tir.is_scheduled"):
            continue
        name = g_var.name_hint
        matched_rule = None
        scheduled_func = None
        for rule in rules:
            space = rule.apply(func, target, False)
            if space is None:
                continue
            if isinstance(space, tir.Schedule):
                space = [space]
            assert len(space) == 1
            scheduled_func = space[0].mod["main"].with_attr("tir.is_scheduled", True)
            updated_functions[g_var] = scheduled_func
            matched_rule = type(rule).__name__
            break
        result.schedule_matches[name] = matched_rule or "none"
        result.tir_snapshots[name] = (func, scheduled_func)
        if matched_rule:
            logger.info("  %-40s → %s", name, matched_rule)
        else:
            logger.debug("  %-40s → no rule matched", name)
    for g_var, func in updated_functions.items():
        mod[g_var] = func
    return mod, result


def _schedule_relax_module(
    mod: tvm.IRModule,
    arch: str,
    trace: GraphCompileTrace | None = None,
) -> tuple[tvm.IRModule, _ScheduleResult | None]:
    """Schedule all TIR functions in *mod*.

    Returns ``(mod, schedule_result)`` where *schedule_result* is ``None``
    when tracing is disabled (no ``trace`` argument).
    """
    target = tvm.target.cuda(arch=arch)
    schedule_result: _ScheduleResult | None = None
    with target:
        mod = relax.transform.LegalizeOps()(mod)
        mod = relax.transform.AnnotateTIROpPattern()(mod)
        mod = relax.transform.FoldConstant()(mod)
        mod = relax.transform.FuseOps()(mod)
        mod = relax.transform.FuseTIR()(mod)

        if trace is not None:
            # Capture pre-schedule function list.
            for g_var, func in mod.functions_items():
                if isinstance(func, tir.PrimFunc):
                    params = []
                    for p in func.params:
                        buf = func.buffer_map.get(p)
                        if buf is not None:
                            shape_str = "x".join(str(s) for s in buf.shape)
                            params.append(f"{buf.dtype}[{shape_str}]")
                    trace.relax_functions[g_var.name_hint] = ", ".join(params)

            logger.info("Schedule rule matching:")
            rules = default_schedule_rules()
            mod, schedule_result = _apply_schedule_rules_traced(mod, rules, target)
            trace.schedule_matches = schedule_result.schedule_matches
        else:
            mod = tvm.dlight.ApplyDefaultSchedule(*default_schedule_rules())(mod)
    return mod, schedule_result


def _lower_primfunc_for_tilelang(func: tir.PrimFunc, name: str) -> tir.PrimFunc:
    func = func.with_attr("global_symbol", name)
    if "tvm_thread_allreduce" in func.script():
        return func
    from tilelang.engine.phase import NormalizeScheduledIR
    mod = tvm.IRModule({name: func})
    mod = NormalizeScheduledIR(mod)
    return mod[name]


# ---------------------------------------------------------------------------
# TileLang TIR compilation with host function support
# ---------------------------------------------------------------------------

def _compile_vmtir_and_kernels(
    device_tir_mod: tvm.IRModule,
    target: tvm.target.Target,
    host_funcs: dict,
) -> tuple:
    """Compile device TIR and VM dispatch as three separate modules.

    Returns ``(vmtir_rt, kernel_host_rt, device_rt)``:

    * ``vmtir_rt`` — C source module with ``__vmtir__*`` dispatch only.
    * ``kernel_host_rt`` — C source module with kernel host wrappers
      (launch stubs) AND shape helper functions.
    * ``device_rt`` — CUDA module with device kernels (or *None*).

    The caller builds the import chain::

        vmtir_dso  -->  kernel_dso  -->  cuda_mod

    ``TVMBackendGetFuncFromEnv`` only searches module *imports* (not
    the module itself).  So everything that ``__vmtir__*`` calls
    (kernel wrappers, shape funcs) must be in ``kernel_dso``, and
    everything that kernel wrappers call (device kernels) must be in
    ``cuda_mod``.
    """
    from tilelang.engine.lower import (
        canon_target_host,
        device_codegen,
        get_device_call,
        get_host_call,
        host_codegen,
        is_cpu_device_backend,
    )
    from tilelang.engine.phase import (
        LowerAndLegalize,
        OptimizeForTarget,
        PreLowerSemanticCheck,
    )

    target_host = canon_target_host(target, None)
    target_host = tvm.target.Target.canon_target(target_host)
    target = tvm.target.Target(target, target_host)

    _is_host_call = get_host_call(is_device_c=is_cpu_device_backend(target))
    _is_device_call = get_device_call(is_device_c=is_cpu_device_backend(target))

    # Standard TileLang 3-phase lowering (device functions only).
    has_device = len(device_tir_mod.get_global_vars()) > 0
    if has_device:
        PreLowerSemanticCheck(device_tir_mod)
        device_tir_mod = LowerAndLegalize(device_tir_mod, target)
        device_tir_mod = OptimizeForTarget(device_tir_mod, target)

    kernel_host_mod = tir.transform.Filter(_is_host_call)(device_tir_mod) if has_device else tvm.IRModule()
    device_mod = tir.transform.Filter(_is_device_call)(device_tir_mod) if has_device else tvm.IRModule()

    # Split host_funcs: __vmtir__* go into vmtir_mod (top-level DSO);
    # everything else (shape helpers, etc.) goes into kernel_host_mod
    # so they're findable via imports from __vmtir__*.
    vmtir_funcs = {}
    callable_funcs = {}
    for name, func in host_funcs.items():
        if name.startswith("__vmtir__"):
            vmtir_funcs[name] = func
        else:
            callable_funcs[name] = func

    target_host_with_host = tvm.target.Target(target_host, host=target_host)

    # Prepare __vmtir__* for C codegen.
    vmtir_mod = tvm.IRModule()
    if vmtir_funcs:
        vmtir_mod = tvm.IRModule(vmtir_funcs)
        vmtir_mod = tir.transform.BindTarget(target_host_with_host)(vmtir_mod)
        vmtir_mod = tir.transform.MakePackedAPI()(vmtir_mod)

    # Prepare shape helpers for C codegen, merge into kernel_host_mod.
    if callable_funcs:
        shape_mod = tvm.IRModule(callable_funcs)
        shape_mod = tir.transform.BindTarget(target_host_with_host)(shape_mod)
        shape_mod = tir.transform.MakePackedAPI()(shape_mod)
        for g_var, func in shape_mod.functions_items():
            kernel_host_mod[g_var.name_hint] = func

    # Codegen: produce THREE separate runtime modules.
    vmtir_rt = host_codegen(vmtir_mod, target_host)
    kernel_host_rt = host_codegen(kernel_host_mod, target_host)
    device_rt = device_codegen(device_mod, target) if has_device else None
    return vmtir_rt, kernel_host_rt, device_rt


# ---------------------------------------------------------------------------
# Relax VM build with TileLang TIR compilation
# ---------------------------------------------------------------------------

def _post_schedule_relax_pipeline() -> Any:
    """Relax lowering passes needed between scheduling and VM codegen.

    ``_schedule_relax_module`` already applies LegalizeOps → FuseTIR.
    This pipeline applies the remaining passes from TVM's default build
    pipeline that prepare the Relax IR for VM bytecode generation.
    """
    return tvm.transform.Sequential([
        relax.transform.RewriteDataflowReshape(),
        relax.transform.ToNonDataflow(),
        relax.transform.RemovePurityChecking(),
        relax.transform.CallTIRRewrite(),
        relax.transform.StaticPlanBlockMemory(),
        relax.transform.LowerAllocTensor(),
        relax.transform.KillAfterLastUse(),
        relax.transform.LowerRuntimeBuiltin(),
        relax.transform.ComputePrimValue(),
        relax.transform.VMShapeLower(),
        relax.transform.AttachGlobalSymbol(),
    ])


def tilelang_relax_build(
    mod: tvm.IRModule,
    target: tvm.target.Target,
) -> tvm.runtime.Module:
    """Build a scheduled Relax module into a VM executable.

    Uses **compiled** exec_mode so that VM dispatch is compiled to
    native C code (``__vmtir__*`` functions) instead of interpreted
    bytecode, reducing per-call overhead by ~30 us.

    PrimFuncs are stripped before VM codegen so ``VMTIRCodeGen`` uses
    ``EmitCallPacked`` (dynamic lookup via ``TVMBackendGetFuncFromEnv``).
    Since ``TVMBackendGetFuncFromEnv`` only searches module *imports*
    (not the module itself), we build a two-DSO import chain::

        vmtir_dso  -->  kernel_dso  -->  cuda_mod

    * ``vmtir_dso``: ``__vmtir__*`` dispatch functions
    * ``kernel_dso``: kernel host wrappers (launch stubs)
    * ``cuda_mod``: device kernels

    The result can be loaded into ``relax.VirtualMachine(ex, device)``.
    """
    from tvm.relax import _ffi_api as relax_ffi, vm_build

    # 1. Apply post-schedule Relax lowering.
    with target:
        mod = _post_schedule_relax_pipeline()(mod)

    # 2. Extract external_mods / constants from module attrs.
    attrs = dict(mod.attrs) if mod.attrs else {}
    ext_libs: list[tvm.runtime.Module] = list(attrs.get("external_mods", []))
    params: dict[str, Any] = dict(attrs.get("const_name_to_constant", {}))

    # 3. Strip TIR PrimFuncs from the module before VM codegen so that
    #    VMTIRCodeGen uses EmitCallPacked (dynamic name-based lookup
    #    via TVMBackendGetFuncFromEnv) instead of EmitCallCPacked.
    device_funcs: dict = {}  # {name: PrimFunc}
    host_funcs: dict = {}    # {name: PrimFunc}
    stripped_mod = tvm.IRModule({}, attrs=mod.attrs)
    for g_var, func in mod.functions_items():
        if isinstance(func, tir.PrimFunc):
            name = g_var.name_hint
            if func.attrs and func.attrs.get("tir.is_host_func"):
                host_funcs[name] = func
            else:
                device_funcs[name] = func
        else:
            stripped_mod[g_var] = func

    # 4. VM codegen (compiled) on the stripped module.
    builder = relax.ExecBuilder()
    leftover = vm_build._vmcodegen(builder, stripped_mod, "compiled")
    tir_mod = vm_build._filter_tir(leftover)

    # 5. Collect __vmtir__* functions into host_funcs.
    if tir_mod is not None:
        for g_var, func in tir_mod.functions_items():
            if isinstance(func, tir.PrimFunc):
                host_funcs[g_var.name_hint] = func

    # 6. Normalize device PrimFuncs for TileLang.
    for name in list(device_funcs):
        device_funcs[name] = _lower_primfunc_for_tilelang(device_funcs[name], name)

    # 7. Compile device PrimFuncs and __vmtir__* as SEPARATE DSOs.
    #
    #    TVMBackendGetFuncFromEnv (used by EmitCallPacked) only
    #    searches module *imports*, NOT the module itself.  So the
    #    __vmtir__* DSO must import the kernel DSO to find kernel
    #    host wrappers.
    #
    #    Module tree:
    #      vmtir_dso (DSO A: __vmtir__* dispatch)
    #        └── kernel_dso (DSO B: host wrappers for kernel launch)
    #              └── cuda_mod (CUDA: device kernels)
    with target:
        vmtir_rt, kernel_host_rt, device_rt = _compile_vmtir_and_kernels(
            tvm.IRModule(device_funcs) if device_funcs else tvm.IRModule(),
            target,
            host_funcs,
        )

    # 8. Build DSO module tree and link VM executable.
    from tvm.contrib import utils as _tvm_utils
    _tmp = _tvm_utils.tempdir()

    # DSO B: kernel host wrappers + CUDA device kernels as import
    _kernel_dso_path = _tmp.relpath("tilelang_kernels.so")
    kernel_host_rt.export_library(_kernel_dso_path)
    kernel_dso = tvm.runtime.load_module(_kernel_dso_path)
    if device_rt is not None:
        kernel_dso.import_module(device_rt)

    # DSO A: __vmtir__* dispatch, imports kernel_dso
    _vmtir_dso_path = _tmp.relpath("tilelang_vmtir.so")
    vmtir_rt.export_library(_vmtir_dso_path)
    vmtir_dso = tvm.runtime.load_module(_vmtir_dso_path)
    vmtir_dso.import_module(kernel_dso)

    # 9. Link VM executable.
    return vm_build.VMExecutable(
        relax_ffi.VMLink(builder, target, vmtir_dso, ext_libs, params),
    )


# ---------------------------------------------------------------------------
# Direct kernel execution (bypasses VM for lower per-call overhead)
# ---------------------------------------------------------------------------

@dataclass
class _TIRCallRecord:
    """A single ``call_tir`` invocation extracted from Relax IR."""
    func_name: str
    arg_names: list
    out_name: str
    out_shape: tuple
    out_dtype: str


def _extract_call_sequence(
    mod: tvm.IRModule,
) -> tuple[list[str], list[_TIRCallRecord], list[str]] | None:
    """Extract the ``call_tir`` execution sequence from a scheduled Relax module.

    Returns ``(param_names, call_records, output_names)`` or ``None``
    if the IR contains operations the direct runner cannot handle
    (e.g. dynamic shapes, non-call_tir ops).

    Relax vars with identical ``name_hint`` are disambiguated via a
    Var-identity→unique-name mapping (e.g. ``lv``, ``lv__1``).
    """
    main_func = mod["main"]

    # Build Var → unique name mapping (handles name_hint collisions).
    # TVM ObjectRef types are hashable by underlying C++ pointer.
    _var_names: dict = {}  # Var → unique name
    _name_counts: dict[str, int] = {}  # name_hint → count

    def _unique_name(var: relax.Var) -> str:
        for known_var, name in _var_names.items():
            if var.same_as(known_var):
                return name
        hint = var.name_hint
        cnt = _name_counts.get(hint, 0)
        _name_counts[hint] = cnt + 1
        name = hint if cnt == 0 else f"{hint}__{cnt}"
        _var_names[var] = name
        return name

    param_names = [_unique_name(p) for p in main_func.params]

    records: list[_TIRCallRecord] = []
    # Track Tuple bindings: output groupings (not kernel calls).
    tuple_outputs: dict[str, list[str]] = {}
    body = main_func.body
    if not isinstance(body, relax.SeqExpr):
        return None

    for block in body.blocks:
        for binding in block.bindings:
            val = binding.value
            # Handle output Tuple grouping (e.g. gv = (lv, inp_0)).
            if isinstance(val, relax.Tuple):
                names = []
                for f in val.fields:
                    if isinstance(f, relax.Var):
                        names.append(_unique_name(f))
                    else:
                        return None
                tuple_outputs[_unique_name(binding.var)] = names
                continue
            if not isinstance(val, relax.Call):
                logger.debug("Direct path: non-Call binding %s: %s",
                             binding.var.name_hint, type(val).__name__)
                return None
            op = val.op
            if not (isinstance(op, tvm.ir.Op) and "call_tir" in str(op.name)):
                op_name = op.name if isinstance(op, tvm.ir.Op) else str(op)
                logger.debug("Direct path: unsupported op %s in binding %s",
                             op_name, binding.var.name_hint)
                return None

            func_name = val.args[0].name_hint
            arg_tuple = val.args[1]
            arg_names: list[str] = []
            if isinstance(arg_tuple, relax.Tuple):
                for f in arg_tuple.fields:
                    if isinstance(f, relax.Var):
                        arg_names.append(_unique_name(f))
                    else:
                        return None
            else:
                return None

            sinfo = binding.var.struct_info
            if not hasattr(sinfo, "shape") or sinfo.shape is None:
                return None
            try:
                out_shape = tuple(int(s) for s in sinfo.shape.values)
            except (TypeError, ValueError):
                return None  # dynamic shape
            out_dtype = str(sinfo.dtype)

            records.append(_TIRCallRecord(
                func_name=func_name,
                arg_names=arg_names,
                out_name=_unique_name(binding.var),
                out_shape=out_shape,
                out_dtype=out_dtype,
            ))

    ret = body.body
    output_names: list[str] = []
    if isinstance(ret, relax.Var):
        name = _unique_name(ret)
        if name in tuple_outputs:
            output_names = tuple_outputs[name]
        else:
            output_names = [name]
    elif isinstance(ret, relax.Tuple):
        for f in ret.fields:
            if isinstance(f, relax.Var):
                output_names.append(_unique_name(f))
            else:
                return None
    else:
        return None

    if not records:
        logger.debug("Direct path: no call_tir records found (identity graph?)")
        return None

    return param_names, records, output_names


def _compile_tir_direct(
    tir_funcs: dict[str, tir.PrimFunc],
    target: tvm.target.Target,
) -> tvm.runtime.Module:
    """Compile TIR PrimFuncs through TileLang's pipeline.

    Returns a loaded runtime module where kernels are callable by name.
    """
    from tilelang.engine.lower import (
        canon_target_host,
        device_codegen,
        get_device_call,
        get_host_call,
        host_codegen,
        is_cpu_device_backend,
    )
    from tilelang.engine.phase import (
        LowerAndLegalize,
        OptimizeForTarget,
        PreLowerSemanticCheck,
    )

    for name in list(tir_funcs):
        tir_funcs[name] = _lower_primfunc_for_tilelang(tir_funcs[name], name)

    mod = tvm.IRModule(tir_funcs)

    target_host = canon_target_host(target, None)
    target_host = tvm.target.Target.canon_target(target_host)
    full_target = tvm.target.Target(target, target_host)

    with full_target:
        PreLowerSemanticCheck(mod)
        mod = LowerAndLegalize(mod, full_target)
        mod = OptimizeForTarget(mod, full_target)

    is_dev_c = is_cpu_device_backend(full_target)
    host_mod = tir.transform.Filter(get_host_call(is_dev_c))(mod)
    device_mod = tir.transform.Filter(get_device_call(is_dev_c))(mod)

    host_rt = host_codegen(host_mod, target_host)
    has_device = len(device_mod.get_global_vars()) > 0
    device_rt = device_codegen(device_mod, full_target) if has_device else None

    from tvm.contrib import utils as _tvm_utils
    _tmp = _tvm_utils.tempdir()
    _so_path = _tmp.relpath("tilelang_direct.so")
    host_rt.export_library(_so_path)
    rt = tvm.runtime.load_module(_so_path)
    if device_rt is not None:
        rt.import_module(device_rt)
    return rt


def compile_subgraph_direct(
    mod: tvm.IRModule,
    target: tvm.target.Target,
) -> tuple[list[str], list[_TIRCallRecord], list[str], tvm.runtime.Module] | None:
    """Compile a scheduled Relax module for direct kernel execution.

    Returns ``(param_names, call_seq, output_names, rt_mod)`` or ``None``
    if the module requires the VM path (dynamic shapes, unsupported ops).
    """
    result = _extract_call_sequence(mod)
    if result is None:
        return None
    param_names, call_seq, output_names = result

    tir_funcs: dict[str, tir.PrimFunc] = {}
    for gvar, func in mod.functions_items():
        if isinstance(func, tir.PrimFunc):
            tir_funcs[gvar.name_hint] = func
    if not tir_funcs:
        return None

    rt_mod = _compile_tir_direct(tir_funcs, target)
    return param_names, call_seq, output_names, rt_mod


def _build_extra_convert_map() -> dict:
    """Build a convert map for ops missing from the default TVM from_fx map.

    These are needed by common models (e.g. transformers Llama) but not yet
    upstream in TVM's Relax PyTorch frontend.
    """
    extra: dict = {}

    # F.embedding (functional form, distinct from nn.Embedding module)
    def _embedding(node, importer):
        # node.args: (input, weight, padding_idx, max_norm, norm_type,
        #             scale_grad_by_freq, sparse)
        x = importer.env[node.args[0]]
        weight = importer.env[node.args[1]]
        return importer._embedding_impl(x, weight)

    extra["embedding"] = _embedding

    # torch.Tensor.diff (used by transformers for position_ids)
    def _diff(node, importer):
        x = importer.env[node.args[0]]
        n = node.args[1] if len(node.args) > 1 else node.kwargs.get("n", 1)
        dim = node.args[2] if len(node.args) > 2 else node.kwargs.get("dim", -1)
        if n != 1:
            raise NotImplementedError("diff with n != 1")
        bb = importer.block_builder
        ndim = x.struct_info.ndim
        if dim < 0:
            dim = ndim + dim
        # diff(x, dim=d) = x[..., 1:, ...] - x[..., :-1, ...]
        shape_vals = x.struct_info.shape.values
        dim_len = shape_vals[dim]
        axes = [dim]
        x_later = bb.emit(relax.op.strided_slice(x, axes=axes, begin=[1], end=[int(dim_len)]))
        x_earlier = bb.emit(relax.op.strided_slice(x, axes=axes, begin=[0], end=[int(dim_len) - 1]))
        return bb.emit(relax.op.subtract(x_later, x_earlier))

    extra["diff"] = _diff

    return extra

