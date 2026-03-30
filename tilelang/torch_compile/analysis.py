"""IR analysis and lowering helpers for the TileLang torch.compile backend."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any

import torch
from tvm import relax, tir

from tilelang import tvm
from tilelang.schedule.gpu import default_schedule_rules

logger = logging.getLogger(__name__)


class KernelTrace:
    """Trace record for a single kernel in the graph."""

    __slots__ = (
        "name",
        "schedule_rule",
        "input_shapes",
        "output_shape",
        "output_dtype",
        "is_dynamic",
        "is_torch_op",
        "compile_time_ms",
        "_unscheduled_func",
        "_scheduled_func",
        "_unscheduled_tir",
        "_scheduled_tir",
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
        self._unscheduled_func: Any = None
        self._scheduled_func: Any = None
        self._unscheduled_tir: str | None = None
        self._scheduled_tir: str | None = None

    @property
    def unscheduled_tir(self) -> str:
        if self._unscheduled_tir is None:
            self._unscheduled_tir = (
                str(self._unscheduled_func) if self._unscheduled_func is not None else ""
            )
        return self._unscheduled_tir

    @property
    def scheduled_tir(self) -> str:
        if self._scheduled_tir is None:
            self._scheduled_tir = (
                str(self._scheduled_func) if self._scheduled_func is not None else ""
            )
        return self._scheduled_tir

    def set_tir_funcs(self, unscheduled: Any, scheduled: Any) -> None:
        self._unscheduled_func = unscheduled
        self._scheduled_func = scheduled


@dataclass
class GraphCompileTrace:
    """Structured trace of a single torch.compile backend invocation."""

    compilation_path: str = ""
    arch: str = ""
    dynamic: bool = False
    trace_time_ms: float = 0.0
    schedule_time_ms: float = 0.0
    compile_time_ms: float = 0.0
    total_time_ms: float = 0.0
    kernels: list[KernelTrace] = field(default_factory=list)
    relax_functions: dict[str, str] = field(default_factory=dict)
    schedule_matches: dict[str, str] = field(default_factory=dict)
    # Compilation composition counts (None on cache hits).
    n_compiled: int | None = None
    n_extern: int | None = None
    n_fallback_eager: int | None = None

    def summary(self) -> str:
        lines = [
            "Graph Compilation Trace",
            f"  path: {self.compilation_path}, arch: {self.arch}, dynamic: {self.dynamic}",
            (
                f"  timing: trace={self.trace_time_ms:.1f}ms, "
                f"schedule={self.schedule_time_ms:.1f}ms, "
                f"compile={self.compile_time_ms:.1f}ms, total={self.total_time_ms:.1f}ms"
            ),
            f"  kernels ({len(self.kernels)}):",
        ]
        for kernel in self.kernels:
            tag = "[torch_op]" if kernel.is_torch_op else f"[{kernel.schedule_rule}]"
            dyn = " (dynamic)" if kernel.is_dynamic else ""
            lines.append(
                f"    {kernel.name}: {tag}{dyn} -> {kernel.output_dtype}{list(kernel.output_shape)} "
                f"({kernel.compile_time_ms:.1f}ms)"
            )
        if self.n_compiled is not None:
            lines.append(
                f"  composition: compiled={self.n_compiled}, extern={self.n_extern}, "
                f"fallback_eager={self.n_fallback_eager}"
            )
        return "\n".join(lines)

    def show_tir(self, kernel_name: str | None = None) -> None:
        targets = self.kernels
        if kernel_name is not None:
            targets = [kernel for kernel in self.kernels if kernel.name == kernel_name]
            if not targets:
                print(f"No kernel named '{kernel_name}' in trace.")
                return
        for kernel in targets:
            if kernel.is_torch_op:
                print(f"--- {kernel.name} [torch_op, no TIR] ---")
                continue
            print("=" * 60)
            print(f"Kernel: {kernel.name}  (rule: {kernel.schedule_rule})")
            print("=" * 60)
            print("--- Unscheduled TIR ---")
            print(kernel.unscheduled_tir or "(not captured)")
            print()
            print("--- Scheduled TIR ---")
            print(kernel.scheduled_tir or "(not captured)")
            print()


def _resolve_arch(arch: str | None) -> str:
    if arch is not None:
        return arch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to auto-detect CUDA arch.")
    major, minor = torch.cuda.get_device_capability()
    return f"sm_{major}{minor}"


@dataclass
class _ScheduleResult:
    schedule_matches: dict[str, str] = field(default_factory=dict)
    tir_snapshots: dict[str, tuple[Any, Any]] = field(default_factory=dict)


def _apply_schedule_rules_traced(
    mod: tvm.IRModule,
    rules: list[Any],
    target: tvm.target.Target,
) -> tuple[tvm.IRModule, _ScheduleResult]:
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
            logger.info("  %-40s -> %s", name, matched_rule)
        else:
            logger.debug("  %-40s -> no rule matched", name)
    for g_var, func in updated_functions.items():
        mod[g_var] = func
    return mod, result


def _schedule_relax_module(
    mod: tvm.IRModule,
    arch: str,
    trace: GraphCompileTrace | None = None,
    pass_configs: dict[str, Any] | None = None,
) -> tuple[tvm.IRModule, _ScheduleResult | None]:
    target = tvm.target.cuda(arch=arch)
    schedule_result: _ScheduleResult | None = None
    pass_configs = dict(pass_configs or {})
    with tvm.transform.PassContext(opt_level=3, config=pass_configs), target:
        mod = relax.transform.FuseTransposeMatmul()(mod)
        mod = relax.transform.LegalizeOps()(mod)
        mod = relax.transform.AnnotateTIROpPattern()(mod)
        mod = relax.transform.FoldConstant()(mod)
        mod = relax.transform.FuseOps()(mod)
        mod = relax.transform.FuseTIR()(mod)
        mod = relax.transform.DeadCodeElimination()(mod)
        mod = relax.transform.CanonicalizeBindings()(mod)

        if trace is not None:
            for g_var, func in mod.functions_items():
                if not isinstance(func, tir.PrimFunc):
                    continue
                params = []
                for param in func.params:
                    buf = func.buffer_map.get(param)
                    if buf is None:
                        continue
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


@dataclass
class _ExternOpInfo:
    """Metadata for an unsupported op emitted as an extern call."""

    qualname: str
    target: Any = None
    arg_spec: list = field(default_factory=list)
    literal_kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass
class _TIRCallRecord:
    """A single ``call_tir`` invocation extracted from Relax IR."""

    func_name: str
    arg_names: list[str]
    out_name: str
    out_shape: tuple
    out_dtype: str
    is_torch_fallback: bool = False
    tir_var_exprs: list = field(default_factory=list)
    extern_op: _ExternOpInfo | None = None
    arg_dtypes: list[str] = field(default_factory=list)


@dataclass
class FXLoweringResult:
    """Result of FX -> Relax lowering with extern-op discovery."""

    mod: tvm.IRModule
    extern_ops: dict[str, _ExternOpInfo] = field(default_factory=dict)


def _extract_call_sequence(
    mod: tvm.IRModule,
) -> tuple[list[str], list[_TIRCallRecord], list[str], dict[str, tuple[int, int]], dict[str, Any]] | None:
    main_func = mod["main"]

    var_names: dict = {}
    name_counts: dict[str, int] = {}

    def _unique_name(var: relax.Var) -> str:
        for known_var, name in var_names.items():
            if var.same_as(known_var):
                return name
        hint = var.name_hint
        count = name_counts.get(hint, 0)
        name_counts[hint] = count + 1
        name = hint if count == 0 else f"{hint}__{count}"
        var_names[var] = name
        return name

    param_names = [_unique_name(param) for param in main_func.params]

    sym_var_map: dict[str, tuple[int, int]] = {}
    for param_index, param in enumerate(main_func.params):
        struct_info = param.struct_info
        if not hasattr(struct_info, "shape") or struct_info.shape is None:
            continue
        for dim_index, dim in enumerate(struct_info.shape.values):
            if isinstance(dim, tir.Var) and dim.name not in sym_var_map:
                sym_var_map[dim.name] = (param_index, dim_index)

    records: list[_TIRCallRecord] = []
    constants: dict[str, Any] = {}
    const_count = 0
    tuple_outputs: dict[str, list[str]] = {}
    body = main_func.body
    if not isinstance(body, relax.SeqExpr):
        return None

    for block in body.blocks:
        for binding in block.bindings:
            value = binding.value
            if isinstance(value, relax.Tuple):
                names = []
                for field in value.fields:
                    if not isinstance(field, relax.Var):
                        logger.debug("Direct path: non-Var tuple field: %s", type(field).__name__)
                        return None
                    names.append(_unique_name(field))
                tuple_outputs[_unique_name(binding.var)] = names
                continue

            if not isinstance(value, relax.Call):
                logger.debug("Direct path: non-Call binding %s: %s", binding.var.name_hint, type(value).__name__)
                return None

            op = value.op
            if not (isinstance(op, tvm.ir.Op) and "call_tir" in str(op.name)):
                op_name = op.name if isinstance(op, tvm.ir.Op) else str(op)
                logger.debug("Direct path: unsupported op %s in binding %s", op_name, binding.var.name_hint)
                return None

            func_name = value.args[0].name_hint
            arg_tuple = value.args[1]
            arg_names: list[str] = []
            if not isinstance(arg_tuple, relax.Tuple):
                logger.debug("Direct path: call_tir arg[1] is not Tuple: %s", type(arg_tuple).__name__)
                return None

            for field in arg_tuple.fields:
                if isinstance(field, relax.Var):
                    arg_names.append(_unique_name(field))
                elif isinstance(field, relax.Constant):
                    const_name = f"_const_{const_count}"
                    const_count += 1
                    arg_names.append(const_name)
                    constants[const_name] = torch.from_numpy(field.data.numpy()).cuda()
                else:
                    logger.debug("Direct path: non-Var call_tir arg: %s", type(field).__name__)
                    return None

            struct_info = binding.var.struct_info
            if not hasattr(struct_info, "shape") or struct_info.shape is None:
                logger.debug("Direct path: missing shape info for %s", binding.var.name_hint)
                return None

            out_shape_elems: list[Any] = []
            for dim in struct_info.shape.values:
                if isinstance(dim, tir.IntImm):
                    out_shape_elems.append(int(dim))
                else:
                    try:
                        out_shape_elems.append(int(dim))
                    except (TypeError, ValueError):
                        out_shape_elems.append(dim)
            out_shape = tuple(out_shape_elems)
            out_dtype = str(struct_info.dtype)

            tir_var_exprs: list = []
            if len(value.args) > 2:
                shape_expr = value.args[2]
                if hasattr(shape_expr, "values"):
                    tir_var_exprs = list(shape_expr.values)

            records.append(
                _TIRCallRecord(
                    func_name=func_name,
                    arg_names=arg_names,
                    out_name=_unique_name(binding.var),
                    out_shape=out_shape,
                    out_dtype=out_dtype,
                    tir_var_exprs=tir_var_exprs,
                )
            )

    ret = body.body
    output_names: list[str] = []
    if isinstance(ret, relax.Var):
        name = _unique_name(ret)
        output_names = tuple_outputs.get(name, [name])
    elif isinstance(ret, relax.Tuple):
        for field in ret.fields:
            if not isinstance(field, relax.Var):
                return None
            output_names.append(_unique_name(field))
    else:
        return None

    if not records:
        logger.debug("Direct path: no call_tir records found (identity graph?)")
        return None

    return param_names, records, output_names, sym_var_map, constants


def _compile_tir_direct(
    tir_funcs: dict[str, tir.PrimFunc],
    target: tvm.target.Target,
    save_so_path: str | None = None,
    pass_configs: dict[str, Any] | None = None,
) -> tvm.runtime.Module:
    """Compile TIR PrimFuncs through TileLang's standard lowering pipeline."""

    from tvm.contrib import utils as tvm_utils
    from tilelang.engine.lower import lower

    # Normalize all functions inside a single IRModule so that shared
    # variables (e.g. from keep_params_as_input=True) are preserved.
    raw_mod = tvm.IRModule(
        {name: func.with_attr("global_symbol", name) for name, func in tir_funcs.items()}
    )
    # Filter out functions that use tvm_thread_allreduce (not normalizable).
    needs_normalize = not any(
        "tvm_thread_allreduce" in func.script() for func in tir_funcs.values()
    )
    if needs_normalize:
        from tilelang.engine.phase import NormalizeScheduledIR
        raw_mod = NormalizeScheduledIR(raw_mod)

    mod = raw_mod
    pass_configs = dict(pass_configs or {})
    with tvm.transform.PassContext(opt_level=3, config=pass_configs), target:
        artifact = lower(
            mod,
            target=target,
            runtime_only=True,
            enable_host_codegen=True,
            enable_device_compile=True,
        )

    temp_dir = tvm_utils.tempdir()
    so_path = temp_dir.relpath("tilelang_direct.so")
    artifact.rt_mod.export_library(so_path)

    if save_so_path is not None:
        os.makedirs(os.path.dirname(save_so_path), exist_ok=True)
        import shutil

        shutil.copy2(so_path, save_so_path)

    return tvm.runtime.load_module(so_path)


def compile_subgraph_direct(
    mod: tvm.IRModule,
    target: tvm.target.Target,
    *,
    extern_ops: dict[str, _ExternOpInfo] | None = None,
    save_so_path: str | None = None,
    pass_configs: dict[str, Any] | None = None,
) -> tuple[list[str], list[_TIRCallRecord], list[str], dict[str, tuple[int, int]], Any, dict[str, Any]] | None:
    """Compile a scheduled Relax module for direct kernel execution."""

    result = _extract_call_sequence(mod)
    if result is None:
        return None
    param_names, call_seq, output_names, sym_var_map, constants = result

    extern_ops = extern_ops or {}
    tir_funcs: dict[str, tir.PrimFunc] = {}
    extern_stubs: set[str] = set()
    for g_var, func in mod.functions_items():
        if not isinstance(func, tir.PrimFunc):
            continue
        if func.attrs and func.attrs.get("tir.is_extern_op"):
            extern_stubs.add(g_var.name_hint)
        else:
            tir_funcs[g_var.name_hint] = func

    for record in call_seq:
        if record.func_name in extern_stubs:
            record.is_torch_fallback = True
            record.extern_op = extern_ops.get(record.func_name)

    for record in call_seq:
        if record.is_torch_fallback or record.func_name not in tir_funcs:
            continue
        func = tir_funcs[record.func_name]
        n_inputs = len(record.arg_names)
        for param in func.params[:n_inputs]:
            buf = func.buffer_map.get(param)
            record.arg_dtypes.append(str(buf.dtype) if buf is not None else "")

    # CompileCapability: reject TIR functions with non-float buffers.
    compilable = compile_capability_check(tir_funcs)
    non_compilable = set(tir_funcs) - set(compilable)
    extern_names = {r.func_name for r in call_seq if r.extern_op is not None}
    for record in call_seq:
        if record.func_name in non_compilable:
            if record.func_name not in extern_names:
                # No extern backing → eager subgraph fallback.
                logger.info(
                    "CompileCapability: %s excluded with no extern fallback — eager subgraph",
                    record.func_name,
                )
                return None
            record.is_torch_fallback = True
            logger.info("CompileCapability: %s excluded (non-float buffers)", record.func_name)

    if not compilable:
        if not extern_stubs and not non_compilable:
            logger.debug("compile_subgraph_direct: no TIR PrimFuncs in module")
            return None
        return param_names, call_seq, output_names, sym_var_map, None, constants

    # Bulk TIR compilation.
    try:
        rt_mod = _compile_tir_direct(
            compilable, target, save_so_path=save_so_path, pass_configs=pass_configs,
        )
        return param_names, call_seq, output_names, sym_var_map, rt_mod, constants
    except Exception:
        logger.debug("Bulk TIR compilation failed, trying per-function", exc_info=True)

    # Per-function probe: collect compilable, mark failed.
    good: dict[str, tir.PrimFunc] = {}
    failed: set[str] = set()
    for name, func in compilable.items():
        try:
            _compile_tir_direct({name: func}, target, pass_configs=pass_configs)
            good[name] = func
        except Exception:
            logger.debug("TIR function %s failed compilation", name, exc_info=True)
            failed.add(name)

    # Failure contract: if any failed function has no extern backing,
    # the entire subgraph must fall back to eager (return None).
    extern_names = {r.func_name for r in call_seq if r.extern_op is not None}
    for name in failed:
        if name not in extern_names:
            logger.info(
                "Function %s failed with no extern fallback — eager subgraph fallback", name,
            )
            return None

    if not good:
        logger.debug("compile_subgraph_direct: all TIR functions failed")
        return None

    for record in call_seq:
        if record.func_name in failed:
            record.is_torch_fallback = True
            logger.info("Function %s will use torch fallback", record.func_name)

    rt_mod = _compile_tir_direct(
        good, target, save_so_path=save_so_path, pass_configs=pass_configs,
    )
    return param_names, call_seq, output_names, sym_var_map, rt_mod, constants


_known_ops_cache: set[str] | None = None


def _get_known_fx_ops() -> set[str]:
    global _known_ops_cache
    if _known_ops_cache is None:
        try:
            from tvm.relax.frontend.torch.fx_translator import TorchFXImporter

            importer = TorchFXImporter()
            _known_ops_cache = frozenset(
                key for key in importer.convert_map if isinstance(key, str)
            )
        except Exception:
            _known_ops_cache = frozenset()
    return _known_ops_cache


def _create_extern_stub(
    name: str,
    tensor_arg_infos: list[tuple[list[int], str]],
    out_shape: list[int],
    out_dtype: str,
) -> tir.PrimFunc:
    params = []
    buffer_map = {}
    for index, (shape, dtype) in enumerate(tensor_arg_infos):
        handle = tir.Var(f"p_{name}_{index}", "handle")
        buf = tir.decl_buffer(shape, dtype, f"{name}_in{index}")
        params.append(handle)
        buffer_map[handle] = buf

    out_handle = tir.Var(f"p_{name}_out", "handle")
    out_buf = tir.decl_buffer(out_shape, out_dtype, f"{name}_out")
    params.append(out_handle)
    buffer_map[out_handle] = out_buf

    body = tir.Evaluate(tir.const(0, "int32"))
    func = tir.PrimFunc(params, body, buffer_map=buffer_map)
    func = func.with_attr("tir.noalias", True)
    func = func.with_attr("tir.is_scheduled", True)
    func = func.with_attr("tir.is_extern_op", True)
    func = func.with_attr("op_pattern", 8)  # kOpaque: prevents fusion
    func = func.with_attr("global_symbol", name)
    return func


_COMPILABLE_DTYPES = frozenset({"float16", "float32", "float64", "bfloat16"})


def compile_capability_check(
    tir_funcs: dict[str, tir.PrimFunc],
) -> dict[str, tir.PrimFunc]:
    """Return only TIR functions whose buffers are all float types.

    TileLang's lowering pipeline cannot handle int64, bool, or other
    non-float buffer dtypes.  Functions with such buffers are excluded.
    """
    compilable: dict[str, tir.PrimFunc] = {}
    for name, func in tir_funcs.items():
        ok = True
        for param in func.params:
            buf = func.buffer_map.get(param)
            if buf is not None and str(buf.dtype) not in _COMPILABLE_DTYPES:
                ok = False
                break
        if ok:
            compilable[name] = func
    return compilable


def _op_identity_key(target: Any) -> str:
    """Return a stable identity key for an FX call_function target.

    - ``OpOverload`` / ``OpOverloadPacket``: ``str(target)`` (e.g. ``"aten.sdpa.default"``)
    - Other callables: ``module.qualname`` (e.g. ``"torch.nn.functional.scaled_dot_product_attention"``)
    - Fallback: ``target.__name__``
    """
    if isinstance(target, (torch._ops.OpOverloadPacket, torch._ops.OpOverload)):
        return str(target)
    mod = getattr(target, "__module__", "") or ""
    qual = getattr(target, "__qualname__", "") or getattr(target, "__name__", "")
    return f"{mod}.{qual}" if mod else qual


# Permanent extern entries keyed by identity key and __name__ fallback.
_PERMANENT_EXTERN_KEYS: frozenset[str] = frozenset({
    "torch._C._nn.scaled_dot_product_attention",
    "torch.nn.functional.scaled_dot_product_attention",
})
_PERMANENT_EXTERN_NAMES: frozenset[str] = frozenset({
    "scaled_dot_product_attention",
    "tensor",
})


@dataclass
class ExternPolicy:
    """Classifies FX ops as extern (bypass TIR, call torch directly).

    An op is extern if:
    1. Its identity key matches permanent extern entries, OR
    2. It is a custom ``torch.library`` op (non-aten namespace), OR
    3. It is absent from TVM's ``from_fx`` convert_map.
    """

    # Aten namespaces that TVM's from_fx handles natively.
    _KNOWN_NAMESPACES: frozenset[str] = frozenset({"aten", "prims", "prim"})

    def is_extern(self, target: Any, known_fx_ops: set[str]) -> bool:
        """Return True if *target* should be an extern call."""
        # Check permanent extern by identity key first.
        key = _op_identity_key(target)
        if key in _PERMANENT_EXTERN_KEYS:
            return True
        # For OpOverload/OpOverloadPacket: extern only if custom (non-aten).
        if isinstance(target, (torch._ops.OpOverloadPacket, torch._ops.OpOverload)):
            ns = getattr(target, "namespace", "")
            return ns not in self._KNOWN_NAMESPACES
        # Check __name__ for permanent entries and TVM convert_map.
        name = getattr(target, "__name__", None)
        if name is None:
            return True
        if name in _PERMANENT_EXTERN_NAMES:
            return True
        if name not in known_fx_ops:
            return True
        return False


def _build_unsupported_op_map(
    gm: torch.fx.GraphModule,
    extern_ops: dict[str, _ExternOpInfo],
    *,
    force_extern_ops: set[str] | None = None,
) -> dict[str, Any]:
    force_extern_ops = set(force_extern_ops or set())
    known_fx = _get_known_fx_ops()
    policy = ExternPolicy()

    # Dedup by identity key. We also track __name__ → identity key
    # because TVM's custom_convert_map is keyed by __name__.
    unsupported: dict[str, Any] = {}     # identity_key → target
    unsup_by_name: dict[str, str] = {}   # __name__ → identity_key
    nontensor_ops: set[str] = set()
    for node in gm.graph.nodes:
        if node.op != "call_function":
            continue
        name = getattr(node.target, "__name__", "")
        ikey = _op_identity_key(node.target)
        if ikey in unsupported or name in nontensor_ops:
            continue
        example_value = node.meta.get("example_value")
        if example_value is None or not hasattr(example_value, "shape"):
            nontensor_ops.add(name)
        elif name in force_extern_ops or policy.is_extern(node.target, known_fx):
            unsupported[ikey] = node.target
            unsup_by_name[name] = ikey

    convert_map: dict[str, Any] = {}
    counter: dict[str, int] = {}

    for ikey, op_target in unsupported.items():
        # TVM's custom_convert_map is keyed by __name__.
        op_name = getattr(op_target, "__name__", ikey)
        qualname = ikey

        def _make_converter(_op_name, _qualname, _target):
            def _converter(node, importer):
                from torch import fx as torch_fx

                bb = importer.block_builder

                def _to_int(dim):
                    try:
                        return int(dim)
                    except (TypeError, ValueError):
                        return 1

                tensor_args: list[relax.Expr] = []
                tensor_arg_infos: list[tuple[list[int], str]] = []
                arg_spec: list = []
                tensor_index = 0

                def _add_tensor(fx_node):
                    nonlocal tensor_index
                    rx_var = importer.env[fx_node]
                    tensor_args.append(rx_var)
                    struct_info = rx_var.struct_info
                    shape = [_to_int(dim) for dim in struct_info.shape.values]
                    dtype = str(struct_info.dtype)
                    tensor_arg_infos.append((shape, dtype))
                    arg_spec.append(("tensor", tensor_index))
                    tensor_index += 1

                for arg in node.args:
                    if isinstance(arg, torch_fx.Node):
                        _add_tensor(arg)
                    elif isinstance(arg, (list, tuple)) and arg and isinstance(arg[0], torch_fx.Node):
                        list_start = tensor_index
                        for element in arg:
                            if isinstance(element, torch_fx.Node):
                                _add_tensor(element)
                            else:
                                arg_spec.append(("literal", element))
                        added = tensor_index - list_start
                        del arg_spec[-added:]
                        arg_spec.append(("tensor_list", list(range(list_start, tensor_index))))
                    else:
                        arg_spec.append(("literal", arg))

                literal_kwargs = {
                    key: value
                    for key, value in node.kwargs.items()
                    if not isinstance(value, torch_fx.Node)
                }

                example = node.meta["example_value"]
                out_dtype = str(example.dtype).replace("torch.", "")
                rx_shape = []
                for dim in example.shape:
                    try:
                        rx_shape.append(int(dim))
                    except (TypeError, ValueError):
                        rx_shape.append(dim)
                out_struct_info = relax.TensorStructInfo(rx_shape, out_dtype)
                stub_out_shape = [_to_int(dim) for dim in example.shape]

                count = counter.get(_op_name, 0)
                counter[_op_name] = count + 1
                stub_name = f"unsup_{_op_name}" if count == 0 else f"unsup_{_op_name}_{count}"

                stub = _create_extern_stub(
                    stub_name,
                    tensor_arg_infos,
                    stub_out_shape,
                    out_dtype,
                )
                extern_ops[stub_name] = _ExternOpInfo(
                    qualname=_qualname,
                    target=_target,
                    arg_spec=arg_spec,
                    literal_kwargs=literal_kwargs,
                )
                gvar = bb.add_func(stub, stub_name)
                call = relax.call_tir(gvar, relax.Tuple(tensor_args), out_struct_info)
                return bb.emit(call)

            return _converter

        if op_name in convert_map:
            logger.warning(
                "Unsupported op name collision: %s (identity keys: %s and %s). "
                "Later converter wins.",
                op_name, convert_map.get(f"_ikey_{op_name}", "?"), ikey,
            )
        convert_map[op_name] = _make_converter(op_name, qualname, op_target)
        convert_map[f"_ikey_{op_name}"] = ikey  # track for collision detection

    # Remove tracking keys before returning.
    convert_map = {k: v for k, v in convert_map.items() if not k.startswith("_ikey_")}

    def _noop(node, importer):
        return None

    for name in nontensor_ops:
        convert_map[name] = _noop

    if unsupported:
        logger.info("Auto-fallback for unsupported ops: %s", ", ".join(sorted(unsupported)))
    return convert_map


def _simplify_fx_graph(gm: torch.fx.GraphModule) -> None:
    """Simplify FX patterns that ``from_fx`` can't handle.

    Removes ``cat([torch.tensor([]), x]) → x`` (KV cache init with
    no prior cache produces an empty tensor concat that triggers a
    dtype mismatch in Relax).
    """
    from torch import fx as _fx

    graph = gm.graph
    changed = False
    for node in list(graph.nodes):
        if node.op != "call_function" or node.target.__name__ != "cat":
            continue
        args = node.args
        if not args or not isinstance(args[0], (list, tuple)):
            continue
        tensors = args[0]
        live = []
        for t in tensors:
            if isinstance(t, _fx.Node):
                ev = t.meta.get("example_value")
                if ev is not None and hasattr(ev, "shape") and 0 in ev.shape:
                    continue
            live.append(t)
        if len(live) == len(tensors):
            continue
        if len(live) == 1 and isinstance(live[0], _fx.Node):
            node.replace_all_uses_with(live[0])
            graph.erase_node(node)
            changed = True
    if changed:
        graph.eliminate_dead_code()
        gm.recompile()


def from_fx_with_fallback(
    gm: torch.fx.GraphModule,
    input_info: list,
    **kwargs,
) -> FXLoweringResult:
    """Run ``from_fx`` with extern-op fallback for unsupported FX ops."""

    from tvm.relax.frontend.torch import from_fx

    _simplify_fx_graph(gm)

    extern_ops: dict[str, _ExternOpInfo] = {}
    convert_map = _build_unsupported_op_map(gm, extern_ops)
    try:
        mod = from_fx(
            gm,
            input_info,
            unwrap_unit_return_tuple=True,
            keep_params_as_input=True,
            custom_convert_map=convert_map or None,
            **kwargs,
        )
        return FXLoweringResult(mod=mod, extern_ops=extern_ops)
    except Exception:
        # from_fx failed even after extern stubs. Let the caller
        # fall back to eager for the whole subgraph.
        raise
