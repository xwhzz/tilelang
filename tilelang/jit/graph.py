"""Graph-mode JIT: end-to-end compilation of PyTorch functions via TileLang.

Traces a PyTorch function → Relax IR → schedules each kernel with TileLang
schedule rules → compiles each kernel via ``tilelang.compile`` → returns a
``GraphRunner`` with pre-allocated intermediate buffers.

Usage::

    @tilelang.jit(mode="graph")
    def mlp(x, w1, w2):
        h = torch.matmul(w1, x)
        h = torch.relu(h)
        return torch.matmul(w2, h)

    result = mlp(x_cuda, w1_cuda, w2_cuda)
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable

import torch
import tvm_ffi
from torch.export import export
from torch.nn import Module
from tvm import relax, tir
from tvm.ir import GlobalVar
from tvm.relax.frontend.torch import from_exported_program, from_fx
from tvm.tir import PrimFunc

import tilelang
from tilelang import tvm
from tilelang.schedule.gpu import default_schedule_rules

logger = logging.getLogger(__name__)

# TVM op pattern constant: prevents FuseOps from fusing this function.
_OP_PATTERN_OPAQUE = 8


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
    """Structured trace of the entire graph compilation pipeline.

    Accessible via ``GraphRunner.trace`` or ``GraphJITImpl.get_trace()``.
    """
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


def _torch_dtype_to_str(dtype: torch.dtype) -> str:
    """Convert a ``torch.dtype`` to a TVM dtype string."""
    from tilelang.language.dtypes import _TORCH_DTYPE_TO_STR
    return _TORCH_DTYPE_TO_STR.get(dtype, str(dtype).split(".")[-1])


def _scan_custom_ops(exported_program) -> dict[str, Any]:
    """Detect user-registered custom ops in an exported FX graph.

    Returns a dict mapping op function names (e.g. ``"my_op.default"``)
    to the original torch op callable, for ops that are NOT in the
    standard Relax convert map.
    """
    from tvm.relax.frontend.torch.exported_program_translator import ExportedProgramImporter

    # Build the standard convert map to know what's already handled.
    dummy_importer = ExportedProgramImporter()
    standard_keys = set(dummy_importer.create_convert_map().keys())

    custom_ops: dict[str, Any] = {}
    for node in exported_program.graph.nodes:
        if node.op != "call_function":
            continue
        func_name = node.target.__name__
        if func_name not in standard_keys:
            custom_ops[func_name] = node.target
    return custom_ops


def _build_custom_op_convert_map(
    custom_ops: dict[str, Any],
    exported_program,
) -> tuple[dict[str, Callable], dict[str, Any]]:
    """Build convert_map entries for user-registered custom ops.

    Returns ``(convert_map, torch_op_map)`` where:
    - *convert_map* maps op names to Relax converters
    - *torch_op_map* maps Relax GV names to original torch callables
    """
    convert_map: dict[str, Callable] = {}
    torch_op_map: dict[str, Any] = {}

    for func_name, op_callable in custom_ops.items():
        safe_name = func_name.replace(".", "_")
        relax_name = f"torch_op_{safe_name}"

        # Extract output shape/dtype from the FX node metadata.
        out_shape: tuple[int, ...] | None = None
        out_dtype: str | None = None
        for node in exported_program.graph.nodes:
            if node.op == "call_function" and node.target.__name__ == func_name:
                val = node.meta.get("val")
                if val is not None and isinstance(val, torch.Tensor):
                    out_shape = tuple(int(s) for s in val.shape)
                    out_dtype = _torch_dtype_to_str(val.dtype)
                break

        if out_shape is None or out_dtype is None:
            continue  # Skip ops we can't infer output info for.

        def _make(rn, os, od):
            def converter(node, importer):
                args = [importer.env[a] for a in node.args if a in importer.env]
                bb = importer.block_builder

                # Build a stub PrimFunc with the correct buffer signature.
                params = []
                buf_map = {}
                for i, arg in enumerate(args):
                    sinfo = arg.struct_info
                    shape = [int(s) for s in sinfo.shape]
                    buf = tir.decl_buffer(shape, sinfo.dtype, f"arg{i}")
                    param = tir.Var(f"p{i}", "handle")
                    params.append(param)
                    buf_map[param] = buf
                out_buf = tir.decl_buffer(os, od, "output")
                out_param = tir.Var("p_out", "handle")
                params.append(out_param)
                buf_map[out_param] = out_buf

                body = tir.Evaluate(tir.const(0))
                stub = tir.PrimFunc(params, body, buffer_map=buf_map)
                stub = stub.with_attr("global_symbol", rn)
                stub = stub.with_attr("op_pattern", _OP_PATTERN_OPAQUE)
                stub = stub.with_attr("tir.is_scheduled", True)
                stub = stub.with_attr("torch_op", True)

                gv = bb.add_func(stub, rn)
                sinfo = relax.TensorStructInfo(os, od)
                return bb.emit(relax.call_tir(gv, args, sinfo))
            return converter

        convert_map[func_name] = _make(relax_name, out_shape, out_dtype)
        torch_op_map[relax_name] = op_callable

    return convert_map, torch_op_map


def _build_dynamic_shapes_spec(
    example_args: tuple[torch.Tensor, ...],
    dynamic_dims: dict[int, list[int]] | None,
) -> dict[str, Any] | None:
    """Convert ``dynamic_dims`` to ``torch.export.Dim`` specifications.

    Parameters
    ----------
    example_args : tuple of Tensor
        Example inputs whose shapes seed the min/max hints.
    dynamic_dims : dict mapping arg index → list of dynamic dimension indices,
        or ``None`` for fully static export.

    Returns
    -------
    dynamic_shapes : dict | None
        A dict suitable for ``torch.export.export(dynamic_shapes=...)``,
        or ``None`` to fall back to all-static export.
    """
    if not dynamic_dims:
        return None

    from torch.export import Dim

    # Use Dim.AUTO for each marked dimension — torch.export automatically
    # determines equality constraints from the traced computation.
    per_arg: list[dict[int, Any] | None] = [None] * len(example_args)
    for arg_idx, dim_indices in dynamic_dims.items():
        if arg_idx < 0 or arg_idx >= len(example_args):
            raise ValueError(
                f"dynamic_dims references arg#{arg_idx} but only "
                f"{len(example_args)} inputs were given."
            )
        per_arg[arg_idx] = {d: Dim.AUTO for d in dim_indices}

    per_arg_final = tuple(spec if spec is not None else {} for spec in per_arg)
    return {"args": per_arg_final}


def _build_relax_module(
    func: Callable[..., torch.Tensor],
    example_args: tuple[torch.Tensor, ...],
    dynamic_dims: dict[int, list[int]] | None = None,
) -> tuple[tvm.IRModule, dict[str, Any]]:
    """Trace and convert a PyTorch function to Relax IR.

    Returns ``(mod, torch_op_map)`` where *torch_op_map* maps Relax GV
    names to original torch callables for user-registered custom ops.

    If the function contains data-dependent control flow (``if tensor:``,
    ``while tensor:``), ``torch.export`` will raise an error.  The caller
    should catch this and fall back to the dynamo-based compilation path.
    """
    class _WrappedModule(Module):
        def __init__(self, inner: Callable[..., torch.Tensor]):
            super().__init__()
            self.inner = inner

        def forward(self, *args: torch.Tensor) -> torch.Tensor:
            return self.inner(*args)

    wrapped = _WrappedModule(func)
    ds_spec = _build_dynamic_shapes_spec(example_args, dynamic_dims)
    exported_program = export(
        wrapped, args=example_args,
        dynamic_shapes=ds_spec if ds_spec is not None else {},
    )

    # Auto-detect user-registered custom ops and generate converters.
    detected_custom_ops = _scan_custom_ops(exported_program)
    torch_op_map: dict[str, Any] = {}
    custom_convert_map: dict[str, Any] | None = None
    if detected_custom_ops:
        custom_convert_map, torch_op_map = _build_custom_op_convert_map(
            detected_custom_ops, exported_program,
        )

    mod = from_exported_program(
        exported_program,
        run_ep_decomposition=None,
        keep_params_as_input=None,
        unwrap_unit_return_tuple=None,
        no_bind_return_tuple=None,
        custom_convert_map=custom_convert_map,
    )
    return mod, torch_op_map


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

    return tuple(calls)


def _extract_input_names(main_func: relax.Function) -> tuple[str, ...]:
    names = []
    for param in main_func.params:
        if not isinstance(param, relax.Var):
            raise RuntimeError(f"Unsupported Relax parameter type: {type(param)}")
        names.append(param.name_hint)
    return tuple(names)


def _compile_kernels(
    mod: tvm.IRModule,
    calls: tuple[CallRecord, ...],
    pass_configs: dict[str, Any] | None = None,
    compile_flags: list[str] | str | None = None,
    out_idx: list[int] | None = None,
    torch_op_map: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Compile all kernels in *calls* from the scheduled Relax module.

    Shared logic between ``_compile_graph`` (export path) and
    ``_compile_subgraph`` (dynamo path).
    """
    compile_kwargs: dict[str, Any] = {}
    if pass_configs:
        compile_kwargs["pass_configs"] = pass_configs
    if compile_flags:
        compile_kwargs["compile_flags"] = compile_flags
    if out_idx is not None:
        compile_kwargs["out_idx"] = out_idx

    kernels: dict[str, Any] = {}
    for call in calls:
        tir_func = mod[call.gv_name]
        is_torch_op = (
            isinstance(tir_func, tir.PrimFunc)
            and tir_func.attrs
            and tir_func.attrs.get("torch_op")
        )
        if is_torch_op:
            if torch_op_map is None:
                raise RuntimeError(
                    f"torch_op attribute set on '{call.gv_name}' but no "
                    f"torch_op_map was provided."
                )
            torch_callable = torch_op_map.get(call.gv_name)
            if torch_callable is None:
                raise RuntimeError(
                    f"torch_op attribute set on '{call.gv_name}' but no "
                    f"matching entry in torch_op_map."
                )
            kernels[call.gv_name] = _TorchOpWrapper(
                torch_callable, dynamic=(out_idx is not None),
            )
        else:
            prepared = _lower_primfunc_for_tilelang(tir_func, call.gv_name)
            kernels[call.gv_name] = tilelang.compile(prepared, **compile_kwargs)
    return kernels


def _lower_primfunc_for_tilelang(func: tir.PrimFunc, name: str) -> tir.PrimFunc:
    func = func.with_attr("global_symbol", name)
    if "tvm_thread_allreduce" in func.script():
        return func
    from tilelang.engine.phase import NormalizeScheduledIR
    mod = tvm.IRModule({name: func})
    mod = NormalizeScheduledIR(mod)
    return mod[name]


def _output_buffer_info(
    func: tir.PrimFunc,
) -> tuple[tuple[int | tir.Var, ...], str, bool]:
    """Extract output buffer shape, dtype, and whether shape is dynamic.

    Returns ``(shape, dtype, is_dynamic)`` where *shape* may contain
    ``tir.Var`` entries for symbolic dimensions.
    """
    output_buffer = func.buffer_map[list(func.params)[-1]]
    shape: list[int | tir.Var] = []
    is_dynamic = False
    for extent in output_buffer.shape:
        if isinstance(extent, tir.IntImm):
            shape.append(int(extent.value))
        else:
            shape.append(extent)
            is_dynamic = True
    return tuple(shape), str(output_buffer.dtype), is_dynamic


def _validate_tensor_args(args: tuple[Any, ...]) -> tuple[torch.Tensor, ...]:
    """Validate that all arguments are contiguous CUDA tensors.

    Non-tensor arguments (bool, int, float, etc.) are silently skipped
    so that mixed-type calls work with the dynamo fallback path.
    """
    if not args:
        raise ValueError("Expected at least one input.")
    tensors = []
    for idx, arg in enumerate(args):
        if not isinstance(arg, torch.Tensor):
            continue  # Non-tensor args are handled by the dynamo path.
        if arg.device.type != "cuda":
            raise RuntimeError(f"Only CUDA tensors are supported now. Arg#{idx} is on {arg.device}.")
        if not arg.is_contiguous():
            raise ValueError(
                f"Only contiguous CUDA tensors are supported now. "
                f"Arg#{idx} has shape {tuple(arg.shape)} and stride {tuple(arg.stride())}."
            )
        tensors.append(arg)
    if not tensors:
        raise ValueError("Expected at least one tensor input.")
    return tuple(tensors)


def _signature_from_inputs(args: tuple[Any, ...]) -> tuple[Any, ...]:
    parts = []
    for arg in args:
        if isinstance(arg, torch.Tensor):
            parts.append(
                (tuple(int(v) for v in arg.shape), str(arg.dtype), arg.device.type, arg.device.index)
            )
        else:
            # Non-tensor args (bool, int, float) contribute their value
            # so that different values trigger recompilation.
            parts.append(("scalar", type(arg).__name__, arg))
    return tuple(parts)


# ---------------------------------------------------------------------------
# Custom kernel injection
# ---------------------------------------------------------------------------

def _inject_custom_kernels(
    mod: tvm.IRModule,
    custom_kernels: dict[str, PrimFunc],
) -> tvm.IRModule:
    """Replace scheduled TIR functions with user-provided PrimFuncs by name.

    For each *(name, primfunc)* pair in *custom_kernels*, find the GlobalVar
    in *mod* whose ``name_hint`` contains *name* as a substring (to tolerate
    ``FuseTIR`` renaming like ``fused_matmul`` → ``fused_matmul_relu``) and
    swap the scheduled function with the user-provided one.
    """
    for pattern, custom_func in custom_kernels.items():
        matched_gv = None
        for gv in mod.get_global_vars():
            if pattern in gv.name_hint:
                matched_gv = gv
                break
        if matched_gv is None:
            available = [gv.name_hint for gv in mod.get_global_vars()]
            raise ValueError(
                f"Custom kernel pattern '{pattern}' did not match any function "
                f"in the scheduled module. Available: {available}"
            )
        # Carry over the global_symbol attribute so downstream passes find it.
        custom_func = custom_func.with_attr("global_symbol", matched_gv.name_hint)
        mod.update_func(matched_gv, custom_func)  # mutates in-place
    return mod


# ---------------------------------------------------------------------------
# Torch op wrapper for graph-mode dispatch
# ---------------------------------------------------------------------------

class _TorchOpWrapper:
    """Adapts a ``torch`` op with ``output = op(*inputs)`` convention to
    GraphRunner's ``op(*inputs, output_buffer)`` calling convention.

    In static mode (``dynamic=False``), the wrapper copies the result into
    a pre-allocated output buffer (last argument).  In dynamic mode
    (``dynamic=True``), the wrapper simply returns the result — no output
    buffer is passed.
    """

    def __init__(self, torch_callable: Any, dynamic: bool = False) -> None:
        self.torch_callable = torch_callable
        self.dynamic = dynamic

    def __call__(self, *args: torch.Tensor) -> torch.Tensor | None:
        if self.dynamic:
            return self.torch_callable(*args)
        inputs = args[:-1]
        out_buf = args[-1]
        result = self.torch_callable(*inputs)
        out_buf.copy_(result)


# ---------------------------------------------------------------------------
# GraphRunner – pre-allocated hot-path executor
# ---------------------------------------------------------------------------

class GraphRunner:
    """Execute a compiled kernel graph with pre-allocated intermediate buffers.

    All intermediate tensors are allocated once in ``__init__``; the call path
    only performs dictionary lookups and kernel dispatches—no ``torch.empty``
    per invocation.

    Optionally, the kernel sequence can be captured as a ``torch.cuda.CUDAGraph``
    via :meth:`enable_cuda_graph` for near-zero CPU dispatch overhead (~1 μs).
    """

    def __init__(
        self,
        mod: tvm.IRModule,
        kernels: dict[str, Any],
        calls: tuple[CallRecord, ...],
        input_names: tuple[str, ...],
        device: torch.device,
        dynamic: bool = False,
    ) -> None:
        self.kernels = kernels
        self.calls = calls
        self.input_names = input_names
        self.device = device
        self.dynamic = dynamic

        self._input_name_set = frozenset(input_names)
        input_set = self._input_name_set

        if dynamic:
            # Dynamic shapes: kernels handle their own output allocation
            # via out_idx=[-1] — no pre-allocation.
            self._call_outputs = [None] * len(calls)
        else:
            # Static shapes: pre-allocate one output buffer per call.
            self._call_outputs: list[torch.Tensor | None] = []
            for call in calls:
                if call.out_name in input_set:
                    self._call_outputs.append(None)
                else:
                    shape, dtype, _ = _output_buffer_info(mod[call.gv_name])
                    self._call_outputs.append(
                        torch.empty(shape, device=device, dtype=getattr(torch, dtype)),
                    )

        self.scheduled_mod = mod

        self._cuda_graph: torch.cuda.CUDAGraph | None = None
        self._cuda_graph_pending: bool = False
        self._warmup_iters: int = 3
        self._graph_inputs: tuple[torch.Tensor, ...] | None = None
        self._graph_output: torch.Tensor | None = None
        self._native_func: Any | None = None
        self.trace: GraphCompileTrace | None = None

    # ------------------------------------------------------------------
    # Kernel source inspection
    # ------------------------------------------------------------------

    def get_kernel_sources(self) -> dict[str, str]:
        """Return a ``{kernel_name: cuda_source}`` mapping for every compiled
        TileLang kernel in this graph.

        Torch-op wrappers (custom ops executed via PyTorch) are skipped.
        """
        from tilelang.jit.kernel import JITKernel
        sources: dict[str, str] = {}
        for name, kernel in self.kernels.items():
            if isinstance(kernel, JITKernel):
                sources[name] = kernel.get_kernel_source()
        return sources

    def show_kernel_sources(self) -> None:
        """Print the CUDA source of every compiled TileLang kernel."""
        sources = self.get_kernel_sources()
        if not sources:
            print("No TileLang kernels in this graph.")
            return
        for name, src in sources.items():
            print(f"{'=' * 60}")
            print(f"Kernel: {name}")
            print(f"{'=' * 60}")
            print(src)
            print()

    # ------------------------------------------------------------------
    # Core kernel dispatch (used by both normal and capture paths)
    # ------------------------------------------------------------------

    def _run_kernels(self, *args: torch.Tensor) -> torch.Tensor:
        """Execute the full kernel sequence and return the final output."""
        if self._native_func is not None:
            return self._run_native(*args)

        env: dict[str, torch.Tensor] = dict(zip(self.input_names, args))

        if self.dynamic:
            # Dynamic path: kernels compiled with out_idx=[-1] allocate
            # their own outputs and return them.
            for call in self.calls:
                result = self.kernels[call.gv_name](
                    *[env[name] for name in call.arg_names],
                )
                env[call.out_name] = result
        else:
            # Static path: use pre-allocated output buffers.
            for idx, call in enumerate(self.calls):
                out_buf = self._call_outputs[idx]
                if out_buf is None:
                    out_buf = env[call.out_name]
                self.kernels[call.gv_name](
                    *[env[name] for name in call.arg_names],
                    out_buf,
                )
                env[call.out_name] = out_buf

        return env[self.calls[-1].out_name]

    # ------------------------------------------------------------------
    # CUDA graph capture
    # ------------------------------------------------------------------

    def enable_cuda_graph(self, warmup_iters: int = 3) -> None:
        """Enable CUDA graph capture for subsequent calls.

        The graph is captured lazily on the next ``__call__``.  After capture,
        every invocation replays the recorded graph with near-zero CPU overhead.

        Input tensors must keep the same shapes / dtypes across calls; their
        data is copied into static buffers before each replay.

        Parameters
        ----------
        warmup_iters : int
            Number of warmup kernel runs before capture (default: 3).
            Warmup triggers any lazy GPU initialization (cuBLAS handles, etc.)
            so that the capture records a clean execution.
        """
        self._warmup_iters = warmup_iters
        self._cuda_graph_pending = True

    def _capture_cuda_graph(self, *args: torch.Tensor) -> None:
        """Allocate static input buffers, warm up, and capture the graph."""
        self._graph_inputs = tuple(torch.empty_like(a) for a in args)
        for src, dst in zip(args, self._graph_inputs):
            dst.copy_(src)

        # Warmup on a side stream to avoid polluting the default stream.
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(self._warmup_iters):
                self._run_kernels(*self._graph_inputs)
        torch.cuda.current_stream().wait_stream(s)

        # Capture.
        self._cuda_graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self._cuda_graph, stream=s):
            self._graph_output = self._run_kernels(*self._graph_inputs)
        torch.cuda.current_stream().wait_stream(s)

        self._cuda_graph_pending = False

    # ------------------------------------------------------------------
    # Native C++ dispatch
    # ------------------------------------------------------------------

    def enable_native_dispatch(self) -> None:
        """Build a compiled C++ dispatch function that replaces the Python loop.

        After calling this, :meth:`_run_kernels` delegates to a single
        native ``PackedFunc`` that sequences all kernel calls in C, avoiding
        per-kernel Python interpreter overhead.

        Not supported when the graph contains torch op wrappers.
        """
        if self._native_func is not None:
            return
        if any(isinstance(k, _TorchOpWrapper) for k in self.kernels.values()):
            raise RuntimeError(
                "native dispatch is not supported when the graph contains "
                "user-registered torch ops. Use the default Python dispatch."
            )

        # Assign a unique TIR parameter for each call's output to avoid
        # aliasing (same logic as _call_outputs indexing).
        param_names: list[str] = list(self.input_names)
        call_out_keys: list[str] = []
        for idx, call in enumerate(self.calls):
            if self._call_outputs[idx] is None:
                call_out_keys.append(call.out_name)
            else:
                key = f"__out_{idx}"
                param_names.append(key)
                call_out_keys.append(key)

        # Resolve each call's arg names to the correct parameter name,
        # tracking the "latest writer" for each out_name.
        arg_env: dict[str, str] = {n: n for n in self.input_names}
        call_resolved_args: list[list[str]] = []
        for idx, call in enumerate(self.calls):
            call_resolved_args.append([arg_env[n] for n in call.arg_names])
            arg_env[call.out_name] = call_out_keys[idx]

        # Register each kernel's PackedFunc in the global TVM registry so
        # that call_packed in the generated C code can find them.
        uid = id(self)
        reg_names: dict[str, str] = {}
        for name, kernel in self.kernels.items():
            exe_mod = kernel.adapter.executable
            reg_name = f"_tl_native_{uid}_{name}"
            tvm_ffi.register_global_func(reg_name, exe_mod[name])
            reg_names[name] = reg_name

        # Build TIR host function: one call_packed per kernel.
        tir_params = []
        var_map: dict[str, tir.Var] = {}
        for name in param_names:
            var = tir.Var(name, "handle")
            tir_params.append(var)
            var_map[name] = var

        stmts = []
        for idx, call in enumerate(self.calls):
            args = [var_map[n] for n in call_resolved_args[idx]]
            args.append(var_map[call_out_keys[idx]])
            stmts.append(tir.Evaluate(tir.call_packed(reg_names[call.gv_name], *args)))

        body = tir.SeqStmt(stmts) if len(stmts) > 1 else stmts[0]
        func = tir.PrimFunc(tir_params, body).with_attrs({
            "global_symbol": "graph_dispatch",
            "tir.noalias": True,
        })

        pipeline = tvm.transform.Sequential([tir.transform.MakePackedAPI()])
        host_mod = tvm.tir.build(
            tvm.IRModule({"graph_dispatch": func}), target="c", pipeline=pipeline,
        )
        rt_mod = tvm.runtime.Executable(host_mod).jit()

        self._native_func = rt_mod["graph_dispatch"]
        self._native_rt_mod = rt_mod  # prevent GC of compiled module

        # Pre-convert intermediate buffers to TVM NDArrays (zero-copy, cached).
        self._native_nd_intermediates: list[Any] = []
        for key in param_names[len(self.input_names):]:
            idx = int(key.split("_")[-1])  # __out_<idx>
            self._native_nd_intermediates.append(
                tvm.runtime.from_dlpack(self._call_outputs[idx])
            )

        # Precompute return value resolution.
        last_key = call_out_keys[-1]
        if last_key in self._input_name_set:
            self._native_return_input_idx: int | None = list(self.input_names).index(last_key)
            self._native_return_output_idx: int | None = None
        else:
            self._native_return_input_idx = None
            self._native_return_output_idx = int(last_key.split("_")[-1])

    def _run_native(self, *args: torch.Tensor) -> torch.Tensor:
        """Execute the kernel sequence via the compiled C dispatch function."""
        nd_args: list[Any] = [tvm.runtime.from_dlpack(a) for a in args]
        nd_args.extend(self._native_nd_intermediates)
        self._native_func(*nd_args)

        if self._native_return_input_idx is not None:
            return args[self._native_return_input_idx]
        return self._call_outputs[self._native_return_output_idx]

    # ------------------------------------------------------------------
    # Public call interface
    # ------------------------------------------------------------------

    def __call__(self, *args: torch.Tensor) -> torch.Tensor:
        _validate_tensor_args(args)
        if len(args) != len(self.input_names):
            raise ValueError(
                f"Expected {len(self.input_names)} inputs, got {len(args)}."
            )

        # Fast path: replay a previously captured CUDA graph.
        if self._cuda_graph is not None:
            for src, dst in zip(args, self._graph_inputs):
                dst.copy_(src)
            self._cuda_graph.replay()
            return self._graph_output

        # Lazy capture: first call after enable_cuda_graph().
        if self._cuda_graph_pending:
            self._capture_cuda_graph(*args)
            # The capture run already computed the output.
            return self._graph_output

        # Normal (non-graph) path.
        return self._run_kernels(*args)


# ---------------------------------------------------------------------------
# Dynamo fallback (data-dependent control flow support)
# ---------------------------------------------------------------------------

class DynamoGraphRunner:
    """Wraps a ``torch.compile``-generated callable for data-dependent control flow.

    When ``torch.export`` cannot trace a function (e.g. ``if tensor:``),
    we fall back to ``torch.compile`` with a custom TileLang backend.
    Each subgraph is compiled through TileLang's schedule pipeline;
    inter-subgraph control flow is handled by Python at runtime.

    .. note::
        CUDA graph capture and native C++ dispatch are not supported
        because Python control flow runs between subgraphs.
    """

    def __init__(self, compiled_fn: Callable) -> None:
        self._compiled_fn = compiled_fn

    def __call__(self, *args: Any) -> torch.Tensor:
        return self._compiled_fn(*args)


def _compile_subgraph(
    gm: torch.fx.GraphModule,
    example_inputs: list[torch.Tensor],
    arch: str,
    pass_configs: dict[str, Any] | None = None,
    compile_flags: list[str] | str | None = None,
) -> Callable:
    """Compile a single dynamo subgraph through TileLang's pipeline.

    Returns a callable matching the FX graph's output format (tuple).
    Raises on failure so the caller can fall back to ``gm.forward``.
    """
    # Build input_info for from_fx (handles both real and FakeTensors).
    input_info: list[tuple[list[int | tir.Var], str]] = []
    for tensor in example_inputs:
        shape: list[int | tir.Var] = []
        for s in tensor.shape:
            if isinstance(s, torch.SymInt):
                shape.append(tir.Var(str(s), "int64"))
            else:
                shape.append(int(s))
        dtype_str = str(tensor.dtype).replace("torch.", "")
        input_info.append((shape, dtype_str))

    # Convert FX graph → Relax IR.
    mod = from_fx(gm, input_info, unwrap_unit_return_tuple=True)

    # Schedule with TileLang rules.
    mod, _ = _schedule_relax_module(mod, arch)

    # Check if the scheduled module has any kernels.
    # Identity subgraphs (e.g. bool predicate pass-through) produce
    # no DataflowBlock after scheduling — return original forward.
    main_func = mod["main"]
    if (
        not isinstance(main_func.body, relax.SeqExpr)
        or len(main_func.body.blocks) == 0
    ):
        return gm.forward

    calls = _extract_call_sequence(main_func)
    input_names = _extract_input_names(main_func)

    if not calls:
        return gm.forward

    kernels = _compile_kernels(
        mod, calls, pass_configs=pass_configs, compile_flags=compile_flags,
    )

    # Build a mini GraphRunner for this subgraph.
    device = torch.device("cuda", torch.cuda.current_device())
    runner = GraphRunner(mod, kernels, calls, input_names, device)

    # torch.compile expects the backend callable to return a tuple
    # matching the FX graph's output format.
    def subgraph_runner(*args):
        return (runner._run_kernels(*args),)

    return subgraph_runner


def _tilelang_dynamo_backend(
    arch: str,
    pass_configs: dict[str, Any] | None = None,
    compile_flags: list[str] | str | None = None,
) -> Callable:
    """Create a ``torch.compile`` backend that compiles subgraphs via TileLang.

    Each captured FX subgraph is converted to Relax IR via ``from_fx``,
    scheduled with TileLang's rules, and compiled into GPU kernels.
    """

    def backend(gm: torch.fx.GraphModule, example_inputs: list[torch.Tensor]):
        gm.graph.eliminate_dead_code()

        try:
            return _compile_subgraph(gm, example_inputs, arch, pass_configs, compile_flags)
        except Exception:
            # If TileLang cannot compile this subgraph (e.g. unsupported
            # ops, 0-dim tensors), fall back to PyTorch eager execution.
            logger.debug(
                "TileLang dynamo backend: subgraph compilation failed, "
                "falling back to eager execution.",
                exc_info=True,
            )
            return gm.forward

    return backend


def _compile_graph_via_dynamo(
    func: Callable[..., torch.Tensor],
    example_args: tuple[torch.Tensor, ...],
    arch: str,
    pass_configs: dict[str, Any] | None = None,
    compile_flags: list[str] | str | None = None,
) -> DynamoGraphRunner:
    """Compile a PyTorch function via ``torch.compile`` with a TileLang backend.

    This is the fallback path when ``torch.export`` cannot trace the function
    (e.g. data-dependent Python control flow like ``if tensor:``).
    """
    import torch._dynamo as dynamo

    dynamo.reset()
    backend = _tilelang_dynamo_backend(arch, pass_configs, compile_flags)
    compiled_fn = torch.compile(func, backend=backend)

    # Warmup call to trigger compilation of all reachable subgraphs.
    with torch.no_grad():
        compiled_fn(*example_args)

    return DynamoGraphRunner(compiled_fn)


# ---------------------------------------------------------------------------
# Graph compilation pipeline
# ---------------------------------------------------------------------------

def _compile_graph(
    func: Callable[..., torch.Tensor],
    example_args: tuple[Any, ...],
    arch: str,
    custom_kernels: dict[str, PrimFunc] | None = None,
    pass_configs: dict[str, Any] | None = None,
    compile_flags: list[str] | str | None = None,
    dynamic_dims: dict[int, list[int]] | None = None,
) -> GraphRunner | DynamoGraphRunner:
    """Trace a PyTorch function, schedule each kernel, and compile."""
    t_total_start = time.perf_counter()
    is_dynamic = bool(dynamic_dims)
    trace = GraphCompileTrace(arch=arch, dynamic=is_dynamic)

    # Non-tensor args (bool, int, float) are not supported by torch.export's
    # Relax importer — use the dynamo path which handles mixed types natively.
    has_non_tensor = any(not isinstance(a, torch.Tensor) for a in example_args)
    if has_non_tensor:
        logger.info("Graph compile: non-tensor args detected, using dynamo path.")
        trace.compilation_path = "dynamo"
        return _compile_graph_via_dynamo(
            func, example_args, arch,
            pass_configs=pass_configs,
            compile_flags=compile_flags,
        )

    # 1. Trace PyTorch function → Relax IR
    logger.info("Graph compile: tracing via torch.export%s ...",
                " (dynamic)" if is_dynamic else "")
    t0 = time.perf_counter()
    try:
        mod, torch_op_map = _build_relax_module(
            func, example_args, dynamic_dims=dynamic_dims,
        )
    except (
        RuntimeError,
        torch.fx.experimental.symbolic_shapes.GuardOnDataDependentSymNode,
    ) as e:
        if "data-dependent" in str(e) or isinstance(
            e, torch.fx.experimental.symbolic_shapes.GuardOnDataDependentSymNode,
        ):
            logger.info("Graph compile: data-dependent control flow detected, "
                        "falling back to dynamo path.")
            trace.compilation_path = "export+dynamo_fallback"
            return _compile_graph_via_dynamo(
                func, example_args, arch,
                pass_configs=pass_configs,
                compile_flags=compile_flags,
            )
        raise
    trace.trace_time_ms = (time.perf_counter() - t0) * 1000
    trace.compilation_path = "export"
    logger.info("Graph compile: tracing done (%.1f ms).", trace.trace_time_ms)

    # 2. Schedule with TileLang rules
    t0 = time.perf_counter()
    mod, schedule_result = _schedule_relax_module(mod, arch, trace=trace)
    trace.schedule_time_ms = (time.perf_counter() - t0) * 1000
    logger.info("Graph compile: scheduling done (%.1f ms).", trace.schedule_time_ms)

    # 3. Inject custom kernels (replace TIR functions by name)
    if custom_kernels:
        mod = _inject_custom_kernels(mod, custom_kernels)

    # 4. Extract execution plan
    main_func = mod["main"]
    calls = _extract_call_sequence(main_func)
    input_names = _extract_input_names(main_func)

    # 5. Compile each kernel with per-kernel tracing.
    t_compile_start = time.perf_counter()
    kernels: dict[str, Any] = {}
    out_idx = [-1] if is_dynamic else None
    compile_kwargs: dict[str, Any] = {}
    if pass_configs:
        compile_kwargs["pass_configs"] = pass_configs
    if compile_flags:
        compile_kwargs["compile_flags"] = compile_flags
    if out_idx is not None:
        compile_kwargs["out_idx"] = out_idx

    for call in calls:
        tir_func = mod[call.gv_name]
        is_torch_op = (
            isinstance(tir_func, tir.PrimFunc)
            and tir_func.attrs
            and tir_func.attrs.get("torch_op")
        )

        kt = KernelTrace(name=call.gv_name)
        kt.schedule_rule = trace.schedule_matches.get(call.gv_name, "unknown")
        kt.is_torch_op = is_torch_op
        if schedule_result is not None and call.gv_name in schedule_result.tir_snapshots:
            unsched, sched = schedule_result.tir_snapshots[call.gv_name]
            kt.set_tir_funcs(unsched, sched)

        if isinstance(tir_func, tir.PrimFunc):
            # Extract input/output shape info from buffer_map.
            params_list = list(tir_func.params)
            for p in params_list[:-1]:
                buf = tir_func.buffer_map.get(p)
                if buf is not None:
                    kt.input_shapes.append(tuple(
                        int(s) if isinstance(s, tir.IntImm) else str(s)
                        for s in buf.shape
                    ))
            out_shape, out_dtype, out_dyn = _output_buffer_info(tir_func)
            kt.output_shape = tuple(
                int(s) if isinstance(s, (int,)) else str(s) for s in out_shape
            )
            kt.output_dtype = out_dtype
            kt.is_dynamic = out_dyn

        if is_torch_op:
            torch_callable = torch_op_map.get(call.gv_name)
            if torch_callable is None:
                raise RuntimeError(
                    f"torch_op attribute set on '{call.gv_name}' but no "
                    f"matching entry in torch_op_map."
                )
            kernels[call.gv_name] = _TorchOpWrapper(
                torch_callable, dynamic=(out_idx is not None),
            )
            logger.info("  %-40s [torch_op]", call.gv_name)
        else:
            t_k = time.perf_counter()
            prepared = _lower_primfunc_for_tilelang(tir_func, call.gv_name)
            kernels[call.gv_name] = tilelang.compile(prepared, **compile_kwargs)
            kt.compile_time_ms = (time.perf_counter() - t_k) * 1000
            logger.info("  %-40s [%s] %.1f ms",
                        call.gv_name, kt.schedule_rule, kt.compile_time_ms)

        trace.kernels.append(kt)

    trace.compile_time_ms = (time.perf_counter() - t_compile_start) * 1000
    trace.total_time_ms = (time.perf_counter() - t_total_start) * 1000
    logger.info("Graph compile: total %.1f ms (%d kernels).",
                trace.total_time_ms, len(trace.kernels))

    # 6. Build runner
    device = example_args[0].device
    runner = GraphRunner(
        mod, kernels, calls, input_names, device,
        dynamic=is_dynamic,
    )
    runner.trace = trace
    return runner


# ---------------------------------------------------------------------------
# GraphJITImpl – the @tilelang.jit(mode="graph") wrapper
# ---------------------------------------------------------------------------

class GraphJITImpl:
    """Decorator wrapper for graph-mode JIT compilation of PyTorch functions.

    When ``dynamic_dims`` is ``None``, each unique shape signature triggers
    a separate compilation (shape-keyed recompilation).

    When ``dynamic_dims`` is provided, the kernel is compiled **once** with
    symbolic dimensions and reused for any concrete shape — no recompilation.
    """

    def __init__(
        self,
        func: Callable[..., torch.Tensor],
        *,
        arch: str | None = None,
        custom_kernels: dict[str, PrimFunc] | None = None,
        pass_configs: dict[str, Any] | None = None,
        compile_flags: list[str] | str | None = None,
        cuda_graph: bool = False,
        native: bool = False,
        dynamic_dims: dict[int, list[int]] | None = None,
    ) -> None:
        self.func = func
        self.arch = arch
        self.custom_kernels = custom_kernels or {}
        self.pass_configs = pass_configs
        self.compile_flags = compile_flags
        self.cuda_graph = cuda_graph
        self.native = native
        self.dynamic_dims = dynamic_dims
        self._cache: dict[tuple[Any, ...], GraphRunner] = {}
        # Single cached runner for dynamic shapes (compiled once).
        self._dynamic_runner: GraphRunner | None = None
        # Cached DynamoGraphRunner (set when torch.export fails and we
        # fall back to torch.compile for data-dependent control flow).
        self._dynamo_runner: DynamoGraphRunner | None = None

    def _configure_runner(self, runner: GraphRunner) -> None:
        """Apply native dispatch and CUDA graph settings to a GraphRunner."""
        if self.native:
            runner.enable_native_dispatch()
        if self.cuda_graph:
            runner.enable_cuda_graph()

    def compile(self, *example_args, _sig: tuple[Any, ...] | None = None) -> GraphRunner | DynamoGraphRunner:
        """Trace, schedule, and compile for the given example inputs."""
        _validate_tensor_args(example_args)

        if self.dynamic_dims:
            if self._dynamic_runner is not None:
                return self._dynamic_runner
            arch = _resolve_arch(self.arch)
            runner = _compile_graph(
                self.func,
                example_args,
                arch,
                custom_kernels=self.custom_kernels,
                pass_configs=self.pass_configs,
                compile_flags=self.compile_flags,
                dynamic_dims=self.dynamic_dims,
            )
            if isinstance(runner, GraphRunner):
                self._configure_runner(runner)
            self._dynamic_runner = runner
            return runner

        # Static path: shape-keyed caching.
        sig = _sig if _sig is not None else _signature_from_inputs(example_args)
        cached = self._cache.get(sig)
        if cached is not None:
            return cached

        arch = _resolve_arch(self.arch)
        runner = _compile_graph(
            self.func,
            example_args,
            arch,
            custom_kernels=self.custom_kernels,
            pass_configs=self.pass_configs,
            compile_flags=self.compile_flags,
        )
        if isinstance(runner, DynamoGraphRunner):
            self._dynamo_runner = runner
            return runner
        self._configure_runner(runner)
        self._cache[sig] = runner
        return runner

    def __call__(self, *args) -> torch.Tensor:
        # Dynamo fallback path (cached after first data-dependent error).
        if self._dynamo_runner is not None:
            return self._dynamo_runner(*args)

        if self.dynamic_dims:
            runner = self._dynamic_runner
            if runner is None:
                runner = self.compile(*args)
            return runner(*args)

        # Static path: check cache before full validation.
        sig = _signature_from_inputs(args)
        runner = self._cache.get(sig)
        if runner is None:
            runner = self.compile(*args, _sig=sig)
        return runner(*args)

    # ------------------------------------------------------------------
    # Kernel source inspection
    # ------------------------------------------------------------------

    def _iter_graph_runners(self) -> list[GraphRunner]:
        """Collect all GraphRunners across caching paths."""
        runners: list[GraphRunner] = []
        if self._dynamic_runner is not None and isinstance(self._dynamic_runner, GraphRunner):
            runners.append(self._dynamic_runner)
        runners.extend(self._cache.values())
        return runners

    def get_kernel_sources(self) -> dict[str, str]:
        """Return ``{kernel_name: cuda_source}`` for all compiled kernels.

        If multiple shape-specialized compilations exist, all are included.
        Raises ``RuntimeError`` if no compilation has been performed yet.
        """
        runners = self._iter_graph_runners()
        if not runners:
            if self._dynamo_runner is not None:
                raise RuntimeError(
                    "Kernel sources are not available for dynamo-compiled "
                    "graphs (Python control flow fallback)."
                )
            raise RuntimeError("No compilation has been performed yet. Call the function first.")
        sources: dict[str, str] = {}
        for runner in runners:
            sources.update(runner.get_kernel_sources())
        return sources

    def show_kernel_sources(self) -> None:
        """Print CUDA source of every compiled TileLang kernel."""
        runners = self._iter_graph_runners()
        if not runners:
            if self._dynamo_runner is not None:
                print("Kernel sources are not available for dynamo-compiled "
                      "graphs (Python control flow fallback).")
                return
            print("No compilation has been performed yet. Call the function first.")
            return
        for runner in runners:
            runner.show_kernel_sources()

    # ------------------------------------------------------------------
    # Compilation trace
    # ------------------------------------------------------------------

    def get_trace(self) -> GraphCompileTrace | list[GraphCompileTrace]:
        """Return the compilation trace(s).

        For dynamic shapes or single-shape calls, returns a single
        ``GraphCompileTrace``.  For multiple shape-keyed compilations,
        returns a list.  Raises ``RuntimeError`` if no compilation yet.
        """
        runners = self._iter_graph_runners()
        if not runners:
            if self._dynamo_runner is not None:
                trace = GraphCompileTrace(
                    compilation_path="dynamo", arch=_resolve_arch(self.arch),
                )
                return trace
            raise RuntimeError("No compilation has been performed yet. Call the function first.")
        traces = [r.trace for r in runners if r.trace is not None]
        if len(traces) == 1:
            return traces[0]
        return traces

    def show_trace(self) -> None:
        """Print a human-readable compilation trace summary."""
        result = self.get_trace()
        if isinstance(result, list):
            for i, t in enumerate(result):
                print(f"--- Compilation #{i} ---")
                print(t.summary())
                print()
        else:
            print(result.summary())
