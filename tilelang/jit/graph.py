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

from dataclasses import dataclass
from typing import Any, Callable

import torch
import tvm_ffi
from torch.export import export
from torch.nn import Module
from tvm import relax, tir
from tvm.ir import GlobalVar
from tvm.relax.frontend.torch import from_exported_program
from tvm.tir import PrimFunc

import tilelang
from tilelang import tvm
from tilelang.schedule.gpu import default_schedule_rules

# TVM op pattern constant: prevents FuseOps from fusing this function.
_OP_PATTERN_OPAQUE = 8


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


def _build_relax_module(
    func: Callable[..., torch.Tensor],
    example_args: tuple[torch.Tensor, ...],
) -> tuple[tvm.IRModule, dict[str, Any]]:
    """Trace and convert a PyTorch function to Relax IR.

    Returns ``(mod, torch_op_map)`` where *torch_op_map* maps Relax GV
    names to original torch callables for user-registered custom ops.
    """
    class _WrappedModule(Module):
        def __init__(self, inner: Callable[..., torch.Tensor]):
            super().__init__()
            self.inner = inner

        def forward(self, *args: torch.Tensor) -> torch.Tensor:
            return self.inner(*args)

    wrapped = _WrappedModule(func)
    exported_program = export(wrapped, args=example_args, dynamic_shapes={})

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


def _schedule_relax_module(
    mod: tvm.IRModule,
    arch: str,
) -> tvm.IRModule:
    target = tvm.target.cuda(arch=arch)
    with target:
        mod = relax.transform.LegalizeOps()(mod)
        mod = relax.transform.AnnotateTIROpPattern()(mod)
        mod = relax.transform.FoldConstant()(mod)
        mod = relax.transform.FuseOps()(mod)
        mod = relax.transform.FuseTIR()(mod)
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
    from tilelang.engine.phase import NormalizeScheduledIR
    mod = tvm.IRModule({name: func})
    mod = NormalizeScheduledIR(mod)
    return mod[name]


def _output_buffer_info(func: tir.PrimFunc) -> tuple[tuple[int, ...], str]:
    output_buffer = func.buffer_map[list(func.params)[-1]]
    return tuple(int(extent) for extent in output_buffer.shape), output_buffer.dtype


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
        (tuple(int(v) for v in arg.shape), str(arg.dtype), arg.device.type, arg.device.index)
        for arg in args
    )


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

    The wrapper calls the original torch op normally, then copies the result
    into the pre-allocated output buffer.
    """

    def __init__(self, torch_callable: Any) -> None:
        self.torch_callable = torch_callable

    def __call__(self, *args_and_out: torch.Tensor) -> None:
        inputs = args_and_out[:-1]
        out_buf = args_and_out[-1]
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
    ) -> None:
        self.kernels = kernels
        self.calls = calls
        self.input_names = input_names
        self.device = device

        # Pre-allocate one output buffer per call, indexed by position.
        # Keying by name would alias calls that share the same out_name.
        self._input_name_set = frozenset(input_names)
        input_set = self._input_name_set
        self._call_outputs: list[torch.Tensor | None] = []
        for call in calls:
            if call.out_name in input_set:
                self._call_outputs.append(None)
            else:
                shape, dtype = _output_buffer_info(mod[call.gv_name])
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

    # ------------------------------------------------------------------
    # Core kernel dispatch (used by both normal and capture paths)
    # ------------------------------------------------------------------

    def _run_kernels(self, *args: torch.Tensor) -> torch.Tensor:
        """Execute the full kernel sequence and return the final output."""
        if self._native_func is not None:
            return self._run_native(*args)
        env: dict[str, torch.Tensor] = dict(zip(self.input_names, args))
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
# Graph compilation pipeline
# ---------------------------------------------------------------------------

def _compile_graph(
    func: Callable[..., torch.Tensor],
    example_args: tuple[torch.Tensor, ...],
    arch: str,
    custom_kernels: dict[str, PrimFunc] | None = None,
    pass_configs: dict[str, Any] | None = None,
    compile_flags: list[str] | str | None = None,
) -> GraphRunner:
    """Trace a PyTorch function, schedule each kernel, and compile."""
    # 1. Trace PyTorch function → Relax IR
    mod, torch_op_map = _build_relax_module(func, example_args)

    # 2. Schedule with TileLang rules
    mod = _schedule_relax_module(mod, arch)

    # 3. Inject custom kernels (replace TIR functions by name)
    if custom_kernels:
        mod = _inject_custom_kernels(mod, custom_kernels)

    # 4. Extract execution plan
    main_func = mod["main"]
    calls = _extract_call_sequence(main_func)
    input_names = _extract_input_names(main_func)

    # 5. Compile each kernel.
    kernels: dict[str, Any] = {}
    for call in calls:
        tir_func = mod[call.gv_name]
        is_torch_op = (
            isinstance(tir_func, tir.PrimFunc)
            and tir_func.attrs
            and tir_func.attrs.get("torch_op")
        )
        if is_torch_op:
            # Wrap the original torch callable to match GraphRunner's
            # (*inputs, output_buffer) calling convention.
            torch_callable = torch_op_map.get(call.gv_name)
            if torch_callable is None:
                raise RuntimeError(
                    f"torch_op attribute set on '{call.gv_name}' but no "
                    f"matching entry in torch_op_map."
                )
            kernels[call.gv_name] = _TorchOpWrapper(torch_callable)
        else:
            prepared = _lower_primfunc_for_tilelang(tir_func, call.gv_name)
            kernels[call.gv_name] = tilelang.compile(
                prepared,
                pass_configs=pass_configs,
                compile_flags=compile_flags,
            )

    # 6. Build pre-allocated runner
    device = example_args[0].device
    return GraphRunner(mod, kernels, calls, input_names, device)


# ---------------------------------------------------------------------------
# GraphJITImpl – the @tilelang.jit(mode="graph") wrapper
# ---------------------------------------------------------------------------

class GraphJITImpl:
    """Decorator wrapper for graph-mode JIT compilation of PyTorch functions.

    On first call (or when the input shape signature changes), the PyTorch
    function is traced, scheduled, and compiled.  Subsequent calls with the
    same shape signature reuse the cached ``GraphRunner``.
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
    ) -> None:
        self.func = func
        self.arch = arch
        self.custom_kernels = custom_kernels or {}
        self.pass_configs = pass_configs
        self.compile_flags = compile_flags
        self.cuda_graph = cuda_graph
        self.native = native
        self._cache: dict[tuple[Any, ...], GraphRunner] = {}

    def compile(self, *example_args: torch.Tensor) -> GraphRunner:
        """Trace, schedule, and compile for the given example inputs."""
        tensors = _validate_tensor_args(example_args)
        sig = _signature_from_inputs(tensors)
        cached = self._cache.get(sig)
        if cached is not None:
            return cached

        arch = _resolve_arch(self.arch)
        runner = _compile_graph(
            self.func,
            tensors,
            arch,
            custom_kernels=self.custom_kernels,
            pass_configs=self.pass_configs,
            compile_flags=self.compile_flags,
        )
        if self.native:
            runner.enable_native_dispatch()
        if self.cuda_graph:
            runner.enable_cuda_graph()
        self._cache[sig] = runner
        return runner

    def __call__(self, *args: torch.Tensor) -> torch.Tensor:
        # Fast path: check cache before full validation.
        sig = _signature_from_inputs(args)
        runner = self._cache.get(sig)
        if runner is None:
            runner = self.compile(*args)
        return runner(*args)
