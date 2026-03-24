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

import types
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


def _build_relax_module(
    func: Callable[..., torch.Tensor],
    example_args: tuple[torch.Tensor, ...],
    custom_convert_map: dict[str, Any] | None = None,
) -> tvm.IRModule:
    class _WrappedModule(Module):
        def __init__(self, inner: Callable[..., torch.Tensor]):
            super().__init__()
            self.inner = inner

        def forward(self, *args: torch.Tensor) -> torch.Tensor:
            return self.inner(*args)

    wrapped = _WrappedModule(func)
    exported_program = export(wrapped, args=example_args, dynamic_shapes={})
    return from_exported_program(
        exported_program,
        run_ep_decomposition=None,
        keep_params_as_input=None,
        unwrap_unit_return_tuple=None,
        no_bind_return_tuple=None,
        custom_convert_map=custom_convert_map,
    )


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
# JITKernel auto-detection for graph-mode
# ---------------------------------------------------------------------------

_tl_lib_counter: int = 0


def _detect_jit_kernels(func: Callable) -> dict[str, tilelang.JITKernel]:
    """Find JITKernel instances referenced by *func*'s globals or closure."""
    from tilelang.jit.kernel import JITKernel

    found: dict[str, tilelang.JITKernel] = {}
    # Globals referenced in the bytecode.
    for name in func.__code__.co_names:
        val = func.__globals__.get(name)
        if isinstance(val, JITKernel):
            found[name] = val
    # Closure (free) variables.
    if func.__closure__:
        for name, cell in zip(func.__code__.co_freevars, func.__closure__):
            try:
                val = cell.cell_contents
                if isinstance(val, JITKernel):
                    found[name] = val
            except ValueError:
                pass
    return found


class _JITKernelTracer:
    """Registers torch custom ops for detected JITKernels so that
    ``torch.export`` can trace through them, and provides a
    ``custom_convert_map`` that emits Relax ``call_tir`` nodes
    referencing the kernel's original PrimFunc.
    """

    def __init__(self) -> None:
        global _tl_lib_counter
        _tl_lib_counter += 1
        self._ns = f"tl_jit_{_tl_lib_counter}"
        self._lib = torch.library.Library(self._ns, "DEF")
        # {var_name: (custom_op_callable, relax_gv_name)}
        self._ops: dict[str, tuple[Callable, str]] = {}
        # {relax_gv_name: JITKernel}
        self._kernels: dict[str, tilelang.JITKernel] = {}
        # {relax_gv_name: (shape, dtype)} — cached to avoid repeated extraction
        self._buf_info: dict[str, tuple[tuple[int, ...], str]] = {}

    @property
    def precompiled_kernels(self) -> dict[str, tilelang.JITKernel]:
        """Mapping from Relax GlobalVar name to the pre-compiled JITKernel."""
        return dict(self._kernels)

    def register(self, var_name: str, kernel: tilelang.JITKernel) -> None:
        """Create a torch custom op backed by *kernel*."""
        primfunc = kernel.prim_func
        out_shape, out_dtype = _output_buffer_info(primfunc)
        n_inputs = len(list(primfunc.params)) - 1

        op_name = f"kernel_{var_name}"
        relax_name = f"tl_{var_name}"

        # Schema: tl_jit_N::kernel_<name>(Tensor a, ...) -> Tensor
        param_list = ", ".join(f"Tensor arg{i}" for i in range(n_inputs))
        self._lib.define(f"{op_name}({param_list}) -> Tensor")

        torch_dtype = getattr(torch, out_dtype)

        def cuda_impl(*args, _k=kernel, _s=out_shape, _d=torch_dtype):
            out = torch.empty(_s, device=args[0].device, dtype=_d)
            _k(*args, out)
            return out

        def meta_impl(*args, _s=out_shape, _d=torch_dtype):
            return torch.empty(_s, device=args[0].device, dtype=_d)

        self._lib.impl(op_name, cuda_impl, "CUDA")
        self._lib.impl(op_name, meta_impl, "Meta")

        op_callable = getattr(getattr(torch.ops, self._ns), op_name)
        self._ops[var_name] = (op_callable, relax_name)
        self._kernels[relax_name] = kernel
        self._buf_info[relax_name] = (out_shape, out_dtype)

    # ------------------------------------------------------------------

    def make_traced_func(self, func: Callable) -> Callable:
        """Return a copy of *func* with JITKernel refs swapped for custom ops."""
        replacements = {name: op for name, (op, _) in self._ops.items()}

        new_globals = dict(func.__globals__)
        for name, op in replacements.items():
            if name in new_globals:
                new_globals[name] = op

        new_closure = func.__closure__
        if func.__closure__:
            free_vars = func.__code__.co_freevars
            if any(n in free_vars for n in replacements):
                cells: list[types.CellType] = []
                for name, cell in zip(free_vars, func.__closure__):
                    if name in replacements:
                        cells.append(types.CellType(replacements[name]))
                    else:
                        cells.append(cell)
                new_closure = tuple(cells)

        return types.FunctionType(
            func.__code__, new_globals, func.__name__,
            func.__defaults__, new_closure,
        )

    # ------------------------------------------------------------------

    def get_convert_map(self) -> dict[str, Callable]:
        """Build ``custom_convert_map`` for ``from_exported_program``."""
        from tvm import relax as _relax

        convert_map: dict[str, Callable] = {}
        for var_name, (_, relax_name) in self._ops.items():
            primfunc = self._kernels[relax_name].prim_func
            out_shape, out_dtype = self._buf_info[relax_name]

            def _make(pf, rn, os, od):
                def converter(node, importer):
                    args = [importer.env[a] for a in node.args if a in importer.env]
                    bb = importer.block_builder
                    annotated = pf.with_attr("global_symbol", rn).with_attr(
                        "op_pattern", _OP_PATTERN_OPAQUE,
                    ).with_attr("tir.is_scheduled", True)
                    gv = bb.add_func(annotated, rn)
                    sinfo = _relax.TensorStructInfo(os, od)
                    return bb.emit(_relax.call_tir(gv, args, sinfo))
                return converter

            op_key = f"kernel_{var_name}.default"
            convert_map[op_key] = _make(primfunc, relax_name, out_shape, out_dtype)
        return convert_map


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
        kernels: dict[str, tilelang.JITKernel],
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
        """
        if self._native_func is not None:
            return

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
    # 1. Detect JITKernels referenced by the function
    detected = _detect_jit_kernels(func)
    tracer: _JITKernelTracer | None = None
    traced_func = func
    custom_convert_map: dict[str, Any] | None = None
    if detected:
        tracer = _JITKernelTracer()
        for name, kernel in detected.items():
            tracer.register(name, kernel)
        traced_func = tracer.make_traced_func(func)
        custom_convert_map = tracer.get_convert_map()

    # 2. Trace PyTorch function → Relax IR
    mod = _build_relax_module(traced_func, example_args, custom_convert_map=custom_convert_map)

    # 3. Schedule with TileLang rules
    mod = _schedule_relax_module(mod, arch)

    # 4. Inject custom kernels (replace TIR functions by name)
    if custom_kernels:
        mod = _inject_custom_kernels(mod, custom_kernels)

    # 5. Extract execution plan
    main_func = mod["main"]
    calls = _extract_call_sequence(main_func)
    input_names = _extract_input_names(main_func)

    # 6. Compile each kernel.
    # Pre-compiled JITKernels are re-compiled without out_idx so they accept
    # the output buffer as an explicit argument (graph-mode calling convention).
    pre_compiled = tracer.precompiled_kernels if tracer else {}
    kernels: dict[str, tilelang.JITKernel] = {}
    for call in calls:
        if call.gv_name in pre_compiled:
            kernels[call.gv_name] = tilelang.compile(
                pre_compiled[call.gv_name].prim_func,
                pass_configs=pass_configs,
                compile_flags=compile_flags,
            )
        else:
            prepared = _lower_primfunc_for_tilelang(mod[call.gv_name], call.gv_name)
            kernels[call.gv_name] = tilelang.compile(
                prepared,
                pass_configs=pass_configs,
                compile_flags=compile_flags,
            )

    # 7. Build pre-allocated runner
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
