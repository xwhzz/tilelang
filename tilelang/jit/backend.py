"""TileLang backend for ``torch.compile``.

Registers TileLang as a custom ``torch.compile`` backend so that Dynamo
handles subgraph capture and TileLang handles compilation::

    compiled = torch.compile(model, backend="tilelang")
    result = compiled(x)

Options can be passed via the ``options`` dict::

    compiled = torch.compile(model, backend="tilelang", options={
        "arch": "sm_90",
        "print_trace": True,
    })
"""

from __future__ import annotations

import hashlib
import logging
import os
import threading
from typing import Any, Callable

import torch

logger = logging.getLogger(__name__)

# Module-level trace registry for inspection after compilation.
_compilation_traces: list = []
_compilation_traces_lock = threading.Lock()

# ---------------------------------------------------------------------------
# In-memory compile cache
# ---------------------------------------------------------------------------
# Cache keyed by (graph_structure_hash, arch) → compiled runner callable.
# In LLaMA-2, all 32 layers share ~11 subgraph structures, giving ~97%
# cache hit rate (11 compiles instead of 352).
_subgraph_cache: dict[str, Callable] = {}
_subgraph_cache_lock = threading.Lock()

# Disk cache directory
_CACHE_DIR = os.path.join(
    os.environ.get("TILELANG_CACHE_DIR",
                    os.path.join(os.path.expanduser("~"), ".cache", "tilelang")),
    "graphs",
)


def _compute_subgraph_key(gm: torch.fx.GraphModule, input_info: list, arch: str) -> str:
    """Compute a hash key for the subgraph structure + shapes + constants + arch.

    The key captures the op sequence, constant literal values, and input
    specifications — but NOT specific tensor identities, so structurally
    identical subgraphs across different layers share the same cache entry.
    """
    parts = [arch]
    for node in gm.graph.nodes:
        # Include op type and target name (not specific tensor values)
        if node.op == "placeholder":
            parts.append(f"placeholder")
        elif node.op == "output":
            parts.append(f"output")
        elif node.op == "call_function":
            parts.append(f"call:{node.target.__name__}")
            # Include constant (non-Node) args so graphs with different
            # literal values (e.g. x*2.0 vs x*0.5) hash differently.
            for arg in node.args:
                if not isinstance(arg, torch.fx.Node):
                    parts.append(repr(arg))
            for k, v in sorted(node.kwargs.items()):
                if not isinstance(v, torch.fx.Node):
                    parts.append(f"{k}={v!r}")
        elif node.op == "call_method":
            parts.append(f"method:{node.target}")
        else:
            parts.append(f"{node.op}:{node.target}")
    for shape, dtype in input_info:
        parts.append(f"{'x'.join(str(s) for s in shape)}:{dtype}")
    key_str = "|".join(parts)
    return hashlib.sha256(key_str.encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Output dtype helpers
# ---------------------------------------------------------------------------

def _extract_output_dtypes(gm: torch.fx.GraphModule) -> list[torch.dtype]:
    """Extract expected output dtypes from FX graph output node metadata."""
    output_nodes = [n for n in gm.graph.nodes if n.op == "output"]
    if not output_nodes:
        return []

    output_node = output_nodes[0]
    ret = output_node.args[0]

    def _dtype_of(node_or_val):
        if isinstance(node_or_val, torch.fx.Node):
            for key in ("val", "example_value"):
                val = node_or_val.meta.get(key)
                if val is not None and hasattr(val, "dtype"):
                    return val.dtype
        logger.debug("Could not determine output dtype for %s, assuming float32", node_or_val)
        return torch.float32

    if isinstance(ret, (tuple, list)):
        return [_dtype_of(v) for v in ret]
    return [_dtype_of(ret)]


def _dtype_size(dtype: torch.dtype) -> int:
    """Element size in bytes, cached. Returns 0 for unknown types."""
    size = _dtype_size_cache.get(dtype)
    if size is None:
        try:
            size = torch.tensor([], dtype=dtype).element_size()
        except (TypeError, RuntimeError):
            size = 0
        _dtype_size_cache[dtype] = size
    return size

_dtype_size_cache: dict[torch.dtype, int] = {}


def _should_narrow(actual: torch.dtype, expected: torch.dtype) -> bool:
    """Return True if *actual* should be cast down to *expected*."""
    if actual == expected:
        return False
    a, e = _dtype_size(actual), _dtype_size(expected)
    return a > e > 0


# ---------------------------------------------------------------------------
# Subgraph runner factory
# ---------------------------------------------------------------------------

def _make_runner(vm_main, expected_dtypes):
    """Build a torch-compatible runner closure around a VM entry point."""
    from tilelang import tvm as _tvm

    def _tvm_to_torch(obj):
        if hasattr(obj, "__dlpack__"):
            return torch.from_dlpack(obj)
        if hasattr(obj, "numpy"):
            return torch.from_numpy(obj.numpy()).cuda()
        raise TypeError(f"Cannot convert TVM result of type {type(obj)} to torch.Tensor")

    def _unpack_vm_result(result):
        if isinstance(result, (tuple, list)):
            return [_tvm_to_torch(r) for r in result]
        if hasattr(result, "__len__") and hasattr(result, "__getitem__"):
            n = len(result)
            if n > 0 and not hasattr(result, "__dlpack__"):
                return [_tvm_to_torch(result[i]) for i in range(n)]
        return [_tvm_to_torch(result)]

    def subgraph_runner(*args):
        tvm_args = [
            _tvm.runtime.from_dlpack(a) for a in args
            if isinstance(a, torch.Tensor)
        ]
        result = vm_main(*tvm_args)
        outputs = _unpack_vm_result(result)
        for i, t in enumerate(outputs):
            if i < len(expected_dtypes) and _should_narrow(t.dtype, expected_dtypes[i]):
                outputs[i] = t.to(expected_dtypes[i])
        return tuple(outputs)

    return subgraph_runner


# ---------------------------------------------------------------------------
# Direct kernel runner (bypasses VM for ~10x lower per-call overhead)
# ---------------------------------------------------------------------------

_TVM_DTYPE_TO_TORCH: dict[str, torch.dtype] = {
    "float16": torch.float16,
    "float32": torch.float32,
    "float64": torch.float64,
    "bfloat16": torch.bfloat16,
    "int8": torch.int8,
    "int16": torch.int16,
    "int32": torch.int32,
    "int64": torch.int64,
    "uint8": torch.uint8,
    "bool": torch.bool,
}


def _make_direct_runner(param_names, call_seq, output_names, rt_mod, tvm_device, expected_dtypes):
    """Build a runner that calls TIR kernels directly, bypassing the VM.

    Single-kernel subgraphs (most common) use a specialized fast path.
    Multi-kernel subgraphs pre-allocate intermediate buffers and execute
    the kernel sequence with minimal Python overhead.
    """
    from tilelang import tvm as _tvm

    from_dlpack = _tvm.runtime.from_dlpack
    output_name_set = frozenset(output_names)
    torch_device = torch.device("cuda", tvm_device.index)

    # Pre-fetch kernel PackedFuncs.
    kernels = {rec.func_name: rt_mod[rec.func_name] for rec in call_seq}

    # Pre-allocate workspace for intermediates (reused across calls).
    workspace = {}
    for rec in call_seq:
        if rec.out_name not in output_name_set:
            workspace[rec.out_name] = _tvm.runtime.empty(
                rec.out_shape, rec.out_dtype, tvm_device,
            )

    n_params = len(param_names)

    # --- Fast path: single kernel, single output ---
    if len(call_seq) == 1 and len(output_names) == 1:
        rec = call_seq[0]
        func = kernels[rec.func_name]
        torch_dtype = _TVM_DTYPE_TO_TORCH[rec.out_dtype]
        narrow = bool(expected_dtypes and _should_narrow(torch_dtype, expected_dtypes[0]))
        narrow_dtype = expected_dtypes[0] if narrow else None
        out_shape = rec.out_shape

        def _single_runner(*args):
            tvm_inputs = [from_dlpack(args[i]) for i in range(n_params)]
            out = torch.empty(out_shape, dtype=torch_dtype, device=torch_device)
            tvm_inputs.append(from_dlpack(out))
            func(*tvm_inputs)
            if narrow:
                out = out.to(narrow_dtype)
            return (out,)

        return _single_runner

    # --- General path: multi-kernel sequence ---
    # Pre-compute per-kernel info to avoid dict lookups in hot loop.
    _kernel_plan = []  # [(func, arg_name_list, is_output, torch_dtype, out_shape, out_name)]
    for rec in call_seq:
        is_out = rec.out_name in output_name_set
        td = _TVM_DTYPE_TO_TORCH[rec.out_dtype] if is_out else None
        _kernel_plan.append((
            kernels[rec.func_name], rec.arg_names, is_out,
            td, rec.out_shape, rec.out_name,
        ))
    # Pre-compute output collection plan: list of (source, key) tuples.
    param_name_set = frozenset(param_names)
    param_idx = {name: i for i, name in enumerate(param_names)}
    _output_plan = []  # [("computed", name) | ("passthrough", idx)]
    for name in output_names:
        if name in param_name_set:
            _output_plan.append(("passthrough", param_idx[name]))
        else:
            _output_plan.append(("computed", name))

    def _multi_runner(*args):
        env = {}
        for i in range(n_params):
            env[param_names[i]] = from_dlpack(args[i])

        torch_outs = {}
        for func, arg_names, is_out, td, shape, out_name in _kernel_plan:
            fa = [env[n] for n in arg_names]
            if is_out:
                tout = torch.empty(shape, dtype=td, device=torch_device)
                tvm_out = from_dlpack(tout)
                torch_outs[out_name] = tout
            else:
                tvm_out = workspace[out_name]
            fa.append(tvm_out)
            func(*fa)
            env[out_name] = tvm_out

        results = []
        for source, key in _output_plan:
            if source == "computed":
                results.append(torch_outs[key])
            else:
                results.append(args[key])
        for i, t in enumerate(results):
            if i < len(expected_dtypes) and _should_narrow(t.dtype, expected_dtypes[i]):
                results[i] = t.to(expected_dtypes[i])
        return tuple(results)

    return _multi_runner


# ---------------------------------------------------------------------------
# CUDA Graph runner (eliminates kernel launch overhead on repeated calls)
# ---------------------------------------------------------------------------

# Environment variable to disable CUDA graphs (e.g. for debugging).
_CUDA_GRAPHS_ENABLED = os.environ.get("TILELANG_CUDA_GRAPHS", "1") not in ("0", "false", "no")


def _wrap_with_cuda_graph(base_runner, warmup_iters=3):
    """Wrap a runner with CUDA graph recording/replay.

    1. First *warmup_iters* calls execute normally (warmup + allocator priming).
    2. Next call: record the kernel sequence into a ``torch.cuda.CUDAGraph``
       using a private memory pool on the current stream.
    3. Subsequent calls: copy inputs into static buffers, replay, clone outputs.

    This eliminates per-call kernel launch overhead (~5-10 us each)
    which compounds across many subgraphs.
    """
    call_count = 0
    graph = None
    static_inputs = None
    static_outputs = None
    pool = torch.cuda.graph_pool_handle()

    def runner(*args):
        nonlocal call_count, graph, static_inputs, static_outputs

        call_count += 1

        # --- Warmup phase: execute normally to prime allocator ---
        if call_count <= warmup_iters:
            return base_runner(*args)

        # --- Record phase: capture CUDA graph ---
        if graph is None:
            static_inputs = [a.clone() for a in args if isinstance(a, torch.Tensor)]

            # Record on the current stream with a private pool so that
            # graph-internal allocations don't conflict with external ones.
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, pool=pool):
                static_outputs = base_runner(*static_inputs)

            if not isinstance(static_outputs, tuple):
                static_outputs = (static_outputs,)

        # --- Replay phase ---
        j = 0
        for a in args:
            if isinstance(a, torch.Tensor):
                static_inputs[j].copy_(a)
                j += 1

        graph.replay()
        return tuple(o.clone() for o in static_outputs)

    return runner


# ---------------------------------------------------------------------------
# FX-level diamond breaking (improves FuseOps fusion)
# ---------------------------------------------------------------------------

# Method calls that are cheap element-wise dtype conversions.
_CHEAP_METHODS = frozenset({"float", "half", "bfloat16", "to", "int", "double"})
# Function names for cheap unary/cast ops.
_CHEAP_FUNC_NAMES = frozenset({"neg", "abs", "_to_copy", "convert_element_type"})
# Binary element-wise ops safe to duplicate (small cost, breaks diamonds
# created by upstream duplication, e.g. residual add in RMSNorm).
_CHEAP_BINARY_FUNC_NAMES = frozenset({"add", "mul", "sub", "truediv"})


def _is_cheap_to_duplicate(node: torch.fx.Node) -> bool:
    """Return True if *node* is cheap enough to duplicate for fusion."""
    if node.op == "call_method" and node.target in _CHEAP_METHODS:
        return True
    if node.op == "call_function":
        name = getattr(node.target, "__name__", "")
        if name in _CHEAP_FUNC_NAMES:
            return True
        # Binary element-wise: only duplicate if inputs are from
        # placeholders (subgraph parameters) to avoid cascading.
        if name in _CHEAP_BINARY_FUNC_NAMES:
            return all(
                a.op == "placeholder"
                for a in node.args
                if isinstance(a, torch.fx.Node)
            )
    return False


def _break_fx_diamonds(gm: torch.fx.GraphModule) -> None:
    """Duplicate cheap element-wise FX nodes with multiple consumers.

    When a cheap op (dtype cast, unary, etc.) feeds multiple consumers,
    TVM's FuseOps creates a diamond dependency that prevents full fusion.
    Duplicating the op eliminates the diamond, allowing FuseOps to produce
    a single fused TIR function (e.g., RMSNorm: 1 kernel instead of 3).

    Runs iteratively because duplicating one node can create new
    multi-consumer patterns in its predecessors.
    """
    graph = gm.graph
    for _round in range(4):  # bound iterations
        changed = False
        for node in list(graph.nodes):
            if len(node.users) <= 1:
                continue
            if not _is_cheap_to_duplicate(node):
                continue

            users = list(node.users.keys())
            for user in users[1:]:
                with graph.inserting_before(user):
                    if node.op == "call_method":
                        dup = graph.call_method(node.target, node.args, node.kwargs)
                    else:
                        dup = graph.call_function(node.target, node.args, node.kwargs)
                    dup.meta = dict(node.meta)
                    user.replace_input_with(node, dup)
                    changed = True

        if not changed:
            break
        graph.lint()
        gm.recompile()


# ---------------------------------------------------------------------------
# Core compilation
# ---------------------------------------------------------------------------

def _compile_subgraph(
    gm: torch.fx.GraphModule,
    example_inputs: list[torch.Tensor],
    arch: str,
    options: dict[str, Any],
) -> Callable:
    """Compile a Dynamo-captured subgraph through TileLang's pipeline.

    Uses an in-memory cache so structurally identical subgraphs (e.g.
    the same RMSNorm across all 32 layers) are compiled only once.
    """
    from tvm import relax, tir
    from tvm.relax.frontend.torch import from_fx
    from tvm.runtime.vm import VirtualMachine

    from tilelang import tvm as _tvm
    from tilelang.jit.graph import (
        GraphCompileTrace,
        _build_extra_convert_map,
        _schedule_relax_module,
        compile_subgraph_direct,
        tilelang_relax_build,
    )

    # Build input_info for from_fx.
    sym_var_cache: dict[str, tir.Var] = {}

    def _dim_to_tir(s):
        if isinstance(s, torch.SymInt):
            name = str(s)
            if name not in sym_var_cache:
                sym_var_cache[name] = tir.Var(name, "int64")
            return sym_var_cache[name]
        return int(s)

    input_info: list[tuple[list[int | tir.Var], str]] = []
    for node in gm.graph.nodes:
        if node.op != "placeholder":
            continue
        ev = node.meta.get("example_value")
        if ev is None or not hasattr(ev, "shape") or not hasattr(ev, "dtype"):
            continue  # SymInt scalar placeholder — skip
        shape = [_dim_to_tir(s) for s in ev.shape]
        dtype_str = str(ev.dtype).replace("torch.", "")
        input_info.append((shape, dtype_str))

    # --- Check in-memory cache ---
    cache_key = _compute_subgraph_key(gm, input_info, arch)
    with _subgraph_cache_lock:
        cached = _subgraph_cache.get(cache_key)
    if cached is not None:
        logger.info("Cache hit for subgraph %s", cache_key)
        # Record a trace stub for the cache hit.
        trace = GraphCompileTrace(arch=arch, compilation_path="cache_hit")
        with _compilation_traces_lock:
            _compilation_traces.append(trace)
        return cached

    logger.info("Cache miss for subgraph %s — compiling", cache_key)

    # Break diamond dependencies to improve FuseOps fusion.
    _break_fx_diamonds(gm)

    # Build convert map for ops not in TVM's default from_fx map.
    extra_map = _build_extra_convert_map()

    # Convert FX graph → Relax IR.
    mod = from_fx(
        gm, input_info, unwrap_unit_return_tuple=True,
        custom_convert_map=extra_map,
    )

    # Schedule with TileLang rules.
    trace = GraphCompileTrace(arch=arch, compilation_path="dynamo_backend")
    mod, _ = _schedule_relax_module(mod, arch, trace=trace)

    # Identity subgraphs → fall back.
    main_func = mod["main"]
    if (
        not isinstance(main_func.body, relax.SeqExpr)
        or len(main_func.body.blocks) == 0
    ):
        return gm.forward

    target = _tvm.target.cuda(arch=arch)
    tvm_device = _tvm.cuda(torch.cuda.current_device())
    expected_dtypes = _extract_output_dtypes(gm)

    if options.get("print_trace"):
        logger.info(trace.summary())
    with _compilation_traces_lock:
        _compilation_traces.append(trace)

    # Try direct kernel execution (bypasses VM, ~10x less per-call overhead).
    direct_result = None
    try:
        direct_result = compile_subgraph_direct(mod, target)
    except Exception:
        logger.info("Direct path: compilation failed, will use VM", exc_info=True)

    # Whether this subgraph has only static (concrete int) shapes.
    is_static = not sym_var_cache

    if direct_result is not None:
        param_names, call_seq, output_names, rt_mod = direct_result
        runner = _make_direct_runner(
            param_names, call_seq, output_names,
            rt_mod, tvm_device, expected_dtypes,
        )
        logger.info("Using direct kernel runner (%d kernels)", len(call_seq))
    else:
        # Fall back to VM path.
        logger.info("Falling back to VM runner")
        pass_configs = options.get("pass_configs")
        if pass_configs:
            pass_ctx = _tvm.transform.PassContext(opt_level=3, config=pass_configs)
        else:
            pass_ctx = _tvm.transform.PassContext(opt_level=3)
        with pass_ctx:
            ex = tilelang_relax_build(mod, target)
        vm = VirtualMachine(ex, tvm_device)
        runner = _make_runner(vm["main"], expected_dtypes)

    # Wrap with CUDA graph for static-shape subgraphs to eliminate
    # kernel launch overhead on repeated calls.
    use_cuda_graphs = (
        _CUDA_GRAPHS_ENABLED
        and is_static
        and not options.get("disable_cuda_graphs", False)
    )
    if use_cuda_graphs:
        runner = _wrap_with_cuda_graph(runner)
        logger.info("CUDA graph enabled for subgraph %s", cache_key)

    # Store in cache.
    with _subgraph_cache_lock:
        _subgraph_cache[cache_key] = runner

    return runner


# -------------------------------------------------------------------
# Public backend function
# -------------------------------------------------------------------

def tilelang_backend(
    gm: torch.fx.GraphModule,
    example_inputs: list[torch.Tensor],
    *,
    options: dict[str, Any] | None = None,
) -> Callable:
    """TileLang backend for ``torch.compile``."""
    from tilelang.jit.graph import _resolve_arch

    options = options or {}
    arch = _resolve_arch(options.get("arch"))

    gm.graph.eliminate_dead_code()

    try:
        return _compile_subgraph(gm, example_inputs, arch, options)
    except Exception:
        logger.info(
            "TileLang backend: subgraph compilation failed, "
            "falling back to eager execution.",
            exc_info=True,
        )
        return gm.forward


# -------------------------------------------------------------------
# Trace inspection API
# -------------------------------------------------------------------

def get_compilation_traces():
    """Return all compilation traces from the current process."""
    with _compilation_traces_lock:
        return list(_compilation_traces)


def clear_compilation_traces():
    """Clear the compilation trace registry."""
    with _compilation_traces_lock:
        _compilation_traces.clear()


def clear_subgraph_cache():
    """Clear the in-memory subgraph compile cache."""
    with _subgraph_cache_lock:
        _subgraph_cache.clear()


# -------------------------------------------------------------------
# Backend registration
# -------------------------------------------------------------------

try:
    from torch._dynamo.backends.registry import register_backend as _register_backend
    _register_backend(name="tilelang")(tilelang_backend)
except ImportError:
    logger.warning(
        "torch._dynamo not available; 'tilelang' backend not registered."
    )
