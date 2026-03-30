"""Compilation orchestration for the TileLang torch.compile backend."""

from __future__ import annotations

import copy
import hashlib
import logging
import os
import threading
from dataclasses import dataclass, field
from typing import Any, Callable

import torch

from .analysis import (
    GraphCompileTrace,
    _resolve_arch,
    _schedule_relax_module,
    compile_subgraph_direct,
    from_fx_with_fallback,
)
from .codegen import WrapperCodeGen
from .runtime import CompiledGraphModule

logger = logging.getLogger(__name__)

_HAS_CUDAGRAPH_TREES = False
try:
    from torch._inductor.cudagraph_trees import cudagraphify_impl as _cudagraphify_impl

    _HAS_CUDAGRAPH_TREES = True
except ImportError:
    _cudagraphify_impl = None


def _cuda_graphs_enabled() -> bool:
    return os.environ.get("TILELANG_CUDA_GRAPHS", "0") not in ("0", "false", "no")


@dataclass(frozen=True)
class CompileOptions:
    print_trace: bool = False
    disable_cuda_graphs: bool = False
    pass_configs: dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def from_dict(options: dict[str, Any] | None) -> "CompileOptions":
        options = dict(options or {})
        return CompileOptions(
            print_trace=bool(options.get("print_trace", False)),
            disable_cuda_graphs=bool(options.get("disable_cuda_graphs", False)),
            pass_configs=dict(options.get("pass_configs") or {}),
        )


def _compute_subgraph_key(gm: torch.fx.GraphModule, input_info: list, arch: str) -> str:
    parts = [arch]
    for node in gm.graph.nodes:
        if node.op == "placeholder":
            parts.append("placeholder")
        elif node.op == "output":
            parts.append("output")
        elif node.op == "call_function":
            parts.append(f"call:{node.target.__name__}")
            for arg in node.args:
                if not isinstance(arg, torch.fx.Node):
                    parts.append(repr(arg))
            for key, value in sorted(node.kwargs.items()):
                if not isinstance(value, torch.fx.Node):
                    parts.append(f"{key}={value!r}")
        elif node.op == "call_method":
            parts.append(f"method:{node.target}")
        else:
            parts.append(f"{node.op}:{node.target}")
    for shape, dtype in input_info:
        parts.append(f"{'x'.join(str(dim) for dim in shape)}:{dtype}")
    key_str = "|".join(parts)
    return hashlib.sha256(key_str.encode()).hexdigest()[:16]


def _extract_output_dtypes(gm: torch.fx.GraphModule) -> list[torch.dtype]:
    output_nodes = [node for node in gm.graph.nodes if node.op == "output"]
    if not output_nodes:
        return []

    output_node = output_nodes[0]
    ret = output_node.args[0]

    def _dtype_of(node_or_val):
        if isinstance(node_or_val, torch.fx.Node):
            for key in ("val", "example_value"):
                value = node_or_val.meta.get(key)
                if value is not None and hasattr(value, "dtype"):
                    return value.dtype
        logger.debug("Could not determine output dtype for %s, defaulting to float32", node_or_val)
        return torch.float32

    if isinstance(ret, (tuple, list)):
        return [_dtype_of(value) for value in ret]
    return [_dtype_of(ret)]


def _build_compiled_module(
    param_names: list[str],
    call_seq: list,
    output_names: list[str],
    expected_dtypes: list[torch.dtype],
    device_index: int,
    rt_mod: Any,
    cache_key: str,
    *,
    sym_var_map: dict[str, tuple[int, int]] | None = None,
    constants: dict[str, torch.Tensor] | None = None,
) -> CompiledGraphModule:
    sym_shape_map: dict[str, str] = {}
    if sym_var_map:
        for var_name, (param_index, dim_index) in sym_var_map.items():
            sym_shape_map[var_name] = f"inp_{param_index}.shape[{dim_index}]"

    extern_ops: dict[str, object] = {}
    for record in call_seq:
        if record.extern_op is not None:
            extern_ops[record.func_name] = record.extern_op.target

    codegen = WrapperCodeGen(
        param_names=param_names,
        call_seq=call_seq,
        output_names=output_names,
        expected_dtypes=expected_dtypes,
        device_index=device_index,
        sym_shape_map=sym_shape_map,
        extern_ops=extern_ops,
        constants=constants or {},
    )
    return CompiledGraphModule.from_codegen(codegen, rt_mod, cache_key)


def _wrap_with_cuda_graph_trees(
    base_runner: Callable,
    example_inputs: list[torch.Tensor],
    device_index: int,
) -> Callable:
    probe_out = base_runner(*example_inputs)
    n_outputs = len(probe_out) if isinstance(probe_out, (tuple, list)) else 1
    stack_traces: list[str | None] = [None] * n_outputs

    def _list_runner(inputs):
        args = list(inputs)
        inputs.clear()
        return list(base_runner(*args))

    wrapped = _cudagraphify_impl(
        _list_runner,
        list(example_inputs),
        static_input_idxs=(),
        device_index=device_index,
        is_backward=False,
        is_inference=True,
        stack_traces=stack_traces,
        constants=(),
        placeholders=(),
        mutated_input_idxs=(),
    )

    def runner(*args):
        return tuple(wrapped(list(args)))

    return runner


def _wrap_with_cuda_graph_simple(base_runner: Callable, warmup_iters: int = 3) -> Callable:
    from tvm_ffi import use_torch_stream

    call_count = 0
    graph = None
    static_inputs = None
    static_outputs = None
    pool = torch.cuda.graph_pool_handle()

    def runner(*args):
        nonlocal call_count, graph, static_inputs, static_outputs

        call_count += 1
        if call_count <= warmup_iters:
            with use_torch_stream():
                return base_runner(*args)

        if graph is None:
            static_inputs = [arg.clone() for arg in args if isinstance(arg, torch.Tensor)]
            graph = torch.cuda.CUDAGraph()
            with use_torch_stream(torch.cuda.graph(graph, pool=pool)):
                static_outputs = base_runner(*static_inputs)
            if not isinstance(static_outputs, tuple):
                static_outputs = (static_outputs,)

        tensor_index = 0
        for arg in args:
            if isinstance(arg, torch.Tensor):
                static_inputs[tensor_index].copy_(arg)
                tensor_index += 1

        graph.replay()
        return tuple(output.clone() for output in static_outputs)

    return runner


def _wrap_with_cuda_graph(
    base_runner: Callable,
    *,
    example_inputs: list[torch.Tensor],
    device_index: int,
) -> Callable:
    if _HAS_CUDAGRAPH_TREES:
        try:
            return _wrap_with_cuda_graph_trees(base_runner, example_inputs, device_index)
        except Exception:
            logger.debug("cudagraph_trees wrapping failed, falling back to simple wrapper", exc_info=True)
    return _wrap_with_cuda_graph_simple(base_runner)


def _fold_scalar_inputs(
    gm: torch.fx.GraphModule,
    concrete_args: tuple[Any, ...],
) -> torch.fx.GraphModule:
    gm = copy.deepcopy(gm)
    graph = gm.graph

    folded: dict[torch.fx.Node, float | int] = {}
    arg_index = 0
    for node in list(graph.nodes):
        if node.op != "placeholder":
            continue
        if arg_index >= len(concrete_args):
            break
        value = concrete_args[arg_index]
        arg_index += 1
        if isinstance(value, torch.Tensor) and value.ndim == 0:
            folded[node] = value.item()

    if not folded:
        return gm

    changed = True
    while changed:
        changed = False
        for node in list(graph.nodes):
            if node in folded:
                continue
            if node.op == "call_method" and node.target in ("item", "float", "int"):
                src = node.args[0]
                if src in folded:
                    value = folded[src]
                    folded[node] = int(value) if node.target == "int" else float(value)
                    changed = True
            elif node.op == "call_function":
                name = getattr(node.target, "__name__", "")
                if name in ("float", "int", "item"):
                    src = node.args[0]
                    if src in folded:
                        folded[node] = folded[src]
                        changed = True

    for node in reversed(list(graph.nodes)):
        if node not in folded:
            continue
        scalar = folded[node]
        for user in list(node.users.keys()):
            user.args = tuple(
                scalar if isinstance(arg, torch.fx.Node) and arg is node else arg
                for arg in user.args
            )
            user.kwargs = {
                key: (scalar if isinstance(value, torch.fx.Node) and value is node else value)
                for key, value in user.kwargs.items()
            }
        graph.erase_node(node)

    graph.lint()
    gm.recompile()
    return gm


_CHEAP_METHODS = frozenset({"float", "half", "bfloat16", "to", "int", "double"})
_CHEAP_FUNC_NAMES = frozenset({"neg", "abs", "_to_copy", "convert_element_type"})
_CHEAP_BINARY_FUNC_NAMES = frozenset({"add", "mul", "sub", "truediv"})


def _is_cheap_to_duplicate(node: torch.fx.Node) -> bool:
    if node.op == "call_method" and node.target in _CHEAP_METHODS:
        return True
    if node.op == "call_function":
        name = getattr(node.target, "__name__", "")
        if name in _CHEAP_FUNC_NAMES:
            return True
        if name in _CHEAP_BINARY_FUNC_NAMES:
            return all(
                arg.op == "placeholder"
                for arg in node.args
                if isinstance(arg, torch.fx.Node)
            )
    return False


def _break_fx_diamonds(gm: torch.fx.GraphModule) -> None:
    graph = gm.graph
    for _round in range(4):
        changed = False
        for node in list(graph.nodes):
            if len(node.users) <= 1 or not _is_cheap_to_duplicate(node):
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


class SubgraphCompiler:
    """Compile one Dynamo-produced FX subgraph through TileLang."""

    def __init__(
        self,
        state,
        gm: torch.fx.GraphModule,
        example_inputs: list[Any],
        arch: str,
        options: CompileOptions,
    ) -> None:
        self.state = state
        self.gm = gm
        self.example_inputs = example_inputs
        self.arch = arch
        self.options = options

    def compile(self) -> Callable:
        input_info, is_dynamic = self._build_input_info()
        cache_key = _compute_subgraph_key(self.gm, input_info, self.arch)

        cached = self.state.cache_get(cache_key)
        if cached is not None:
            self.state.add_trace(
                GraphCompileTrace(
                    arch=self.arch,
                    dynamic=is_dynamic,
                    compilation_path="cache_hit",
                )
            )
            return cached

        if self.state.disk_cache_enabled() and not is_dynamic:
            disk_hit = CompiledGraphModule.from_disk(self.state.cache_dir(), cache_key)
            if disk_hit is not None:
                self.state.add_trace(
                    GraphCompileTrace(
                        arch=self.arch,
                        dynamic=False,
                        compilation_path="disk_cache_hit",
                    )
                )
                self.state.cache_put(cache_key, disk_hit)
                return disk_hit

        try:
            _break_fx_diamonds(self.gm)
            expected_dtypes = _extract_output_dtypes(self.gm)
            if is_dynamic:
                runner = self._compile_dynamic(cache_key, input_info, expected_dtypes)
            else:
                runner = self._compile_static(cache_key, input_info, expected_dtypes)

            if (
                _cuda_graphs_enabled()
                and not is_dynamic
                and not self.options.disable_cuda_graphs
                and any(isinstance(arg, torch.Tensor) for arg in self.example_inputs)
                and runner is not self.gm.forward
            ):
                tensor_inputs = [arg for arg in self.example_inputs if isinstance(arg, torch.Tensor)]
                device_index = tensor_inputs[0].device.index if tensor_inputs else 0
                runner = _wrap_with_cuda_graph(
                    runner,
                    example_inputs=tensor_inputs,
                    device_index=device_index,
                )
                logger.info("CUDA graph enabled for subgraph %s", cache_key)

            self.state.cache_put(cache_key, runner)
            return runner
        except Exception:
            logger.info(
                "TileLang backend: subgraph compilation failed for %s, falling back to eager",
                cache_key,
                exc_info=True,
            )
            self.state.add_trace(
                GraphCompileTrace(
                    arch=self.arch,
                    dynamic=is_dynamic,
                    compilation_path="fallback_eager",
                )
            )
            self.state.cache_put(cache_key, self.gm.forward)
            return self.gm.forward

    def _build_input_info(self) -> tuple[list[tuple[list[Any], str]], bool]:
        from tvm import tir

        sym_var_cache: dict[str, tir.Var] = {}

        def _dim_to_tir(dim):
            if isinstance(dim, torch.SymInt):
                name = str(dim)
                if name not in sym_var_cache:
                    sym_var_cache[name] = tir.Var(name, "int64")
                return sym_var_cache[name]
            return int(dim)

        input_info: list[tuple[list[Any], str]] = []
        for node in self.gm.graph.nodes:
            if node.op != "placeholder":
                continue
            example_value = node.meta.get("example_value")
            if example_value is None or not hasattr(example_value, "shape") or not hasattr(example_value, "dtype"):
                continue
            shape = [_dim_to_tir(dim) for dim in example_value.shape]
            dtype_str = str(example_value.dtype).replace("torch.", "")
            input_info.append((shape, dtype_str))
        return input_info, bool(sym_var_cache)

    def _compile_static(
        self,
        cache_key: str,
        input_info: list[tuple[list[Any], str]],
        expected_dtypes: list[torch.dtype],
    ) -> Callable:
        from tvm import relax
        from tilelang import tvm as tilelang_tvm

        trace = GraphCompileTrace(
            arch=self.arch,
            dynamic=False,
            compilation_path="dynamo_backend",
        )
        lowering = from_fx_with_fallback(self.gm, input_info)
        mod, _ = _schedule_relax_module(
            lowering.mod,
            self.arch,
            trace=trace,
            pass_configs=self.options.pass_configs,
        )

        main_func = mod["main"]
        if not isinstance(main_func.body, relax.SeqExpr) or len(main_func.body.blocks) == 0:
            trace.compilation_path = "identity_fallback"
            self.state.add_trace(trace)
            return self.gm.forward

        target = tilelang_tvm.target.cuda(arch=self.arch)
        tvm_device = tilelang_tvm.cuda(torch.cuda.current_device())
        if self.options.print_trace:
            logger.info(trace.summary())

        save_so_path = None
        if self.state.disk_cache_enabled():
            save_so_path = os.path.join(self.state.cache_dir(), f"{cache_key}.so")

        direct_result = compile_subgraph_direct(
            mod,
            target,
            extern_ops=lowering.extern_ops,
            save_so_path=save_so_path,
            pass_configs=self.options.pass_configs,
        )
        if direct_result is None:
            trace.compilation_path = "fallback_eager"
            self.state.add_trace(trace)
            return self.gm.forward

        param_names, call_seq, output_names, sym_var_map, rt_mod, constants = direct_result
        compiled = _build_compiled_module(
            param_names,
            call_seq,
            output_names,
            expected_dtypes,
            tvm_device.index,
            rt_mod,
            cache_key,
            sym_var_map=sym_var_map,
            constants=constants,
        )
        if self.state.disk_cache_enabled():
            compiled.save_to_disk(self.state.cache_dir())

        self.state.add_trace(trace)
        logger.info("Using codegen runner (%d kernels, key=%s)", len(call_seq), cache_key[:8])
        return compiled

    def _compile_dynamic(
        self,
        cache_key: str,
        input_info: list[tuple[list[Any], str]],
        expected_dtypes: list[torch.dtype],
    ) -> Callable:
        trace = GraphCompileTrace(
            arch=self.arch,
            dynamic=True,
            compilation_path="dynamo_symbolic",
        )
        runner = self._try_symbolic_compilation(cache_key, input_info, expected_dtypes)
        if runner is not None:
            self.state.add_trace(trace)
            logger.info("Using symbolic runner (key=%s)", cache_key[:8])
            return runner

        trace.compilation_path = "dynamo_dynamic_jit"
        first_runner = self._compile_for_concrete_shapes(tuple(self.example_inputs), expected_dtypes)
        if first_runner is None:
            trace.compilation_path = "fallback_eager"
            self.state.add_trace(trace)
            return self.gm.forward

        first_shape_key = tuple(
            (tuple(arg.shape), str(arg.dtype))
            for arg in self.example_inputs
            if isinstance(arg, torch.Tensor) and arg.ndim > 0
        )
        runner = self._make_dynamic_direct_runner(expected_dtypes, first_runner, first_shape_key)
        self.state.add_trace(trace)
        logger.info("Using dynamic per-shape JIT runner")
        return runner

    def _compile_for_concrete_shapes(
        self,
        concrete_args: tuple[Any, ...],
        expected_dtypes: list[torch.dtype],
    ) -> Callable | None:
        from tvm import relax
        from tilelang import tvm as tilelang_tvm

        gm = _fold_scalar_inputs(self.gm, concrete_args)
        input_info: list[tuple[list[int], str]] = []
        for arg in concrete_args:
            if isinstance(arg, torch.Tensor) and arg.ndim > 0:
                input_info.append(
                    ([int(dim) for dim in arg.shape], str(arg.dtype).replace("torch.", ""))
                )

        try:
            lowering = from_fx_with_fallback(gm, input_info)
        except Exception:
            logger.debug("Dynamic compile: from_fx failed", exc_info=True)
            return None

        try:
            mod, _ = _schedule_relax_module(
                lowering.mod,
                self.arch,
                pass_configs=self.options.pass_configs,
            )
        except Exception:
            logger.debug("Dynamic compile: scheduling failed", exc_info=True)
            return None

        main_func = mod["main"]
        if not isinstance(main_func.body, relax.SeqExpr) or len(main_func.body.blocks) == 0:
            return None

        target = tilelang_tvm.target.cuda(arch=self.arch)
        tvm_device = tilelang_tvm.cuda(torch.cuda.current_device())
        try:
            direct_result = compile_subgraph_direct(
                mod,
                target,
                extern_ops=lowering.extern_ops,
                pass_configs=self.options.pass_configs,
            )
        except Exception:
            logger.debug("Dynamic compile: TIR compilation failed", exc_info=True)
            return None

        if direct_result is None:
            return None

        param_names, call_seq, output_names, sym_var_map, rt_mod, constants = direct_result
        first_tensor = next(
            (arg for arg in concrete_args if isinstance(arg, torch.Tensor) and arg.ndim > 0),
            None,
        )
        shape_tag = "x".join(str(dim) for dim in first_tensor.shape) if first_tensor is not None else "scalar"
        return _build_compiled_module(
            param_names,
            call_seq,
            output_names,
            expected_dtypes,
            tvm_device.index,
            rt_mod,
            f"dyn_{shape_tag}",
            sym_var_map=sym_var_map,
            constants=constants,
        )

    def _make_dynamic_direct_runner(
        self,
        expected_dtypes: list[torch.dtype],
        first_runner: Callable,
        first_shape_key: tuple,
    ) -> Callable:
        lock = threading.Lock()
        shape_cache: dict[tuple, Callable] = {first_shape_key: first_runner}

        def runner(*args):
            tensor_args = tuple(
                arg for arg in args if isinstance(arg, torch.Tensor) and arg.ndim > 0
            )
            shape_key = tuple((tuple(arg.shape), str(arg.dtype)) for arg in tensor_args)
            with lock:
                cached = shape_cache.get(shape_key)
            if cached is not None:
                return cached(*tensor_args)

            logger.info("Dynamic runner: compiling for new shape %s", shape_key)
            concrete = self._compile_for_concrete_shapes(args, expected_dtypes)
            if concrete is None:
                raise RuntimeError(
                    f"TileLang dynamic runner failed to compile for shapes {shape_key}"
                )

            with lock:
                shape_cache[shape_key] = concrete
            return concrete(*tensor_args)

        return runner

    def _try_symbolic_compilation(
        self,
        cache_key: str,
        input_info: list[tuple[list[Any], str]],
        expected_dtypes: list[torch.dtype],
    ) -> Callable | None:
        from tvm import relax
        from tilelang import tvm as tilelang_tvm

        gm = _fold_scalar_inputs(self.gm, tuple(self.example_inputs))
        _break_fx_diamonds(gm)
        input_info_sym = [(shape, dtype) for shape, dtype in input_info if len(shape) > 0]

        try:
            lowering = from_fx_with_fallback(gm, input_info_sym)
        except Exception as exc:
            logger.debug("Symbolic compile: from_fx failed: %s", exc)
            return None

        try:
            mod, _ = _schedule_relax_module(
                lowering.mod,
                self.arch,
                pass_configs=self.options.pass_configs,
            )
        except Exception as exc:
            logger.debug("Symbolic compile: scheduling failed: %s", exc)
            return None

        main_func = mod["main"]
        if not isinstance(main_func.body, relax.SeqExpr) or len(main_func.body.blocks) == 0:
            logger.debug("Symbolic compile: identity subgraph (no bindings)")
            return None

        target = tilelang_tvm.target.cuda(arch=self.arch)
        tvm_device = tilelang_tvm.cuda(torch.cuda.current_device())
        try:
            direct_result = compile_subgraph_direct(
                mod,
                target,
                extern_ops=lowering.extern_ops,
                pass_configs=self.options.pass_configs,
            )
        except Exception as exc:
            logger.debug("Symbolic compile: TIR compilation failed: %s", exc)
            return None

        if direct_result is None:
            logger.debug("Symbolic compile: compile_subgraph_direct returned None")
            return None

        param_names, call_seq, output_names, sym_var_map, rt_mod, constants = direct_result
        compiled = _build_compiled_module(
            param_names,
            call_seq,
            output_names,
            expected_dtypes,
            tvm_device.index,
            rt_mod,
            cache_key,
            sym_var_map=sym_var_map,
            constants=constants,
        )
        logger.info(
            "Symbolic compilation succeeded (%d kernels, %d sym vars, key=%s)",
            len(call_seq),
            len(sym_var_map),
            cache_key[:8],
        )

        def runner(*args):
            tensor_args = tuple(
                arg for arg in args if isinstance(arg, torch.Tensor) and arg.ndim > 0
            )
            return compiled(*tensor_args)

        return runner


def compile_subgraph(
    state,
    gm: torch.fx.GraphModule,
    example_inputs: list[Any],
    *,
    options: dict[str, Any] | None = None,
) -> Callable:
    arch = _resolve_arch((options or {}).get("arch"))
    compile_options = CompileOptions.from_dict(options)
    return SubgraphCompiler(state, gm, example_inputs, arch, compile_options).compile()
