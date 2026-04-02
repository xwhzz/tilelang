"""Convert PyTorch FX graph to Relax IR with automatic dtype cast insertion."""

import logging

import torch
from torch import fx

from tilelang import tvm as tvm
from tvm import relax
from tvm.relax.frontend.torch.fx_translator import TorchFXImporter

from tilelang.graph.utils import torch_dtype_to_tvm

logger = logging.getLogger(__name__)


def extract_input_info(gm, example_inputs):
    """Extract (shape, dtype_str) pairs using FX graph metadata.

    Uses node.meta["val"] from placeholder nodes which preserves SymInt
    for dynamic dimensions. Falls back to example_inputs for shapes.
    """
    from tvm import tir

    shape_vars: dict[str, tir.Var] = {}
    input_info = []

    # Collect placeholder metadata from FX graph
    placeholders = []
    for node in gm.graph.nodes:
        if node.op == "placeholder":
            val = node.meta.get("val", node.meta.get("example_value"))
            if val is not None and isinstance(val, torch.Tensor):
                placeholders.append(val)

    # Use FX metadata (has SymInt) if available, else example_inputs
    sources = placeholders if placeholders else [
        inp for inp in example_inputs if isinstance(inp, torch.Tensor)]

    for tensor in sources:
        shape = []
        for s in tensor.shape:
            if isinstance(s, torch.SymInt):
                key = str(s)
                if key not in shape_vars:
                    shape_vars[key] = tir.Var(key, "int64")
                shape.append(shape_vars[key])
            else:
                shape.append(int(s))
        dtype_str = torch_dtype_to_tvm(tensor.dtype)
        input_info.append((tuple(shape), dtype_str))
    return input_info


def _struct_info_from_fake(val) -> relax.TensorStructInfo:
    """Build Relax TensorStructInfo from a FakeTensor."""
    shape = [int(s) for s in val.shape]
    dtype = torch_dtype_to_tvm(val.dtype)
    return relax.TensorStructInfo(shape, dtype)


class TileLangFXImporter(TorchFXImporter):
    """FX→Relax importer with dtype cast insertion and per-op fallback.

    - After each conversion, inserts R.astype if output dtype mismatches
      the FX expectation (fused as epilogue by FuseOps/FuseTIR).
    - Unsupported ops emit opaque ``R.call_dps_packed`` calls that act as
      fusion barriers; the codegen layer maps them back to torch ops.
    """

    def _get_expected_dtype(self, node: fx.Node):
        val = node.meta.get("val", node.meta.get("example_value"))
        if val is not None and isinstance(val, torch.Tensor):
            return torch_dtype_to_tvm(val.dtype)
        return None

    def _maybe_cast(self, node: fx.Node, expr: relax.Expr) -> relax.Expr:
        """Insert a cast if the Relax expr dtype differs from FX expectation."""
        expected = self._get_expected_dtype(node)
        if expected is None:
            return expr
        sinfo = expr.struct_info_ if hasattr(expr, "struct_info_") else None
        if sinfo is None or not isinstance(sinfo, relax.TensorStructInfo):
            return expr
        actual = sinfo.dtype
        if actual and expected and actual != expected:
            logger.debug("Inserting cast %s -> %s for %s", actual, expected, node.name)
            return self.block_builder.emit(relax.op.astype(expr, expected))
        return expr

    def _cat(self, node: fx.Node) -> relax.Expr:
        """Handle torch.cat, filtering out empty-tensor inputs (KV cache prefill)."""
        args = self.retrieve_args(node)
        if isinstance(args[0], (list, tuple)):
            tensors = list(args[0])
        else:
            tensors = [args[0]]

        # Filter out 0-element tensors (e.g. empty KV cache during prefill).
        # Only catches statically-known zero dims (IntImm); symbolic zeros
        # are not filtered and will go through the normal concat path.
        non_empty = []
        for t in tensors:
            sinfo = t.struct_info_ if hasattr(t, "struct_info_") else None
            if sinfo and isinstance(sinfo, relax.TensorStructInfo):
                shape = sinfo.shape
                if shape is not None and any(
                    isinstance(d, tvm.tir.IntImm) and int(d) == 0
                    for d in shape
                ):
                    continue
            non_empty.append(t)

        if len(non_empty) == 0:
            return tensors[0]
        if len(non_empty) == 1:
            return non_empty[0]

        axis = node.kwargs.get("dim", 0)
        return self.block_builder.emit(relax.op.concat(non_empty, axis=axis))

    def __init__(self, *args, extern_dispatch=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.fallback_calls: dict[str, tuple] = {}
        self.extern_dispatch = extern_dispatch or default_extern_dispatch

    def _convert_or_fallback(self, node: fx.Node, key):
        """Try the converter for key; extern-dispatch ops fallback to torch."""
        if self.extern_dispatch(node):
            return self._emit_torch_fallback(node)
        if key in self.convert_map:
            try:
                result = self.convert_map[key](node)
                return self._maybe_cast(node, result)
            except Exception as e:
                logger.debug("Converter failed for %s, falling back to torch: %s", key, e)
        return self._emit_torch_fallback(node)

    def _emit_torch_fallback(self, node: fx.Node) -> relax.Expr:
        """Emit an opaque packed call for an unsupported op.

        The call acts as a fusion barrier in FuseOps. At runtime the
        codegen layer dispatches it to the original torch op.
        """
        op_name = (node.target.__name__ if node.op == "call_function"
                   else node.target)
        logger.info("Unsupported op '%s', emitting torch fallback", op_name)

        # Build arg template: track which args are tensors (looked up at
        # runtime from env) vs scalars (stored as-is).
        # For list/tuple args containing tensors (e.g. torch.cat([t1, t2])),
        # store as ("list", [var_name1, var_name2, ...]).
        raw_args = self.retrieve_args(node)
        arg_template = []
        flat_tensor_args = []
        for a in (raw_args if isinstance(raw_args, (list, tuple)) else [raw_args]):
            if isinstance(a, relax.Expr):
                var_name = a.name_hint if hasattr(a, "name_hint") else None
                arg_template.append((True, var_name))
                flat_tensor_args.append(a)
            elif isinstance(a, (list, tuple)) and any(isinstance(e, relax.Expr) for e in a):
                # List/tuple of tensors (e.g., cat inputs)
                names = []
                for e in a:
                    if isinstance(e, relax.Expr):
                        var_name = e.name_hint if hasattr(e, "name_hint") else None
                        names.append(var_name)
                        flat_tensor_args.append(e)
                    else:
                        names.append(e)
                arg_template.append(("list", names))
            else:
                arg_template.append((False, a))

        # Resolve kwargs: FX Node references → Relax var names for tensors
        kwargs = {}
        for k, v in node.kwargs.items():
            if isinstance(v, fx.Node) and v in self.env:
                expr = self.env[v]
                if isinstance(expr, relax.Expr) and hasattr(expr, "name_hint"):
                    kwargs[k] = ("__tensor__", expr.name_hint)
                    flat_tensor_args.append(expr)
                else:
                    kwargs[k] = v
            else:
                kwargs[k] = v

        # Build output struct info from FakeTensor metadata
        val = node.meta.get("val", node.meta.get("example_value"))
        if val is not None and isinstance(val, torch.Tensor):
            out_sinfo = _struct_info_from_fake(val)
        else:
            out_sinfo = relax.ObjectStructInfo()

        # Emit an opaque call (acts as fusion barrier).
        # Use call_dps_packed only for tensor outputs; scalar outputs
        # (ObjectStructInfo) use a regular Call.
        extern = relax.ExternFunc(f"torch_fallback.{op_name}")
        is_tensor_output = isinstance(out_sinfo, relax.TensorStructInfo)
        if flat_tensor_args and is_tensor_output:
            call = relax.call_dps_packed(extern, flat_tensor_args, out_sinfo)
        else:
            call = relax.Call(extern, flat_tensor_args, sinfo_args=[out_sinfo])
        result = self.block_builder.emit(call)

        # Store full call info keyed by op_name (stable across pipeline transforms)
        # Optimize SDPA: if using a causal bool mask, switch to is_causal=True
        # to avoid mask-conversion overhead (saves ~0.7ms on LLaMA).
        if op_name == "scaled_dot_product_attention" and self._is_causal_sdpa(node):
            # Remove the mask tensor from args and kwargs
            flat_tensor_args_no_mask = flat_tensor_args[:3]  # Q, K, V only
            kwargs = dict(kwargs)
            kwargs.pop("attn_mask", None)
            kwargs["is_causal"] = True
            # Rebuild the Relax call without the mask tensor
            extern = relax.ExternFunc(f"torch_fallback.{op_name}")
            call = relax.call_dps_packed(extern, flat_tensor_args_no_mask, out_sinfo)
            result = self.block_builder.emit(call)
            arg_template = arg_template[:3]  # Q, K, V only

        self.fallback_calls[op_name] = (node.target, arg_template, kwargs)

        return result

    @staticmethod
    def _is_causal_sdpa(node: fx.Node) -> bool:
        """Check if an SDPA node uses a standard causal (lower-triangular) mask."""
        mask_node = node.kwargs.get("attn_mask")
        if mask_node is None:
            return False
        val = mask_node.meta.get("val", mask_node.meta.get("example_value"))
        if val is None or not isinstance(val, torch.Tensor):
            return False
        # Must be bool dtype, square in last two dims (causal mask shape)
        if val.dtype != torch.bool:
            return False
        if val.ndim < 2 or val.shape[-1] != val.shape[-2]:
            return False
        return True

    def from_fx(
        self,
        model,
        input_info: list[tuple[tuple[int], str]],
        keep_params_as_input: bool,
        unwrap_unit_return_tuple: bool,
        no_bind_return_tuple: bool,
        custom_convert_map: dict = None,
    ) -> tvm.IRModule:
        if custom_convert_map:
            custom_ops = set(custom_convert_map.keys())
            self.update_convert_map(custom_convert_map)
        else:
            custom_ops = set()
        self.named_modules = dict(model.named_modules())

        graph: fx.Graph = model.graph
        inputs = self.create_input_vars(input_info)

        func_name = "main"
        self.block_builder = relax.BlockBuilder()
        params = []
        if keep_params_as_input:
            func_attrs = {"num_input": len(inputs)}
            for name, param in sorted(model.named_parameters(), key=lambda x: x[0]):
                shape = param.data.shape
                dtype = self._convert_data_type(str(param.data.dtype))
                inputs.append(relax.Var(name, relax.TensorStructInfo(shape, dtype)))
                self.params[param] = inputs[-1]
                params.append(tvm.runtime.tensor(param.data.cpu().numpy()))
        else:
            func_attrs = None

        with self.block_builder.function(name=func_name, params=inputs.copy(), attrs=func_attrs):
            output = None
            with self.block_builder.dataflow():
                for _, param in model.named_parameters():
                    shape = param.data.shape
                    dtype = self._convert_data_type(str(param.data.dtype))
                    if dtype in ("float32", "float16", "bfloat16"):
                        if not keep_params_as_input:
                            self.params[param] = self._convert_torch_tensor_to_relax(param)
                    else:
                        raise ValueError(f"Unsupported parameter dtype: {dtype}")

                for node in graph.nodes:
                    if node.op == "placeholder":
                        if "grapharg" in node.meta and node.meta["grapharg"].fake_tensor is None:
                            continue
                        ev = node.meta.get("example_value")
                        if ev is not None and not hasattr(ev, "shape"):
                            continue
                        assert len(inputs) > 0
                        self.env[node] = inputs.pop(0)

                    elif node.op == "output":
                        args = self.retrieve_args(node)
                        assert len(args) == 1
                        if isinstance(args[0], (tuple, list, relax.Tuple)):
                            if unwrap_unit_return_tuple and len(args[0]) == 1:
                                output = self.block_builder.emit_output(args[0][0])
                            elif no_bind_return_tuple:
                                output = [self.block_builder.emit_output(r) for r in args[0]]
                        if output is None:
                            output = self.block_builder.emit_output(args[0])
                        break

                    elif node.op == "get_attr":
                        self.env[node] = self._fetch_attr(model, node.target)

                    elif node.op == "call_module":
                        module = self.named_modules[node.target]
                        key = type(module)
                        self.env[node] = self._convert_or_fallback(node, key)

                    elif node.op == "call_function":
                        fn = node.target.__name__
                        if fn in custom_ops:
                            result = self.convert_map[fn](node, self)
                            self.env[node] = self._maybe_cast(node, result)
                        else:
                            self.env[node] = self._convert_or_fallback(node, fn)

                    elif node.op == "call_method":
                        self.env[node] = self._convert_or_fallback(node, node.target)

                    else:
                        raise ValueError(f"Unsupported op {node.op}")

            assert output is not None
            self.block_builder.emit_func_output(output)

        mod = self.block_builder.get()
        if keep_params_as_input:
            mod["main"] = mod["main"].with_attr("params", params)
        return mod


def fx_to_relax(
    gm: fx.GraphModule,
    example_inputs: list[torch.Tensor],
    *,
    extern_dispatch: "Callable[[fx.Node], bool] | None" = None,
) -> tuple[tvm.IRModule, dict[str, tuple]]:
    """Convert an FX GraphModule to a Relax IRModule.

    Supported ops → Relax ops (compiled by TileLang).
    Unsupported ops → opaque calls (fusion barriers, run as torch ops).

    Parameters
    ----------
    extern_dispatch : (fx.Node) -> bool, optional
        Predicate that decides whether an FX node should be dispatched to
        torch instead of compiled by TileLang. Receives the full FX node
        so users can inspect op name, shapes, dtypes, etc.
        ``None`` uses :func:`default_extern_dispatch`.

    Returns (module, fallback_calls) where fallback_calls maps op names
    to (callable, arg_template, kwargs) for runtime dispatch.
    """
    input_info = extract_input_info(gm, example_inputs)
    importer = TileLangFXImporter(extern_dispatch=extern_dispatch)
    mod = importer.from_fx(
        gm, input_info,
        keep_params_as_input=False,
        unwrap_unit_return_tuple=False,
        no_bind_return_tuple=False,
    )
    return mod, importer.fallback_calls


_supported_ops_cache = None

def _get_supported_ops() -> dict:
    """Lazily build and cache the convert_map keys to avoid instantiating
    TileLangFXImporter on every disk-cache hit."""
    global _supported_ops_cache
    if _supported_ops_cache is None:
        _supported_ops_cache = TileLangFXImporter().convert_map
    return _supported_ops_cache


def _get_matmul_mnk(node: fx.Node) -> tuple[int, int, int] | None:
    """Extract (M, N, K) from a matmul-family FX node, or None if not a matmul."""
    fn_name = node.target.__name__ if node.op == "call_function" else ""
    if fn_name not in ("linear", "mm", "addmm", "bmm", "matmul"):
        return None

    arg_vals = []
    for a in node.args:
        if isinstance(a, fx.Node):
            av = a.meta.get("val", a.meta.get("example_value"))
            if av is not None and hasattr(av, "shape"):
                arg_vals.append(av)

    if len(arg_vals) < 2:
        return None

    try:
        a_shape = list(arg_vals[0].shape)
        b_shape = list(arg_vals[1].shape)

        if fn_name == "linear":
            M = int(a_shape[-2]) if len(a_shape) >= 2 else 1
            K, N = int(a_shape[-1]), int(b_shape[0])
        elif fn_name in ("mm", "matmul"):
            M = int(a_shape[-2]) if len(a_shape) >= 2 else 1
            K, N = int(a_shape[-1]), int(b_shape[-1])
        elif fn_name == "bmm":
            M, K, N = int(a_shape[-2]), int(a_shape[-1]), int(b_shape[-1])
        elif fn_name == "addmm" and len(arg_vals) >= 3:
            a_shape = list(arg_vals[1].shape)
            b_shape = list(arg_vals[2].shape)
            M = int(a_shape[-2]) if len(a_shape) >= 2 else 1
            K, N = int(a_shape[-1]), int(b_shape[-1])
        else:
            return None

        return M, N, K
    except (TypeError, ValueError):
        return None  # symbolic (SymInt) shapes — can't determine statically


def default_extern_dispatch(node: fx.Node, *, gemm_threshold: int = 128) -> bool:
    """Default predicate for dispatching FX nodes to torch.

    Dispatches to torch (cuBLAS/FlashAttention) when:
    - The op is ``scaled_dot_product_attention``
    - The op is a matmul-family op where all of M, N, K >= ``gemm_threshold``

    Everything else is compiled by TileLang.

    Users can replace this with a custom callable to control dispatch::

        def my_dispatch(node):
            # Always use torch for LayerNorm
            if node.target.__name__ == "layer_norm":
                return True
            # Use default for everything else
            return default_extern_dispatch(node, gemm_threshold=256)

        compiled = torch.compile(model, backend="tilelang")
        # Or via fx_to_relax(gm, inputs, extern_dispatch=my_dispatch)
    """
    fn_name = node.target.__name__ if node.op == "call_function" else ""
    if fn_name == "scaled_dot_product_attention":
        return True

    mnk = _get_matmul_mnk(node)
    if mnk is not None:
        M, N, K = mnk
        return M >= gemm_threshold and N >= gemm_threshold and K >= gemm_threshold

    return False


def extract_fallback_calls(
    gm: fx.GraphModule,
    *,
    extern_dispatch: "Callable[[fx.Node], bool] | None" = None,
) -> dict[str, tuple]:
    """Scan FX graph for unsupported / extern-dispatched ops.

    Used when loading from disk cache — avoids re-running full conversion
    but still provides the callable references for fallback ops.
    """
    convert_map = _get_supported_ops()
    modules = dict(gm.named_modules())
    _dispatch = extern_dispatch or default_extern_dispatch
    fallback_calls = {}

    def _build_arg_template(node):
        tpl = []
        for a in node.args:
            if isinstance(a, fx.Node):
                tpl.append((True, a.name))
            elif isinstance(a, (list, tuple)) and any(isinstance(e, fx.Node) for e in a):
                tpl.append(("list", [e.name if isinstance(e, fx.Node) else e for e in a]))
            else:
                tpl.append((False, a))
        return tpl

    def _is_fallback(node, key):
        if _dispatch(node):
            return True
        if key not in convert_map:
            return True
        return False

    for node in gm.graph.nodes:
        if node.op == "call_function":
            fn = node.target.__name__
            if _is_fallback(node, fn):
                fallback_calls[fn] = (node.target, _build_arg_template(node), dict(node.kwargs))
        elif node.op == "call_method":
            if node.target not in convert_map:
                fallback_calls[node.target] = (node.target, _build_arg_template(node), dict(node.kwargs))
        elif node.op == "call_module":
            module = modules.get(node.target)
            if module is not None and type(module) not in convert_map:
                fn = type(module).__name__
                fallback_calls[fn] = (type(module), _build_arg_template(node), dict(node.kwargs))
    return fallback_calls
