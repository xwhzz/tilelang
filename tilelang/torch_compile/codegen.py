"""Wrapper code generation for the TileLang torch.compile backend."""

from __future__ import annotations

import textwrap
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch
from tvm import tir as _tir

if TYPE_CHECKING:
    from .analysis import _TIRCallRecord


def _safe_name(name: str) -> str:
    return name.replace(".", "_").replace("-", "_")


def _prim_expr_to_code(expr) -> str:
    """Convert a TVM PrimExpr to a Python code string for wrapper codegen."""
    if isinstance(expr, int):
        return str(expr)
    if isinstance(expr, _tir.IntImm):
        return str(int(expr))
    if isinstance(expr, _tir.Var):
        return f"_sym_{expr.name}"
    if isinstance(expr, _tir.Add):
        return f"({_prim_expr_to_code(expr.a)} + {_prim_expr_to_code(expr.b)})"
    if isinstance(expr, _tir.Mul):
        return f"({_prim_expr_to_code(expr.a)} * {_prim_expr_to_code(expr.b)})"
    if isinstance(expr, _tir.Sub):
        return f"({_prim_expr_to_code(expr.a)} - {_prim_expr_to_code(expr.b)})"
    if isinstance(expr, _tir.FloorDiv):
        return f"({_prim_expr_to_code(expr.a)} // {_prim_expr_to_code(expr.b)})"
    if isinstance(expr, _tir.FloorMod):
        return f"({_prim_expr_to_code(expr.a)} % {_prim_expr_to_code(expr.b)})"
    if isinstance(expr, _tir.Max):
        return f"max({_prim_expr_to_code(expr.a)}, {_prim_expr_to_code(expr.b)})"
    if isinstance(expr, _tir.Min):
        return f"min({_prim_expr_to_code(expr.a)}, {_prim_expr_to_code(expr.b)})"
    if isinstance(expr, _tir.Cast):
        return _prim_expr_to_code(expr.value)
    try:
        return str(int(expr))
    except (TypeError, ValueError):
        return str(expr)


_DTYPE_SIZES: dict[torch.dtype, int] = {
    torch.float16: 2,
    torch.bfloat16: 2,
    torch.float32: 4,
    torch.float64: 8,
    torch.int8: 1,
    torch.int16: 2,
    torch.int32: 4,
    torch.int64: 8,
    torch.uint8: 1,
    torch.bool: 1,
}


def _dtype_size(dtype: torch.dtype) -> int:
    return _DTYPE_SIZES.get(dtype, 0)


def _should_narrow(actual: torch.dtype, expected: torch.dtype) -> bool:
    if actual == expected:
        return False
    actual_size = _dtype_size(actual)
    expected_size = _dtype_size(expected)
    return actual_size > expected_size > 0


@dataclass
class WrapperCodeGen:
    """Generate Python wrapper code from an extracted TIR call sequence."""

    param_names: list[str]
    call_seq: list["_TIRCallRecord"]
    output_names: list[str]
    expected_dtypes: list[torch.dtype] = field(default_factory=list)
    device_index: int = 0
    sym_shape_map: dict[str, str] = field(default_factory=dict)
    extern_ops: dict[str, object] = field(default_factory=dict)
    constants: dict[str, object] = field(default_factory=dict)

    _param_idx: dict[str, int] = field(init=False, repr=False)
    _output_set: frozenset[str] = field(init=False, repr=False)
    _param_set: frozenset[str] = field(init=False, repr=False)
    _extern_outputs: frozenset[str] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._param_idx = {name: i for i, name in enumerate(self.param_names)}
        self._output_set = frozenset(self.output_names)
        self._param_set = frozenset(self.param_names)
        self._extern_outputs = frozenset(
            record.out_name for record in self.call_seq if record.is_torch_fallback
        )

    def generate(self) -> str:
        parts = [
            self._gen_header(),
            self._gen_setup_open(),
            self._gen_kernel_fetches(),
            self._gen_workspace_allocs(),
            self._gen_call_open(),
            self._gen_shape_extraction(),
            self._gen_input_conversions(),
            self._gen_kernel_calls(),
            self._gen_return(),
            self._gen_call_close(),
            self._gen_setup_close(),
        ]
        return "\n".join(parts) + "\n"

    def _has_tir_kernels(self) -> bool:
        return any(not record.is_torch_fallback for record in self.call_seq)

    def _intermediates(self) -> list["_TIRCallRecord"]:
        output_set = self._output_set
        return [
            record
            for record in self.call_seq
            if record.out_name not in output_set and not record.is_torch_fallback
        ]

    def _torch_dtype_str(self, tvm_dtype: str) -> str:
        return f"torch.{tvm_dtype}"

    def _is_dynamic(self) -> bool:
        return bool(self.sym_shape_map)

    def _shape_to_code(self, shape: tuple) -> str:
        parts = []
        for dim in shape:
            if isinstance(dim, int):
                parts.append(str(dim))
            else:
                parts.append(_prim_expr_to_code(dim))
        if not parts:
            return "()"
        if len(parts) == 1:
            return f"({parts[0]},)"
        return f"({', '.join(parts)})"

    def _torch_var(self, arg_name: str) -> str:
        if arg_name.startswith("_const_"):
            return f'_constants["{arg_name}"]'
        if arg_name in self._param_idx:
            return f"inp_{self._param_idx[arg_name]}"
        safe = _safe_name(arg_name)
        if arg_name in self._output_set:
            return f"_out_{safe}"
        return f"_ws_{safe}"

    def _gen_header(self) -> str:
        return textwrap.dedent(
            """\
            # Auto-generated by TileLang torch.compile backend
            import torch
            from tvm_ffi import use_torch_stream
            """
        )

    def _gen_setup_open(self) -> str:
        return "def _setup(_rt_mod, _from_dlpack, _extern_ops=None, _constants=None):"

    def _gen_kernel_fetches(self) -> str:
        lines: list[str] = []
        seen_funcs: set[str] = set()
        for record in self.call_seq:
            if record.is_torch_fallback or record.func_name in seen_funcs:
                continue
            seen_funcs.add(record.func_name)
            safe = _safe_name(record.func_name)
            lines.append(f'    _k_{safe} = _rt_mod["{record.func_name}"]')

        seen_externs: set[str] = set()
        for record in self.call_seq:
            if record.extern_op is None or record.func_name in seen_externs:
                continue
            seen_externs.add(record.func_name)
            safe = _safe_name(record.func_name)
            lines.append(f'    _ext_{safe} = _extern_ops["{record.func_name}"]')

        seen_constants: set[str] = set()
        for record in self.call_seq:
            for name in record.arg_names:
                if not name.startswith("_const_") or name in seen_constants:
                    continue
                seen_constants.add(name)
                lines.append(
                    f'    _tvm_{name} = _from_dlpack(_constants["{name}"])'
                )

        if not lines:
            lines.append("    pass")
        return "\n".join(lines)

    def _gen_workspace_allocs(self) -> str:
        if self._is_dynamic():
            return ""
        lines: list[str] = []
        device = f'"cuda:{self.device_index}"'
        for record in self._intermediates():
            safe = _safe_name(record.out_name)
            dtype_str = self._torch_dtype_str(record.out_dtype)
            lines.append(
                f"    _ws_{safe} = torch.empty({self._shape_to_code(record.out_shape)}, "
                f"dtype={dtype_str}, device={device})"
            )
        return "\n".join(lines)

    def _gen_call_open(self) -> str:
        arg_list = ", ".join(f"inp_{i}" for i in range(len(self.param_names)))
        return f"\n    def _call({arg_list}):"

    def _gen_shape_extraction(self) -> str:
        if not self._is_dynamic():
            return ""
        lines = []
        for var_name in sorted(self.sym_shape_map):
            lines.append(f"        _sym_{var_name} = {self.sym_shape_map[var_name]}")
        return "\n".join(lines)

    def _gen_input_conversions(self) -> str:
        if not self._has_tir_kernels():
            return ""
        lines = []
        for index in range(len(self.param_names)):
            lines.append(f"        _tvm_inp_{index} = _from_dlpack(inp_{index}.contiguous())")
        return "\n".join(lines)

    def _gen_kernel_calls(self) -> str:
        output_set = self._output_set
        param_idx = self._param_idx
        device = f'"cuda:{self.device_index}"'

        lines: list[str] = []
        indent = "        "
        if self._has_tir_kernels():
            lines.append("")
            lines.append("        with use_torch_stream():")
            indent = "            "

        for index, record in enumerate(self.call_seq):
            safe_out = _safe_name(record.out_name)
            is_output = record.out_name in output_set
            dtype_str = self._torch_dtype_str(record.out_dtype)

            if record.is_torch_fallback:
                if record.extern_op is None:
                    # No extern backing — should not reach codegen.
                    # compile_subgraph_direct returns None for this case.
                    raise RuntimeError(
                        f"Function {record.func_name} is fallback with no "
                        f"extern op — should have triggered eager subgraph fallback"
                    )

                lines.append(f"{indent}# Extern {index}: {record.extern_op.qualname}")

                def _literal_code(value) -> str:
                    if isinstance(value, torch.device):
                        return f'torch.device("{value}")'
                    if isinstance(value, torch.dtype):
                        return f"torch.{value}".replace("torch.torch.", "torch.")
                    return repr(value)

                call_parts: list[str] = []
                tensor_index = 0
                for kind, value in record.extern_op.arg_spec:
                    if kind == "tensor":
                        call_parts.append(self._torch_var(record.arg_names[tensor_index]))
                        tensor_index += 1
                    elif kind == "tensor_list":
                        items = [self._torch_var(record.arg_names[i]) for i in value]
                        call_parts.append(f"[{', '.join(items)}]")
                        tensor_index = max(value) + 1 if value else tensor_index
                    else:
                        call_parts.append(_literal_code(value))
                for kwarg_name, kwarg_value in record.extern_op.literal_kwargs.items():
                    call_parts.append(f"{kwarg_name}={_literal_code(kwarg_value)}")
                expr = f"_ext_{_safe_name(record.func_name)}({', '.join(call_parts)})"

                target_name = f"_out_{safe_out}" if is_output else f"_ws_{safe_out}"
                lines.append(f"{indent}{target_name} = {expr}")
                continue

            lines.append(f"{indent}# Kernel {index}: {record.func_name}")
            shape_code = self._shape_to_code(record.out_shape)
            if is_output:
                lines.append(
                    f"{indent}_out_{safe_out} = torch.empty({shape_code}, dtype={dtype_str}, device={device})"
                )
                lines.append(f"{indent}_tvm_out_{safe_out} = _from_dlpack(_out_{safe_out})")
            else:
                if self._is_dynamic():
                    lines.append(
                        f"{indent}_ws_{safe_out} = torch.empty({shape_code}, dtype={dtype_str}, device={device})"
                    )
                lines.append(f"{indent}_tvm_ws_{safe_out} = _from_dlpack(_ws_{safe_out})")

            tvm_args: list[str] = []
            for arg_i, arg_name in enumerate(record.arg_names):
                if arg_name.startswith("_const_"):
                    tvm_args.append(f"_tvm_{arg_name}")
                elif arg_name in param_idx:
                    tvm_args.append(f"_tvm_inp_{param_idx[arg_name]}")
                else:
                    safe_arg = _safe_name(arg_name)
                    if arg_name in self._extern_outputs:
                        # Extern op output → convert to TVM via DLPack.
                        # Cast to expected dtype if TIR kernel has different expectations.
                        torch_var = self._torch_var(arg_name)
                        bridge = f"{torch_var}.contiguous()"
                        if arg_i < len(record.arg_dtypes) and record.arg_dtypes[arg_i]:
                            bridge = f"{bridge}.to({self._torch_dtype_str(record.arg_dtypes[arg_i])})"
                        tvm_args.append(f"_from_dlpack({bridge})")
                    elif arg_name in output_set:
                        tvm_args.append(f"_tvm_out_{safe_arg}")
                    else:
                        tvm_args.append(f"_tvm_ws_{safe_arg}")

            tvm_args.append(f"_tvm_out_{safe_out}" if is_output else f"_tvm_ws_{safe_out}")
            for expr in record.tir_var_exprs:
                tvm_args.append(_prim_expr_to_code(expr))
            lines.append(f"{indent}_k_{_safe_name(record.func_name)}({', '.join(tvm_args)})")

        return "\n".join(lines)

    def _gen_return(self) -> str:
        result_exprs: list[str] = []
        for output_index, output_name in enumerate(self.output_names):
            if output_name in self._param_set:
                result_exprs.append(f"inp_{self._param_idx[output_name]}")
                continue

            safe = _safe_name(output_name)
            expr = f"_out_{safe}"
            if output_index < len(self.expected_dtypes):
                for record in self.call_seq:
                    if record.out_name != output_name:
                        continue
                    actual_dtype = getattr(torch, record.out_dtype, torch.float32)
                    if _should_narrow(actual_dtype, self.expected_dtypes[output_index]):
                        expected_name = str(self.expected_dtypes[output_index]).replace("torch.", "")
                        expr = f"_out_{safe}.to(torch.{expected_name})"
                    break
            result_exprs.append(expr)

        if not result_exprs:
            return "        return ()"
        items = ", ".join(result_exprs)
        if len(result_exprs) == 1:
            return f"\n        return ({items},)"
        return f"\n        return ({items})"

    def _gen_call_close(self) -> str:
        return "\n    return _call"

    def _gen_setup_close(self) -> str:
        return ""

