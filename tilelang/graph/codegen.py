"""Generate a Python wrapper from lowered Relax IR + compiled TileLang kernels."""

import logging
import os
from dataclasses import dataclass

import torch

from tilelang import tvm as tvm
from tvm import relax, tir, ir, runtime

from tilelang.graph.utils import tvm_dtype_to_torch

logger = logging.getLogger(__name__)

_DTYPE_BYTES = {
    "float16": 2, "bfloat16": 2, "float32": 4, "float64": 8,
    "int8": 1, "int16": 2, "int32": 4, "int64": 8,
    "uint8": 1, "bool": 1,
}


@dataclass
class AllocStorageInstr:
    """Allocate a raw storage buffer (from StaticPlanBlockMemory)."""
    var: str
    size: object  # total bytes: int (static) or tir.PrimExpr (dynamic)
    dtype: str  # storage dtype (usually "uint8")


@dataclass
class AllocTensorInstr:
    """Allocate a tensor view from a storage buffer."""
    var: str
    storage_var: str  # references an AllocStorageInstr var
    offset: int  # byte offset into storage
    shape: list  # list of int or tir.Var
    dtype: torch.dtype


@dataclass
class AllocInstr:
    """Allocate an output tensor (legacy — for un-lowered alloc_tensor)."""
    var: str
    shape: list
    dtype: torch.dtype


@dataclass
class KernelCallInstr:
    """Call a compiled TileLang kernel."""
    kernel_name: str
    arg_vars: list[str]
    flatten_args: bool = False  # True when kernel expects 1D (from transform_layout)
    sym_vars: list[str] = None  # symbolic var names to pass as extra int args


@dataclass
class ConstantInstr:
    """Bind a pre-extracted constant tensor."""
    var: str


@dataclass
class ReshapeInstr:
    """Reshape a tensor (zero-copy view)."""
    var: str
    input_var: str
    shape: list[int]


@dataclass
class TupleInstr:
    """Construct a tuple from elements."""
    var: str
    element_vars: list[str]


@dataclass
class TupleGetItemInstr:
    """Extract an element from a tuple."""
    var: str
    tuple_var: str
    index: int


@dataclass
class AliasInstr:
    """Bind var to point to the same tensor as source_var."""
    var: str
    source_var: str


@dataclass
class RelaxOpInstr:
    """Un-lowered Relax op (dynamic shapes prevented TIR lowering)."""
    var: str
    op_name: str  # e.g. "relax.multiply", "relax.astype"
    arg_vars: list[str]
    attrs: dict  # op-specific attributes (e.g. dtype for astype)


@dataclass
class TorchFallbackInstr:
    """Call an unsupported op via torch at runtime."""
    var: str
    op_name: str
    arg_vars: list
    shape: list[int]
    dtype: torch.dtype


def _get_shape_values(shape_expr):
    """Extract shape dimensions from a ShapeExpr.

    Returns a list of int (static dims) or tir.Var (symbolic dims).
    """
    dims = []
    for v in shape_expr.values:
        if isinstance(v, tir.IntImm):
            dims.append(int(v))
        elif isinstance(v, tir.Var):
            dims.append(v)  # symbolic dim — resolved at runtime
        else:
            try:
                dims.append(int(v))
            except (TypeError, ValueError):
                dims.append(v)
    return dims



def _is_alloc_tensor_op(call):
    """Check if a Call node is relax.builtin.alloc_tensor."""
    if not isinstance(call, relax.Call):
        return False
    op = call.op
    if isinstance(op, ir.Op) and op.name == "relax.builtin.alloc_tensor":
        return True
    return False


def _is_call_to_global_var(call):
    """Check if a Call node calls a GlobalVar (i.e., a TIR function)."""
    return isinstance(call, relax.Call) and isinstance(call.op, ir.GlobalVar)


def _is_torch_fallback_call(call):
    """Check if a Call is an opaque torch fallback (ExternFunc)."""
    if not isinstance(call, relax.Call):
        return False
    op = call.op
    if isinstance(op, relax.ExternFunc):
        return str(op.global_symbol).startswith("torch_fallback.")
    # After CallTIRRewrite, call_dps_packed becomes a direct call
    # Check nested: call(call_dps_packed_op, [extern_func, ...])
    if isinstance(op, ir.Op) and "call_dps_packed" in op.name:
        if len(call.args) > 0 and isinstance(call.args[0], relax.ExternFunc):
            return str(call.args[0].global_symbol).startswith("torch_fallback.")
    return False


def _get_torch_fallback_info(call, unique_name_fn=None):
    """Extract (op_name, arg_vars) from a torch fallback call."""
    if isinstance(call.op, relax.ExternFunc):
        op_name = str(call.op.global_symbol)[len("torch_fallback."):]
        arg_exprs = call.args
    else:
        op_name = str(call.args[0].global_symbol)[len("torch_fallback."):]
        arg_exprs = call.args[1] if len(call.args) > 1 else []
        if isinstance(arg_exprs, relax.Tuple):
            arg_exprs = arg_exprs.fields
    arg_vars = []
    for a in arg_exprs:
        if hasattr(a, "name_hint"):
            # Var or DataflowVar
            arg_vars.append(unique_name_fn(a) if unique_name_fn else a.name_hint)
        else:
            arg_vars.append(a)
    return op_name, arg_vars


def _compile_instructions(mod: tvm.IRModule) -> tuple[list, list[str], list[str], dict[str, torch.Tensor]]:
    """Walk the lowered Relax IR and produce a list of instructions.

    Returns (instructions, param_names, output_var_names, constants).
    """
    main_func = None
    for gvar, func in mod.functions.items():
        if isinstance(func, relax.Function):
            main_func = func
            break

    if main_func is None:
        raise RuntimeError("No Relax function found in module")

    # Build a unique-name mapping: Relax can reuse name_hint for different
    # SSA Vars.  We append _N suffixes to disambiguate.
    _name_counts: dict[str, int] = {}
    _vid_to_name: dict = {}  # Var.vid → unique name

    def _unique_name(var):
        """Get or assign a unique name for a Relax Var, keyed by vid (SSA identity)."""
        vid = var.vid
        if vid in _vid_to_name:
            return _vid_to_name[vid]
        hint = var.name_hint
        count = _name_counts.get(hint, 0)
        _name_counts[hint] = count + 1
        uname = hint if count == 0 else f"{hint}_{count}"
        _vid_to_name[vid] = uname
        return uname

    def _resolve_expr_name(expr):
        """Get unique name for a Relax expression (Var or other)."""
        if isinstance(expr, relax.Var):
            return _unique_name(expr)
        return str(expr)

    param_names = [_unique_name(p) for p in main_func.params]

    body = main_func.body
    if not isinstance(body, relax.SeqExpr):
        raise RuntimeError("Expected SeqExpr as function body")

    instructions = []
    constants = {}  # unique_name → torch.Tensor

    for block in body.blocks:
        for binding in block.bindings:
            if not isinstance(binding, relax.VarBinding):
                continue

            var_name = _unique_name(binding.var)
            value = binding.value

            if isinstance(value, relax.Constant):
                nd_arr = value.data
                constants[var_name] = torch.from_numpy(nd_arr.numpy())
                instructions.append(ConstantInstr(var=var_name))

            elif (isinstance(value, relax.Call) and isinstance(value.op, ir.Op)
                  and value.op.name == "relax.memory.alloc_storage"):
                # R.memory.alloc_storage(ShapeExpr([size]), device_idx, scope, dtype)
                size_shape = value.args[0]
                if isinstance(size_shape, relax.ShapeExpr):
                    sv = size_shape.values[0]
                    size = int(sv) if isinstance(sv, tir.IntImm) else sv
                else:
                    size = 0
                dtype_arg = value.args[3]
                dtype_str = str(dtype_arg.value) if hasattr(dtype_arg, "value") else "uint8"
                instructions.append(AllocStorageInstr(
                    var=var_name, size=size, dtype=dtype_str))

            elif (isinstance(value, relax.Call) and isinstance(value.op, ir.Op)
                  and value.op.name == "relax.memory.alloc_tensor"):
                # R.memory.alloc_tensor(storage, offset, shape, dtype)
                storage_var = _unique_name(value.args[0])
                offset_arg = value.args[1]
                offset = int(offset_arg.value) if hasattr(offset_arg, "value") else 0
                shape_expr = value.args[2]
                shape = _get_shape_values(shape_expr)
                dtype_arg = value.args[3]
                dtype_str = str(dtype_arg.value) if hasattr(dtype_arg, "value") else "float16"
                instructions.append(AllocTensorInstr(
                    var=var_name, storage_var=storage_var,
                    offset=offset, shape=shape,
                    dtype=tvm_dtype_to_torch(dtype_str)))

            elif (isinstance(value, relax.Call) and isinstance(value.op, ir.Op)
                  and value.op.name in ("relax.null_value",)):
                # VM infrastructure — skip at runtime
                pass

            elif (isinstance(value, relax.Call) and isinstance(value.op, relax.ExternFunc)
                  and str(value.op.global_symbol).startswith("vm.builtin.")):
                # VM validation ops (check_tensor_info, match_shape) — skip
                pass

            elif _is_alloc_tensor_op(value):
                shape_expr = value.args[0]
                dtype_imm = value.args[1]
                shape = _get_shape_values(shape_expr)
                dtype_str = str(dtype_imm.value) if hasattr(dtype_imm, 'value') else str(dtype_imm)
                torch_dtype = tvm_dtype_to_torch(dtype_str)
                instructions.append(AllocInstr(var=var_name, shape=shape, dtype=torch_dtype))

            elif _is_torch_fallback_call(value):
                op_name, arg_vars = _get_torch_fallback_info(value, _unique_name)
                sinfo = binding.var.struct_info_
                shape = _get_shape_values(sinfo.shape) if (
                    sinfo and isinstance(sinfo, relax.TensorStructInfo) and sinfo.shape
                ) else []
                dtype = tvm_dtype_to_torch(sinfo.dtype) if (
                    sinfo and isinstance(sinfo, relax.TensorStructInfo) and sinfo.dtype
                ) else torch.float32
                instructions.append(TorchFallbackInstr(
                    var=var_name, op_name=op_name, arg_vars=arg_vars,
                    shape=shape, dtype=dtype))

            elif (isinstance(value, relax.Call) and isinstance(value.op, ir.Op)
                  and value.op.name == "relax.vm.call_tir_dyn"):
                # Dynamic-shape TIR call: call_tir_dyn(GlobalVar, Tuple(inputs..., ShapeExpr))
                gvar = value.args[0]
                kernel_name = gvar.name_hint
                packed_tuple = value.args[1]
                arg_vars = []
                if isinstance(packed_tuple, relax.Tuple):
                    for f in packed_tuple.fields:
                        if isinstance(f, relax.ShapeExpr):
                            # Output shape descriptor — skip (alloc handled separately)
                            pass
                        elif hasattr(f, "name_hint"):
                            arg_vars.append(_unique_name(f))
                # Find symbolic var params in the TIR function
                tir_func = mod[gvar]
                sym_vars = []
                if isinstance(tir_func, tir.PrimFunc):
                    for p in tir_func.params:
                        if p not in tir_func.buffer_map and isinstance(p, tir.Var):
                            sym_vars.append(p.name)
                # Detect if kernel expects 1D (flattened) buffers
                flatten = False
                if isinstance(tir_func, tir.PrimFunc):
                    flatten = all(len(b.shape) <= 1
                                 for b in tir_func.buffer_map.values())
                instructions.append(KernelCallInstr(
                    kernel_name=kernel_name, arg_vars=arg_vars,
                    sym_vars=sym_vars if sym_vars else None,
                    flatten_args=flatten))

            elif _is_call_to_global_var(value):
                kernel_name = value.op.name_hint
                arg_vars = []
                sym_vars = []
                tir_func = mod[value.op] if value.op in mod.functions else None
                for arg in value.args:
                    if isinstance(arg, relax.Constant):
                        cname = f"__inline_const_{len(constants)}"
                        constants[cname] = torch.from_numpy(arg.data.numpy())
                        arg_vars.append(cname)
                    elif hasattr(arg, "name_hint"):
                        arg_vars.append(_unique_name(arg))
                    else:
                        arg_vars.append(arg)
                # Check for symbolic params and flattened buffers
                flatten = False
                if isinstance(tir_func, tir.PrimFunc):
                    for p in tir_func.params:
                        if p not in tir_func.buffer_map and isinstance(p, tir.Var):
                            sym_vars.append(p.name)
                    flatten = all(len(b.shape) <= 1
                                 for b in tir_func.buffer_map.values())
                instructions.append(KernelCallInstr(
                    kernel_name=kernel_name, arg_vars=arg_vars,
                    sym_vars=sym_vars if sym_vars else None,
                    flatten_args=flatten))

            elif isinstance(value, relax.Tuple):
                element_vars = [_resolve_expr_name(f) for f in value.fields]
                instructions.append(TupleInstr(var=var_name, element_vars=element_vars))

            elif isinstance(value, relax.TupleGetItem):
                tuple_var = _resolve_expr_name(value.tuple_value)
                instructions.append(TupleGetItemInstr(
                    var=var_name, tuple_var=tuple_var, index=value.index))

            elif isinstance(value, relax.Var):
                instructions.append(AliasInstr(
                    var=var_name, source_var=_unique_name(value)))

            elif (isinstance(value, relax.Call) and isinstance(value.op, ir.Op)
                  and value.op.name == "relax.reshape"):
                arg0 = value.args[0]
                if isinstance(arg0, relax.Constant):
                    cname = f"__inline_const_{len(constants)}"
                    constants[cname] = torch.from_numpy(arg0.data.numpy())
                    input_var = cname
                else:
                    input_var = _resolve_expr_name(arg0)
                shape_expr = value.args[1]
                shape = _get_shape_values(shape_expr)
                instructions.append(ReshapeInstr(
                    var=var_name, input_var=input_var, shape=shape))

            elif isinstance(value, relax.Call) and isinstance(value.op, ir.Op):
                # Un-lowered Relax op (dynamic shapes prevented LegalizeOps
                # from creating TIR).  Emit as a runtime torch call.
                op_name = value.op.name
                arg_vars = []
                for a in value.args:
                    if isinstance(a, relax.Constant):
                        cname = f"__inline_const_{len(constants)}"
                        constants[cname] = torch.from_numpy(a.data.numpy())
                        arg_vars.append(cname)
                    elif hasattr(a, "name_hint"):
                        arg_vars.append(_unique_name(a))
                    # Skip non-tensor args like ShapeExpr
                attrs = {}
                if value.attrs:
                    try:
                        for key in value.attrs.keys():
                            attrs[key] = value.attrs[key]
                    except AttributeError:
                        for field in dir(value.attrs):
                            if not field.startswith("_") and field not in ("handle", "same_as"):
                                try:
                                    attrs[field] = getattr(value.attrs, field)
                                except Exception:
                                    pass
                instructions.append(RelaxOpInstr(
                    var=var_name, op_name=op_name,
                    arg_vars=arg_vars, attrs=attrs))

            else:
                logger.debug("Unhandled binding type for %s: %s", var_name, type(value))

    # Determine output vars from the body expression
    output_body = body.body
    if isinstance(output_body, relax.Var):
        output_vars = [_unique_name(output_body)]
    elif isinstance(output_body, relax.Tuple):
        output_vars = [_resolve_expr_name(f) for f in output_body.fields]
    else:
        output_vars = []

    return instructions, param_names, output_vars, constants


def _collect_refs(instructions, output_vars):
    """Collect all var names referenced by instructions."""
    referenced = set(output_vars)
    for instr in instructions:
        if isinstance(instr, KernelCallInstr):
            referenced.update(a for a in instr.arg_vars if isinstance(a, str))
        elif isinstance(instr, TorchFallbackInstr):
            referenced.update(a for a in instr.arg_vars if isinstance(a, str))
        elif isinstance(instr, ReshapeInstr):
            referenced.add(instr.input_var)
        elif isinstance(instr, AliasInstr):
            referenced.add(instr.source_var)
        elif isinstance(instr, TupleInstr):
            referenced.update(instr.element_vars)
        elif isinstance(instr, TupleGetItemInstr):
            referenced.add(instr.tuple_var)
        elif isinstance(instr, AllocTensorInstr):
            referenced.add(instr.storage_var)
    return referenced


def _eliminate_dead_instructions(instructions, output_vars):
    """Remove instructions whose outputs are never referenced."""
    referenced = _collect_refs(instructions, output_vars)

    # Remove fallback instructions that produce unused outputs AND whose
    # DPS output arg is also unused. Kernel calls are never eliminated
    # (they have side effects on output buffers).
    live = []
    for instr in instructions:
        if isinstance(instr, TorchFallbackInstr):
            var = instr.var
            all_args_dead = (var not in referenced and
                             all(a not in referenced
                                 for a in instr.arg_vars if isinstance(a, str)))
            if all_args_dead:
                continue
        live.append(instr)

    # Second pass: remove allocs/storages whose var is no longer referenced
    referenced2 = _collect_refs(live, output_vars)

    result = []
    for instr in live:
        var = getattr(instr, "var", None)
        if isinstance(instr, (AllocInstr, AllocStorageInstr)) and var not in referenced2:
            continue
        result.append(instr)

    return result


def _sanitize_var(name: str) -> str:
    """Make a var name safe for use as a Python identifier."""
    return name.replace(".", "_").replace("-", "_").replace(" ", "_")


def _plan_memory_reuse(instructions, output_vars):
    """Build a memory reuse plan for AllocInstr buffers.

    Analyzes tensor lifetimes and assigns each allocation to a reusable
    pool slot. Buffers with the same shape+dtype whose lifetimes don't
    overlap share the same slot.

    Returns:
        alloc_to_slot: dict mapping alloc var name → pool slot index
        pool_specs: list of (shape, dtype) for each slot
    """
    # Collect all alloc vars with their shapes/dtypes
    allocs = {}  # var_name → (shape, dtype, instr_idx)
    for i, instr in enumerate(instructions):
        if isinstance(instr, AllocInstr):
            # Skip dynamic shapes — can't pre-allocate
            if any(not isinstance(s, int) for s in instr.shape):
                continue
            allocs[instr.var] = (tuple(instr.shape), instr.dtype, i)

    if not allocs:
        return {}, []

    # Build alias mapping: alloc_var → set of all vars that reference it
    # (through alias chains like alloc → lv → lv2 → ...)
    alias_to_alloc = {}  # alias_var → alloc_var
    for instr in instructions:
        if isinstance(instr, AliasInstr):
            # trace source back to its alloc
            src = instr.source_var
            root = alias_to_alloc.get(src, src)
            if root in allocs:
                alias_to_alloc[instr.var] = root

    # Find last-use index for each alloc var, including through aliases
    last_use = {var: idx for var, (_, _, idx) in allocs.items()}

    for i, instr in enumerate(instructions):
        refs = set()
        if isinstance(instr, KernelCallInstr):
            refs.update(a for a in instr.arg_vars if isinstance(a, str))
        elif isinstance(instr, TorchFallbackInstr):
            refs.update(a for a in instr.arg_vars if isinstance(a, str))
        elif isinstance(instr, AliasInstr):
            refs.add(instr.source_var)
        elif isinstance(instr, ReshapeInstr):
            refs.add(instr.input_var)
        elif isinstance(instr, RelaxOpInstr):
            refs.update(instr.arg_vars)
        elif isinstance(instr, TupleInstr):
            refs.update(instr.element_vars)
        elif isinstance(instr, TupleGetItemInstr):
            refs.add(instr.tuple_var)

        for ref in refs:
            # Resolve through aliases to the underlying alloc
            root = alias_to_alloc.get(ref, ref)
            if root in allocs:
                last_use[root] = max(last_use[root], i)

    # Don't reuse buffers that appear in the output tuple
    output_allocs = set()
    for ov in output_vars:
        root = alias_to_alloc.get(ov, ov)
        if root in allocs:
            output_allocs.add(root)

    # Greedy interval scheduling: assign slots
    # Sort allocs by start index
    sorted_allocs = sorted(allocs.items(), key=lambda x: x[1][2])

    pool_specs = []  # (shape, dtype) per slot
    pool_free_at = []  # instruction index when slot becomes free
    alloc_to_slot = {}

    for var, (shape, dtype, start_idx) in sorted_allocs:
        end_idx = last_use[var]

        # Don't reuse output buffers
        if var in output_allocs:
            slot = len(pool_specs)
            pool_specs.append((shape, dtype))
            pool_free_at.append(float('inf'))
            alloc_to_slot[var] = slot
            continue

        # Find a free slot with matching shape+dtype
        best_slot = None
        for s, (s_shape, s_dtype) in enumerate(pool_specs):
            if s_shape == shape and s_dtype == dtype and pool_free_at[s] <= start_idx:
                best_slot = s
                break

        if best_slot is not None:
            alloc_to_slot[var] = best_slot
            pool_free_at[best_slot] = end_idx + 1
        else:
            slot = len(pool_specs)
            pool_specs.append((shape, dtype))
            pool_free_at.append(end_idx + 1)
            alloc_to_slot[var] = slot

    return alloc_to_slot, pool_specs


def _tir_expr_to_python(expr, sym_var_map):
    """Convert a TIR PrimExpr to a Python runtime expression string."""
    if isinstance(expr, tir.IntImm):
        return str(int(expr))
    if isinstance(expr, tir.Var):
        if expr.name in sym_var_map:
            pidx, didx = sym_var_map[expr.name]
            return f"_tensor_inputs[{pidx}].shape[{didx}]"
        return expr.name
    if isinstance(expr, tir.Max):
        a = _tir_expr_to_python(expr.a, sym_var_map)
        b = _tir_expr_to_python(expr.b, sym_var_map)
        return f"max({a}, {b})"
    if isinstance(expr, tir.Min):
        a = _tir_expr_to_python(expr.a, sym_var_map)
        b = _tir_expr_to_python(expr.b, sym_var_map)
        return f"min({a}, {b})"
    if isinstance(expr, tir.FloorDiv):
        a = _tir_expr_to_python(expr.a, sym_var_map)
        b = _tir_expr_to_python(expr.b, sym_var_map)
        return f"({a} // {b})"
    if isinstance(expr, tir.FloorMod):
        a = _tir_expr_to_python(expr.a, sym_var_map)
        b = _tir_expr_to_python(expr.b, sym_var_map)
        return f"({a} % {b})"
    if isinstance(expr, (tir.Add, tir.Sub, tir.Mul, tir.Div)):
        a = _tir_expr_to_python(expr.a, sym_var_map)
        b = _tir_expr_to_python(expr.b, sym_var_map)
        op = {tir.Add: "+", tir.Sub: "-", tir.Mul: "*", tir.Div: "/"}[type(expr)]
        return f"({a} {op} {b})"
    if isinstance(expr, tir.Cast):
        return _tir_expr_to_python(expr.value, sym_var_map)
    if isinstance(expr, tir.Select):
        cond = _tir_expr_to_python(expr.condition, sym_var_map)
        t = _tir_expr_to_python(expr.true_value, sym_var_map)
        f = _tir_expr_to_python(expr.false_value, sym_var_map)
        return f"({t} if {cond} else {f})"
    # Comparison ops
    for cls, op in [(tir.LT, "<"), (tir.LE, "<="), (tir.GT, ">"), (tir.GE, ">="),
                    (tir.EQ, "=="), (tir.NE, "!=")]:
        if isinstance(expr, cls):
            a = _tir_expr_to_python(expr.a, sym_var_map)
            b = _tir_expr_to_python(expr.b, sym_var_map)
            return f"({a} {op} {b})"
    # tir.Call: if_then_else and others
    if isinstance(expr, tir.Call):
        op_name = getattr(expr.op, "name", "")
        if op_name == "tir.if_then_else":
            cond = _tir_expr_to_python(expr.args[0], sym_var_map)
            t = _tir_expr_to_python(expr.args[1], sym_var_map)
            f = _tir_expr_to_python(expr.args[2], sym_var_map)
            return f"({t} if {cond} else {f})"
        args = ", ".join(_tir_expr_to_python(a, sym_var_map) for a in expr.args)
        return f"{op_name}({args})"
    # Fallback: try int conversion
    try:
        return str(int(expr))
    except (TypeError, ValueError):
        return str(expr)


def _shape_code(shape, sym_var_map, bracket="["):
    """Generate shape code, resolving symbolic dims."""
    close = "]" if bracket == "[" else ")"
    has_symbolic = any(not isinstance(s, int) and not isinstance(s, tir.IntImm)
                       for s in shape)
    if has_symbolic:
        parts = [_tir_expr_to_python(s, sym_var_map) for s in shape]
        return f"{bracket}{', '.join(parts)}{close}"
    return repr(list(shape) if bracket == "[" else tuple(shape))


def _emit_python_source(instructions, param_names, output_vars, constants,
                        compiled_kernels, fallback_calls, sym_var_map,
                        alloc_to_slot):

    lines = []

    lines.append("def _compiled_wrapper(_tensor_inputs, _device, _C, _K, _F, _pool):")

    for i, pname in enumerate(param_names):
        v = _sanitize_var(pname)
        lines.append(f"    {v} = _tensor_inputs[{i}]")
        lines.append(f"    if {v}.device != _device: {v} = {v}.to(_device)")

    for cname in constants:
        lines.append(f"    {_sanitize_var(cname)} = _C[{cname!r}]")

    for instr in instructions:
        if isinstance(instr, ConstantInstr):
            pass

        elif isinstance(instr, AllocStorageInstr):
            v = _sanitize_var(instr.var)
            if isinstance(instr.size, int):
                # Static size: use pre-allocated storage from _pool dict
                lines.append(f"    {v} = _pool[{instr.var!r}]")
            else:
                # Dynamic size: allocate per call
                size_str = _tir_expr_to_python(instr.size, sym_var_map)
                lines.append(f"    {v} = _torch.empty({size_str}, dtype=_torch.uint8, device=_device)")

        elif isinstance(instr, AllocTensorInstr):
            v = _sanitize_var(instr.var)
            sv = _sanitize_var(instr.storage_var)
            dtype_str = str(instr.dtype).replace("torch.", "")
            shape_str = _shape_code(instr.shape, sym_var_map, "[")
            elem_size = _DTYPE_BYTES.get(dtype_str, 2)
            numel_str = " * ".join(
                str(s) if isinstance(s, int) else _tir_expr_to_python(s, sym_var_map)
                for s in instr.shape)
            # Slice bytes from storage, reinterpret as target dtype, reshape
            lines.append(
                f"    {v} = {sv}[{instr.offset}:{instr.offset}+{elem_size}*({numel_str})]"
                f".view(_torch.{dtype_str}).reshape({shape_str})")

        elif isinstance(instr, AllocInstr):
            if instr.var in alloc_to_slot:
                slot = alloc_to_slot[instr.var]
                lines.append(f"    {_sanitize_var(instr.var)} = _pool[{slot}]")
            else:
                shape_str = _shape_code(instr.shape, sym_var_map, "[")
                dtype_str = str(instr.dtype).replace("torch.", "")
                lines.append(
                    f"    {_sanitize_var(instr.var)} = _torch.empty("
                    f"{shape_str}, dtype=_torch.{dtype_str}, device=_device)")

        elif isinstance(instr, KernelCallInstr):
            if instr.flatten_args:
                tensor_args = ", ".join(
                    f"{_sanitize_var(a)}.contiguous().view(-1)"
                    if isinstance(a, str) else repr(a)
                    for a in instr.arg_vars)
            else:
                tensor_args = ", ".join(
                    f"{_sanitize_var(a)}.contiguous() if not {_sanitize_var(a)}.is_contiguous() else {_sanitize_var(a)}"
                    if isinstance(a, str) else repr(a)
                    for a in instr.arg_vars)
            if instr.sym_vars:
                # Append symbolic variable values (resolved from input shapes)
                sym_parts = []
                for sv in instr.sym_vars:
                    if sv in sym_var_map:
                        pidx, didx = sym_var_map[sv]
                        sym_parts.append(f"_tensor_inputs[{pidx}].shape[{didx}]")
                    else:
                        sym_parts.append(sv)
                all_args = tensor_args + ", " + ", ".join(sym_parts) if tensor_args else ", ".join(sym_parts)
            else:
                all_args = tensor_args
            lines.append(f"    _K[{instr.kernel_name!r}]({all_args})")

        elif isinstance(instr, TorchFallbackInstr):
            op_name = instr.op_name
            call_info = fallback_calls.get(op_name)
            if call_info is None:
                lines.append(f"    raise RuntimeError('No fallback for {op_name}')")
                continue
            _, arg_template, raw_kwargs = call_info

            # DPS convention: last arg is the output buffer when
            # it was emitted by call_dps_packed (the codegen sees the
            # alloc → call → alias pattern).  Detect by checking if the
            # number of arg_vars exceeds the template's tensor count.
            n_template_tensors = sum(1 for tag, _ in arg_template if tag is True)
            if instr.arg_vars and len(instr.arg_vars) > n_template_tensors:
                input_vars = instr.arg_vars[:-1]
                dps_var = instr.arg_vars[-1]
            else:
                input_vars = list(instr.arg_vars)
                dps_var = None

            # Build positional args
            tensor_idx = 0
            pos_args = []
            for tag, val in arg_template:
                if tag is True and tensor_idx < len(input_vars):
                    pos_args.append(_sanitize_var(input_vars[tensor_idx]))
                    tensor_idx += 1
                elif tag == "list":
                    items = []
                    for nm in val:
                        if isinstance(nm, str) and tensor_idx < len(input_vars):
                            items.append(_sanitize_var(input_vars[tensor_idx]))
                            tensor_idx += 1
                        else:
                            items.append(repr(nm))
                    pos_args.append(f"[{', '.join(items)}]")
                else:
                    pos_args.append(repr(val))

            # Build kwargs
            kw_parts = []
            for k, v in raw_kwargs.items():
                if isinstance(v, tuple) and len(v) == 2 and v[0] == "__tensor__":
                    if tensor_idx < len(input_vars):
                        kw_parts.append(f"{k}={_sanitize_var(input_vars[tensor_idx])}")
                        tensor_idx += 1
                    else:
                        kw_parts.append(f"{k}={v[1]!r}")
                else:
                    kw_parts.append(f"{k}={v!r}")

            all_args = ", ".join(pos_args + kw_parts)
            result_var = _sanitize_var(instr.var)
            lines.append(f"    {result_var} = _F[{op_name!r}]({all_args})")
            if dps_var and isinstance(dps_var, str):
                lines.append(f"    {_sanitize_var(dps_var)} = {result_var}")

        elif isinstance(instr, RelaxOpInstr):
            v = _sanitize_var(instr.var)
            args = ", ".join(_sanitize_var(a) for a in instr.arg_vars)
            op = instr.op_name
            if op == "relax.multiply":
                lines.append(f"    {v} = {_sanitize_var(instr.arg_vars[0])} * {_sanitize_var(instr.arg_vars[1])}")
            elif op == "relax.add":
                lines.append(f"    {v} = {_sanitize_var(instr.arg_vars[0])} + {_sanitize_var(instr.arg_vars[1])}")
            elif op == "relax.power":
                lines.append(f"    {v} = {_sanitize_var(instr.arg_vars[0])} ** {_sanitize_var(instr.arg_vars[1])}")
            elif op == "relax.mean":
                # mean over last axis
                lines.append(f"    {v} = {_sanitize_var(instr.arg_vars[0])}.mean(dim=-1, keepdim=True)")
            elif op == "relax.astype":
                dtype = str(instr.attrs.get("dtype", "float16"))
                torch_dtype = {"float16": "torch.float16", "float32": "torch.float32",
                               "int32": "torch.int32", "int64": "torch.int64"}.get(dtype, f"torch.{dtype}")
                lines.append(f"    {v} = {_sanitize_var(instr.arg_vars[0])}.to({torch_dtype})")
            elif op == "relax.divide":
                lines.append(f"    {v} = {_sanitize_var(instr.arg_vars[0])} / {_sanitize_var(instr.arg_vars[1])}")
            elif op == "relax.sqrt":
                lines.append(f"    {v} = _torch.sqrt({_sanitize_var(instr.arg_vars[0])})")
            elif op == "relax.rsqrt":
                lines.append(f"    {v} = _torch.rsqrt({_sanitize_var(instr.arg_vars[0])})")
            else:
                short_name = op.replace("relax.", "")
                lines.append(f"    {v} = _torch.{short_name}({args})")

        elif isinstance(instr, ReshapeInstr):
            shape_str = _shape_code(instr.shape, sym_var_map, "(")
            lines.append(
                f"    {_sanitize_var(instr.var)} = {_sanitize_var(instr.input_var)}.reshape({shape_str})")

        elif isinstance(instr, AliasInstr):
            lines.append(f"    {_sanitize_var(instr.var)} = {_sanitize_var(instr.source_var)}")

        elif isinstance(instr, TupleInstr):
            elems = ", ".join(_sanitize_var(v) for v in instr.element_vars)
            lines.append(f"    {_sanitize_var(instr.var)} = ({elems},)")

        elif isinstance(instr, TupleGetItemInstr):
            lines.append(
                f"    {_sanitize_var(instr.var)} = {_sanitize_var(instr.tuple_var)}[{instr.index}]")

    # Return — clone outputs that may be views of the pre-allocated storage pool
    # to prevent the caller from holding references that block pool reuse.
    if len(output_vars) == 1:
        lines.append(f"    return {_sanitize_var(output_vars[0])}.clone()")
    else:
        outs = ", ".join(_sanitize_var(v) for v in output_vars)
        lines.append(f"    return ({outs},)")

    return "\n".join(lines)


_c_expr_counter = [0]


def _tir_expr_to_c_long(expr, sym_var_map, lines, dest):
    """Emit C code computing a TIR PrimExpr into ``dest`` (a C ``long`` variable).

    Handles the same expression set as ``_tir_expr_to_python``.
    Uses unique temp names to avoid shadowing in nested expressions.
    """
    a = lines.append
    if isinstance(expr, tir.IntImm):
        a(f"    {dest} = {int(expr)}L;")
    elif isinstance(expr, tir.Var):
        if expr.name in sym_var_map:
            pidx, didx = sym_var_map[expr.name]
            a(f"    {{ PyObject* _d = PyObject_CallMethod("
              f"PyList_GET_ITEM(inputs, {pidx}), \"size\", \"i\", {didx});")
            a(f"    {dest} = PyLong_AsLong(_d); Py_DECREF(_d); }}")
        else:
            a(f"    {dest} = 0; /* unresolved var {expr.name} */")
    elif isinstance(expr, tir.Cast):
        _tir_expr_to_c_long(expr.value, sym_var_map, lines, dest)
    elif isinstance(expr, (tir.Add, tir.Sub, tir.Mul, tir.Div,
                           tir.FloorDiv, tir.FloorMod,
                           tir.Max, tir.Min)):
        uid = _c_expr_counter[0]; _c_expr_counter[0] += 1
        lv, rv = f"_el{uid}", f"_er{uid}"
        a(f"    long {lv}, {rv};")
        _tir_expr_to_c_long(expr.a, sym_var_map, lines, lv)
        _tir_expr_to_c_long(expr.b, sym_var_map, lines, rv)
        op_map = {tir.Add: "+", tir.Sub: "-", tir.Mul: "*", tir.Div: "/",
                  tir.FloorDiv: "/", tir.FloorMod: "%"}
        op_str = op_map.get(type(expr))
        if op_str:
            a(f"    {dest} = {lv} {op_str} {rv};")
        elif isinstance(expr, tir.Max):
            a(f"    {dest} = ({lv} > {rv}) ? {lv} : {rv};")
        else:
            a(f"    {dest} = ({lv} < {rv}) ? {lv} : {rv};")
    elif isinstance(expr, tir.Select):
        uid = _c_expr_counter[0]; _c_expr_counter[0] += 1
        cv, tv, fv = f"_ec{uid}", f"_et{uid}", f"_ef{uid}"
        a(f"    long {cv}, {tv}, {fv};")
        _tir_expr_to_c_long(expr.condition, sym_var_map, lines, cv)
        _tir_expr_to_c_long(expr.true_value, sym_var_map, lines, tv)
        _tir_expr_to_c_long(expr.false_value, sym_var_map, lines, fv)
        a(f"    {dest} = {cv} ? {tv} : {fv};")
    elif isinstance(expr, (tir.LT, tir.LE, tir.GT, tir.GE, tir.EQ, tir.NE)):
        uid = _c_expr_counter[0]; _c_expr_counter[0] += 1
        lv, rv = f"_el{uid}", f"_er{uid}"
        a(f"    long {lv}, {rv};")
        _tir_expr_to_c_long(expr.a, sym_var_map, lines, lv)
        _tir_expr_to_c_long(expr.b, sym_var_map, lines, rv)
        cmp_map = {tir.LT: "<", tir.LE: "<=", tir.GT: ">",
                   tir.GE: ">=", tir.EQ: "==", tir.NE: "!="}
        a(f"    {dest} = ({lv} {cmp_map[type(expr)]} {rv});")
    else:
        try:
            a(f"    {dest} = {int(expr)}L;")
        except (TypeError, ValueError):
            a(f"    {dest} = 0; /* unsupported: {type(expr).__name__} */")


def _emit_c_shape_dims(shape, sym_var_map, lines, prefix):
    """Emit C code producing a list of PyObject* dims for a shape.

    Returns a list of C variable names (PyObject* PyLong values).
    The caller must Py_DECREF each when done.
    """
    a = lines.append
    dim_vars = []
    for di, s in enumerate(shape):
        dv = f"{prefix}_{di}"
        a(f"    long {dv}_v;")
        _tir_expr_to_c_long(s, sym_var_map, lines, f"{dv}_v")
        a(f"    PyObject* {dv} = PyLong_FromLong({dv}_v);")
        dim_vars.append(dv)
    return dim_vars


def _emit_c_source(instructions, param_names, output_vars, constants,
                   fallback_calls, sym_var_map, alloc_to_slot, pool_specs):
    """Generate C extension source for the wrapper (Python C API)."""
    # Build index maps: kernel_name → idx, fallback_name → idx, etc.
    kernel_names = []
    fallback_names = []
    for instr in instructions:
        if isinstance(instr, KernelCallInstr):
            if instr.kernel_name not in kernel_names:
                kernel_names.append(instr.kernel_name)
        elif isinstance(instr, TorchFallbackInstr):
            if instr.op_name not in fallback_names:
                fallback_names.append(instr.op_name)

    kernel_idx = {n: i for i, n in enumerate(kernel_names)}
    fallback_idx = {n: i for i, n in enumerate(fallback_names)}
    const_names = list(constants.keys())

    n_k = len(kernel_names)
    n_f = len(fallback_names)
    n_c = len(const_names)
    n_pool = len(pool_specs)

    lines = []
    a = lines.append

    # Header
    a("#define PY_SSIZE_T_CLEAN")
    a("#include <Python.h>")
    a("")
    a("#define CHK(x) do { if ((x) == NULL) { "
      "PyObject *_e, *_v, *_t; PyErr_Fetch(&_e, &_v, &_t); "
      "PyErr_Format(PyExc_RuntimeError, "
      "\"C wrapper failed at line %d: %S\", __LINE__, _v ? _v : Py_None); "
      "Py_XDECREF(_e); Py_XDECREF(_v); Py_XDECREF(_t); "
      "return NULL; } } while(0)")
    a("")
    a(f"static PyObject* _k[{max(n_k, 1)}];")
    a(f"static PyObject* _f[{max(n_f, 1)}];")
    a(f"static PyObject* _c[{max(n_c, 1)}];")
    a(f"static PyObject* _pool[{max(n_pool, 1)}];")
    a("static PyObject* _torch_empty = NULL;")
    a("static PyObject* _device = NULL;")
    a("")

    # Collect dtype strings needed for allocs and fallback kwargs
    dynamic_dtypes = set()
    for instr in instructions:
        if isinstance(instr, AllocStorageInstr):
            dynamic_dtypes.add("uint8")
        elif isinstance(instr, AllocTensorInstr):
            dynamic_dtypes.add(str(instr.dtype).replace("torch.", ""))
        elif isinstance(instr, AllocInstr) and instr.var not in alloc_to_slot:
            dynamic_dtypes.add(str(instr.dtype).replace("torch.", ""))
    for instr in instructions:
        if isinstance(instr, TorchFallbackInstr):
            call_info = fallback_calls.get(instr.op_name)
            if call_info:
                _, _, raw_kwargs = call_info
                for v in (raw_kwargs or {}).values():
                    if isinstance(v, torch.dtype):
                        dynamic_dtypes.add(str(v).replace("torch.", ""))

    for dt in sorted(dynamic_dtypes):
        a(f"static PyObject* _dtype_{dt} = NULL;")
    a("")

    # Init function
    a("static PyObject* tl_init(PyObject* self, PyObject* args) {")
    a("    PyObject *kd, *fd, *cd, *pl, *torch_mod, *dev;")
    a('    if (!PyArg_ParseTuple(args, "OOOOOO", &kd, &fd, &cd, &pl, &torch_mod, &dev)) return NULL;')
    a('    _torch_empty = PyObject_GetAttrString(torch_mod, "empty");')
    a("    _device = dev; Py_INCREF(_device);")
    for i, name in enumerate(kernel_names):
        a(f'    _k[{i}] = PyDict_GetItemString(kd, "{name}"); Py_XINCREF(_k[{i}]);')
    for i, name in enumerate(fallback_names):
        a(f'    _f[{i}] = PyDict_GetItemString(fd, "{name}"); Py_XINCREF(_f[{i}]);')
    for i, name in enumerate(const_names):
        a(f'    _c[{i}] = PyDict_GetItemString(cd, "{name}"); Py_XINCREF(_c[{i}]);')
    for i in range(n_pool):
        a(f"    _pool[{i}] = PyList_GET_ITEM(pl, {i}); Py_INCREF(_pool[{i}]);")
    # Resolve torch dtypes
    for dt in sorted(dynamic_dtypes):
        a(f'    _dtype_{dt} = PyObject_GetAttrString(torch_mod, "{dt}");')
    a("    Py_RETURN_NONE;")
    a("}")
    a("")

    # Collect all C variable names that will be used
    var_set = set()
    for pn in param_names:
        var_set.add(_sanitize_var(pn))
    for cn in const_names:
        var_set.add(_sanitize_var(cn))
    for instr in instructions:
        if hasattr(instr, "var"):
            var_set.add(_sanitize_var(instr.var))

    # Run function
    a("static PyObject* tl_run(PyObject* self, PyObject* args) {")
    a("    PyObject* inputs;")
    a('    if (!PyArg_ParseTuple(args, "O", &inputs)) return NULL;')
    a("")

    # Declare all variables as PyObject*
    for v in sorted(var_set):
        a(f"    PyObject* {v} = NULL;")
    a("    PyObject* _tmp = NULL;")
    a("    PyObject* _tmp_args = NULL;")
    a("    PyObject* _tmp_kwargs = NULL;")
    a("")

    # Unpack inputs — move CPU tensors to target device
    for i, pname in enumerate(param_names):
        v = _sanitize_var(pname)
        a(f"    {v} = PyList_GET_ITEM(inputs, {i});")
        a(f"    {{ PyObject* _dev = PyObject_GetAttrString({v}, \"device\");")
        a(f"    PyObject* _dt = PyObject_GetAttrString(_dev, \"type\");")
        a(f"    int _is_cuda = PyUnicode_CompareWithASCIIString(_dt, \"cuda\") == 0;")
        a(f"    Py_DECREF(_dt); Py_DECREF(_dev);")
        a(f"    if (!_is_cuda) {{ {v} = PyObject_CallMethod({v}, \"to\", \"O\", _device); CHK({v}); }} }}")

    # Bind constants
    for i, cname in enumerate(const_names):
        a(f"    {_sanitize_var(cname)} = _c[{i}];")
    a("")

    # Emit instructions
    _instr_id = 0
    for instr in instructions:
        _instr_id += 1
        if isinstance(instr, ConstantInstr):
            pass

        elif isinstance(instr, AllocStorageInstr):
            v = _sanitize_var(instr.var)
            if isinstance(instr.size, int):
                a(f"    /* alloc_storage: {instr.size} bytes */")
                a(f"    {{ PyObject* _sz = PyLong_FromLong({instr.size}L);")
            else:
                # Dynamic size — compute from symbolic dims
                a(f"    /* alloc_storage: dynamic */")
                a(f"    {{ long _sz_val;")
                _tir_expr_to_c_long(instr.size, sym_var_map, lines, "_sz_val")
                a(f"    PyObject* _sz = PyLong_FromLong(_sz_val);")
            a(f"    _tmp_args = PyTuple_Pack(1, _sz); Py_DECREF(_sz);")
            a(f"    _tmp_kwargs = PyDict_New();")
            a(f'    PyDict_SetItemString(_tmp_kwargs, "dtype", _dtype_uint8);')
            a(f'    PyDict_SetItemString(_tmp_kwargs, "device", _device);')
            a(f"    {v} = PyObject_Call(_torch_empty, _tmp_args, _tmp_kwargs);")
            a(f"    Py_DECREF(_tmp_args); Py_DECREF(_tmp_kwargs); CHK({v}); }}")

        elif isinstance(instr, AllocTensorInstr):
            v = _sanitize_var(instr.var)
            sv = _sanitize_var(instr.storage_var)
            dtype_str = str(instr.dtype).replace("torch.", "")
            elem_size = _DTYPE_BYTES.get(dtype_str, 2)
            # Compute numel for byte-slice end
            if all(isinstance(s, int) for s in instr.shape):
                numel = 1
                for s in instr.shape:
                    numel *= s
                end = instr.offset + elem_size * numel
                # storage[offset:end].view(dtype).reshape(shape)
                a(f"    /* alloc_tensor from {sv}[{instr.offset}:{end}] -> {dtype_str} {instr.shape} */")
                a(f"    {{ PyObject* _sl = PySlice_New(PyLong_FromLong({instr.offset}L),"
                  f" PyLong_FromLong({end}L), Py_None);")
                a(f"    PyObject* _sliced = PyObject_GetItem({sv}, _sl); Py_DECREF(_sl); CHK(_sliced);")
                a(f'    PyObject* _viewed = PyObject_CallMethod(_sliced, "view", "O", _dtype_{dtype_str});')
                a(f"    Py_DECREF(_sliced); CHK(_viewed);")
                # Build shape tuple
                n_dims = len(instr.shape)
                dim_strs = ", ".join(f"PyLong_FromLong({s}L)" for s in instr.shape)
                a(f"    PyObject* _shape = PyTuple_Pack({n_dims}, {dim_strs});")
                a(f'    {v} = PyObject_CallMethod(_viewed, "reshape", "O", _shape);')
                a(f"    Py_DECREF(_viewed); Py_DECREF(_shape); CHK({v}); }}")
            else:
                # Dynamic shapes — compute numel and dims at runtime
                a(f"    /* alloc_tensor (dynamic) from {sv} -> {dtype_str} */")
                a(f"    {{ long _numel = 1;")
                for s in instr.shape:
                    a(f"    {{ long _dv;")
                    _tir_expr_to_c_long(s, sym_var_map, lines, "_dv")
                    a(f"    _numel *= _dv; }}")
                a(f"    PyObject* _start = PyLong_FromLong({instr.offset}L);")
                a(f"    PyObject* _end = PyLong_FromLong({instr.offset}L + {elem_size}L * _numel);")
                a(f"    PyObject* _sl = PySlice_New(_start, _end, Py_None);")
                a(f"    Py_DECREF(_start); Py_DECREF(_end);")
                a(f"    PyObject* _sliced = PyObject_GetItem({sv}, _sl); Py_DECREF(_sl); CHK(_sliced);")
                a(f'    PyObject* _viewed = PyObject_CallMethod(_sliced, "view", "O", _dtype_{dtype_str});')
                a(f"    Py_DECREF(_sliced); CHK(_viewed);")
                # Build shape tuple with runtime dims
                dim_vars_c = []
                for di, s in enumerate(instr.shape):
                    dv = f"_adim_{v}_{di}"
                    a(f"    long {dv}_v;")
                    _tir_expr_to_c_long(s, sym_var_map, lines, f"{dv}_v")
                    a(f"    PyObject* {dv} = PyLong_FromLong({dv}_v);")
                    dim_vars_c.append(dv)
                n_dims = len(dim_vars_c)
                a(f"    PyObject* _shape = PyTuple_Pack({n_dims}, {', '.join(dim_vars_c)});")
                for dv in dim_vars_c:
                    a(f"    Py_DECREF({dv});")
                a(f'    {v} = PyObject_CallMethod(_viewed, "reshape", "O", _shape);')
                a(f"    Py_DECREF(_viewed); Py_DECREF(_shape); CHK({v}); }}")

        elif isinstance(instr, AllocInstr):
            v = _sanitize_var(instr.var)
            if instr.var in alloc_to_slot:
                slot = alloc_to_slot[instr.var]
                a(f"    {v} = _pool[{slot}];")
            else:
                # Dynamic or output alloc — call torch.empty at runtime
                dtype_str = str(instr.dtype).replace("torch.", "")
                dim_vars = _emit_c_shape_dims(instr.shape, sym_var_map, lines, "_dim")
                n_dims = len(dim_vars)
                a(f"    _tmp_args = PyTuple_Pack({n_dims}, {', '.join(dim_vars)});")
                for dv in dim_vars:
                    a(f"    Py_DECREF({dv});")
                a(f"    _tmp_kwargs = PyDict_New();")
                a(f'    PyDict_SetItemString(_tmp_kwargs, "dtype", _dtype_{dtype_str});')
                a(f'    PyDict_SetItemString(_tmp_kwargs, "device", _device);')
                a(f"    {v} = PyObject_Call(_torch_empty, _tmp_args, _tmp_kwargs);")
                a(f"    Py_DECREF(_tmp_args); Py_DECREF(_tmp_kwargs);")
                a(f"    CHK({v});")

        elif isinstance(instr, KernelCallInstr):
            idx = kernel_idx[instr.kernel_name]
            # Ensure contiguity (+ flatten to 1D if kernel expects it)
            contig_vars = []
            for ai_k, arg_v in enumerate(instr.arg_vars):
                sv = _sanitize_var(arg_v) if isinstance(arg_v, str) else "Py_None"
                if isinstance(arg_v, str):
                    cv = f"_carg_{_instr_id}_{ai_k}"
                    if instr.flatten_args:
                        a(f"    PyObject* {cv};")
                        a(f"    {{ PyObject* _ct = PyObject_CallMethod({sv}, \"contiguous\", NULL); CHK(_ct);")
                        a(f"    {cv} = PyObject_CallMethod(_ct, \"view\", \"l\", (long)-1);")
                        a(f"    Py_DECREF(_ct); CHK({cv}); }}")
                    else:
                        a(f"    PyObject* {cv} = {sv};")
                        a(f"    {{PyObject* _ic = PyObject_CallMethod({sv}, \"is_contiguous\", NULL);")
                        a(f"     if (_ic && _ic == Py_False) {{ Py_DECREF(_ic); {cv} = PyObject_CallMethod({sv}, \"contiguous\", NULL); CHK({cv}); }}")
                        a(f"     else {{ Py_XDECREF(_ic); }}}}")
                    contig_vars.append(cv)
                else:
                    contig_vars.append(sv)
            n = len(contig_vars)
            pack_args = ", ".join(contig_vars)
            a(f"    _tmp_args = PyTuple_Pack({n}, {pack_args});")
            a(f"    _tmp = PyObject_Call(_k[{idx}], _tmp_args, NULL);")
            a(f"    Py_DECREF(_tmp_args);")
            # Decref any contiguous copies we created
            for ai_k, arg_v in enumerate(instr.arg_vars):
                if isinstance(arg_v, str):
                    sv = _sanitize_var(arg_v)
                    cv = f"_carg_{_instr_id}_{ai_k}"
                    a(f"    if ({cv} != {sv}) Py_DECREF({cv});")
            a(f"    CHK(_tmp); Py_DECREF(_tmp);")

        elif isinstance(instr, TorchFallbackInstr):
            fb_i = fallback_idx[instr.op_name]
            call_info = fallback_calls.get(instr.op_name)
            if call_info is None:
                a(f'    PyErr_SetString(PyExc_RuntimeError, "No fallback for {instr.op_name}");')
                a(f"    return NULL;")
                continue

            _, arg_template, raw_kwargs = call_info
            result_var = _sanitize_var(instr.var)

            # DPS convention: last arg is the output buffer when
            # it was emitted by call_dps_packed (the codegen sees the
            # alloc → call → alias pattern).  Detect by checking if the
            # number of arg_vars exceeds the template's tensor count.
            n_template_tensors = sum(1 for tag, _ in arg_template if tag is True)
            if instr.arg_vars and len(instr.arg_vars) > n_template_tensors:
                input_vars = instr.arg_vars[:-1]
                dps_var = instr.arg_vars[-1]
            else:
                input_vars = list(instr.arg_vars)
                dps_var = None

            # Build positional args — each heap-allocated scalar gets a
            # unique temp so multiple scalars in one call don't clobber _tmp.
            tensor_idx = 0
            pos_items = []
            temps_to_decref = []  # temp var names to Py_DECREF after Pack

            for ai, (tag, val) in enumerate(arg_template):
                if tag is True and tensor_idx < len(input_vars):
                    pos_items.append(_sanitize_var(input_vars[tensor_idx]))
                    tensor_idx += 1
                elif tag == "list":
                    list_items = []
                    for nm in val:
                        if isinstance(nm, str) and tensor_idx < len(input_vars):
                            list_items.append(_sanitize_var(input_vars[tensor_idx]))
                            tensor_idx += 1
                    tv = f"_slist_{result_var}_{ai}"
                    n_list = len(list_items)
                    a(f"    PyObject* {tv} = PyList_New({n_list});")
                    for li, item in enumerate(list_items):
                        a(f"    Py_INCREF({item}); PyList_SET_ITEM({tv}, {li}, {item});")
                    pos_items.append(tv)
                    temps_to_decref.append(tv)
                else:
                    # Scalar — build Python object with unique name
                    if val is None:
                        pos_items.append("Py_None")
                    elif isinstance(val, bool):
                        pos_items.append("Py_True" if val else "Py_False")
                    elif isinstance(val, (int, float, str)):
                        tv = f"_s_{result_var}_{ai}"
                        if isinstance(val, int):
                            a(f"    PyObject* {tv} = PyLong_FromLong({val}L);")
                        elif isinstance(val, float):
                            a(f"    PyObject* {tv} = PyFloat_FromDouble({val});")
                        else:
                            escaped = val.replace('\\', '\\\\').replace('"', '\\"')
                            a(f'    PyObject* {tv} = PyUnicode_FromString("{escaped}");')
                        pos_items.append(tv)
                        temps_to_decref.append(tv)
                    elif isinstance(val, list) and len(val) == 0:
                        tv = f"_s_{result_var}_{ai}"
                        a(f"    PyObject* {tv} = PyList_New(0);")
                        pos_items.append(tv)
                        temps_to_decref.append(tv)
                    else:
                        pos_items.append("Py_None")

            n_pos = len(pos_items)
            pack_pos = ", ".join(pos_items)
            a(f"    _tmp_args = PyTuple_Pack({n_pos}, {pack_pos});")
            for tv in temps_to_decref:
                a(f"    Py_DECREF({tv});")

            # Build kwargs
            has_kwargs = bool(raw_kwargs)
            if has_kwargs:
                a(f"    _tmp_kwargs = PyDict_New();")
                for k, v in raw_kwargs.items():
                    if isinstance(v, tuple) and len(v) == 2 and v[0] == "__tensor__":
                        if tensor_idx < len(input_vars):
                            tv = _sanitize_var(input_vars[tensor_idx])
                            tensor_idx += 1
                            a(f'    PyDict_SetItemString(_tmp_kwargs, "{k}", {tv});')
                    elif v is None:
                        a(f'    PyDict_SetItemString(_tmp_kwargs, "{k}", Py_None);')
                    elif isinstance(v, bool):
                        bval = "Py_True" if v else "Py_False"
                        a(f'    PyDict_SetItemString(_tmp_kwargs, "{k}", {bval});')
                    elif isinstance(v, (int, float)):
                        if isinstance(v, int):
                            a(f'    _tmp = PyLong_FromLong({v}L);')
                        else:
                            a(f'    _tmp = PyFloat_FromDouble({v});')
                        a(f'    PyDict_SetItemString(_tmp_kwargs, "{k}", _tmp);')
                        a(f'    Py_DECREF(_tmp);')
                    elif isinstance(v, str):
                        escaped = v.replace('\\', '\\\\').replace('"', '\\"')
                        a(f'    _tmp = PyUnicode_FromString("{escaped}");')
                        a(f'    PyDict_SetItemString(_tmp_kwargs, "{k}", _tmp);')
                        a(f'    Py_DECREF(_tmp);')
                    elif isinstance(v, torch.dtype):
                        dtype_str = str(v).replace("torch.", "")
                        a(f'    PyDict_SetItemString(_tmp_kwargs, "{k}", _dtype_{dtype_str});')
                    elif isinstance(v, torch.device):
                        a(f'    PyDict_SetItemString(_tmp_kwargs, "{k}", _device);')
                    else:
                        # Complex object — use repr and Py_None as fallback
                        a(f'    PyDict_SetItemString(_tmp_kwargs, "{k}", Py_None);')

            if has_kwargs:
                a(f"    {result_var} = PyObject_Call(_f[{fb_i}], _tmp_args, _tmp_kwargs);")
                a(f"    Py_DECREF(_tmp_args); Py_DECREF(_tmp_kwargs);")
            else:
                a(f"    {result_var} = PyObject_Call(_f[{fb_i}], _tmp_args, NULL);")
                a(f"    Py_DECREF(_tmp_args);")
            a(f"    CHK({result_var});")

            if dps_var and isinstance(dps_var, str):
                dv = _sanitize_var(dps_var)
                a(f"    {dv} = {result_var};")

        elif isinstance(instr, ReshapeInstr):
            v = _sanitize_var(instr.var)
            src = _sanitize_var(instr.input_var)
            dim_vars = _emit_c_shape_dims(instr.shape, sym_var_map, lines, f"_rdim_{v}")
            n_dims = len(dim_vars)
            a(f"    _tmp = PyTuple_Pack({n_dims}, {', '.join(dim_vars)});")
            for dv in dim_vars:
                a(f"    Py_DECREF({dv});")
            a(f'    {v} = PyObject_CallMethod({src}, "reshape", "O", _tmp);')
            a(f"    Py_DECREF(_tmp);")
            a(f"    CHK({v});")

        elif isinstance(instr, AliasInstr):
            a(f"    {_sanitize_var(instr.var)} = {_sanitize_var(instr.source_var)};")

        elif isinstance(instr, TupleInstr):
            v = _sanitize_var(instr.var)
            elems = ", ".join(_sanitize_var(e) for e in instr.element_vars)
            n = len(instr.element_vars)
            a(f"    {v} = PyTuple_Pack({n}, {elems});")

        elif isinstance(instr, TupleGetItemInstr):
            v = _sanitize_var(instr.var)
            tv = _sanitize_var(instr.tuple_var)
            a(f"    {v} = PyTuple_GET_ITEM({tv}, {instr.index});")

    # Return
    a("")
    if len(output_vars) == 1:
        ov = _sanitize_var(output_vars[0])
        a(f"    Py_INCREF({ov});")
        a(f"    return {ov};")
    else:
        outs = ", ".join(_sanitize_var(v) for v in output_vars)
        a(f"    return PyTuple_Pack({len(output_vars)}, {outs});")

    a("}")
    a("")

    # Module definition
    a("static PyMethodDef _methods[] = {")
    a('    {"init", tl_init, METH_VARARGS, ""},')
    a('    {"run", tl_run, METH_VARARGS, ""},')
    a("    {NULL, NULL, 0, NULL}")
    a("};")
    a("")
    a("static PyModuleDef _moduledef = {")
    a('    PyModuleDef_HEAD_INIT, "tl_wrapper", NULL, -1, _methods')
    a("};")
    a("")
    a("PyMODINIT_FUNC PyInit_tl_wrapper(void) {")
    a("    return PyModule_Create(&_moduledef);")
    a("}")

    return "\n".join(lines), kernel_names, fallback_names, const_names


def _compile_c_extension(c_source):
    """Compile C source into a Python extension module and load it."""
    import importlib.util
    import subprocess
    import sysconfig
    import tempfile

    tmpdir = tempfile.mkdtemp(prefix="tilelang_cwrap_")
    c_path = os.path.join(tmpdir, "tl_wrapper.c")
    so_path = os.path.join(tmpdir, "tl_wrapper" + sysconfig.get_config_var("EXT_SUFFIX"))

    with open(c_path, "w") as f:
        f.write(c_source)

    py_include = sysconfig.get_path("include")

    cmd = [
        "gcc", "-O2", "-shared", "-fPIC", "-w",
        f"-I{py_include}",
        "-o", so_path, c_path,
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"C wrapper compilation failed:\n{result.stderr}")

    spec = importlib.util.spec_from_file_location("tl_wrapper", so_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def generate_wrapper(
    mod: tvm.IRModule,
    compiled_kernels: dict[str, runtime.Module],
    fallback_calls: dict[str, tuple] = None,
) -> callable:
    """Generate a Python callable that executes compiled TileLang kernels.

    Walks the lowered Relax IR (after CallTIRRewrite) to produce an instruction
    sequence, then returns a closure that executes these instructions using
    torch tensors and compiled kernels via DLPack.

    Parameters
    ----------
    mod : tvm.IRModule
        The Relax module after the full pipeline (including CallTIRRewrite).
    compiled_kernels : dict[str, callable]
        Mapping from TIR function name to compiled JITKernel.
    fallback_calls : dict[str, tuple], optional
        Mapping from op name to (callable, arg_template, kwargs).

    Returns
    -------
    callable
        A function that takes torch tensors and returns torch tensors.
    """
    fallback_calls = fallback_calls or {}
    instructions, param_names, output_vars, constants = _compile_instructions(mod)
    instructions = _eliminate_dead_instructions(instructions, output_vars)

    # Build symbolic var → (param_index, dim_index) map
    main_func = None
    for gvar, func in mod.functions.items():
        if isinstance(func, relax.Function):
            main_func = func
            break

    sym_var_map = {}
    if main_func:
        for pidx, param in enumerate(main_func.params):
            sinfo = param.struct_info_
            if sinfo and isinstance(sinfo, relax.TensorStructInfo) and sinfo.shape:
                for didx, dim in enumerate(sinfo.shape.values):
                    if isinstance(dim, tir.Var):
                        sym_var_map[dim.name] = (pidx, didx)

    # Move constants to GPU once
    _const_device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    gpu_constants = {k: v.to(device=_const_device) for k, v in constants.items()}

    # Build fallback op_fn lookup: op_name → callable.
    # For call_method nodes, op_fn is a string (method name) — wrap it
    # so the first positional arg is used as self.
    fb_fns = {}
    for op_name, (op_fn, _, _) in fallback_calls.items():
        if isinstance(op_fn, str):
            method_name = op_fn
            fb_fns[op_name] = lambda *args, _m=method_name, **kw: getattr(args[0], _m)(*args[1:], **kw)
        else:
            fb_fns[op_name] = op_fn

    # Memory planning: if StaticPlanBlockMemory produced AllocStorageInstr,
    # pre-allocate static storages once (reused across calls).
    # Fall back to our own pool for legacy AllocInstr.
    has_storage_instrs = any(isinstance(i, AllocStorageInstr) for i in instructions)
    if has_storage_instrs:
        alloc_to_slot = {}
        pool_specs = []
        # Pre-allocate static-size storages
        storage_pool = {}
        for instr in instructions:
            if isinstance(instr, AllocStorageInstr) and isinstance(instr.size, int):
                storage_pool[instr.var] = torch.empty(
                    instr.size, dtype=torch.uint8, device=_const_device)
    else:
        alloc_to_slot, pool_specs = _plan_memory_reuse(instructions, output_vars)
        storage_pool = {}

    pool = [
        torch.empty(shape, dtype=dtype, device=_const_device)
        for shape, dtype in pool_specs
    ]
    n_allocs = sum(1 for i in instructions if isinstance(i, AllocInstr))
    n_storages = sum(1 for i in instructions if isinstance(i, AllocStorageInstr))
    if has_storage_instrs:
        logger.debug("Memory managed by StaticPlanBlockMemory: %d storages", n_storages)
    elif n_allocs:
        logger.debug("Memory pool: %d slots for %d allocations (%.0f%% reuse)",
                     len(pool), n_allocs,
                     (1 - len(pool) / max(n_allocs, 1)) * 100)

    use_c = not os.environ.get("TILELANG_USE_PYTHON_WRAPPER")
    c_module = None

    if use_c:
        try:
            c_source, c_knames, c_fnames, c_cnames = _emit_c_source(
                instructions, param_names, output_vars, constants,
                fallback_calls, sym_var_map, alloc_to_slot, pool_specs)

            if os.environ.get("TILELANG_DUMP_WRAPPER"):
                dump_path = os.environ["TILELANG_DUMP_WRAPPER"]
                with open(dump_path, "w") as f:
                    f.write(c_source)
                logger.info("C wrapper source dumped to %s", dump_path)

            c_module = _compile_c_extension(c_source)

            # Init: pass kernel/fallback/constant dicts + pool list + torch + device
            k_dict = {n: compiled_kernels[n] for n in c_knames}
            f_dict = {n: fb_fns[n] for n in c_fnames if n in fb_fns}
            c_dict = {n: gpu_constants[n] for n in c_cnames if n in gpu_constants}
            c_module.init(k_dict, f_dict, c_dict, pool, torch, _const_device)
            logger.debug("C extension wrapper loaded")
        except Exception:
            logger.info("C wrapper failed, falling back to Python", exc_info=True)
            c_module = None

    if c_module is not None:
        def wrapper(*inputs):
            tensor_inputs = [inp for inp in inputs if isinstance(inp, torch.Tensor)]
            return c_module.run(tensor_inputs)
        return wrapper

    # Fallback: Python wrapper
    source = _emit_python_source(
        instructions, param_names, output_vars, constants,
        compiled_kernels, fallback_calls, sym_var_map, alloc_to_slot)

    if os.environ.get("TILELANG_DUMP_WRAPPER"):
        dump_path = os.environ["TILELANG_DUMP_WRAPPER"]
        with open(dump_path, "w") as f:
            f.write(source)
        logger.info("Python wrapper source dumped to %s", dump_path)

    code_globals = {"_torch": torch, "torch": torch, "device": torch.device}
    exec(compile(source, "<tilelang_wrapper>", "exec"), code_globals)
    compiled_fn = code_globals["_compiled_wrapper"]

    # Merge both pool types into one dict for the wrapper
    merged_pool = storage_pool.copy()
    # Legacy pool uses integer indices
    for i, t in enumerate(pool):
        merged_pool[i] = t

    def wrapper(*inputs):
        tensor_inputs = [inp for inp in inputs if isinstance(inp, torch.Tensor)]
        return compiled_fn(tensor_inputs,
                           tensor_inputs[0].device if tensor_inputs else _const_device,
                           gpu_constants, compiled_kernels, fb_fns, merged_pool)

    return wrapper
