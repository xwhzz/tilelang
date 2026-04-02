"""Re-derive tensor shapes to propagate symbolic dims through the IR.

After from_fx, many intermediate bindings have concrete shapes because
from_fx's op converters read shapes from FakeTensor metadata instead
of using Relax type inference.  This pass re-emits each binding through
BlockBuilder, which re-runs type inference and propagates symbolic
dimensions from the function parameters.
"""

from tilelang import tvm
from tvm import relax, tir, ir


def refine_shapes(mod: tvm.IRModule) -> tvm.IRModule:
    """Re-derive struct info for all bindings to propagate symbolic shapes."""
    main_func = None
    main_gvar = None
    for gvar, func in mod.functions.items():
        if isinstance(func, relax.Function) and gvar.name_hint == "main":
            main_func = func
            main_gvar = gvar
            break
    if main_func is None:
        return mod

    body = main_func.body
    if not isinstance(body, relax.SeqExpr) or not body.blocks:
        return mod

    bb = relax.BlockBuilder()
    with bb.function("main", main_func.params):
        with bb.dataflow():
            env = {}
            for p in main_func.params:
                env[p] = p

            for block in body.blocks:
                for binding in block.bindings:
                    if isinstance(binding, relax.VarBinding):
                        new_value = _remap(binding.value, env)
                        # Re-emit through BlockBuilder to trigger type inference
                        new_var = bb.emit(new_value, name_hint=binding.var.name_hint)
                        env[binding.var] = new_var

            output = _remap(body.body, env)
            bb.emit_output(output)
        bb.emit_func_output(output)

    new_mod = bb.get()
    # Preserve all non-main functions
    existing = {g.name_hint for g in new_mod.functions}
    for gvar, func in mod.functions.items():
        if gvar.name_hint != "main" and gvar.name_hint not in existing:
            new_mod[gvar] = func
    return new_mod


def _remap(expr, env):
    """Replace Var references using env mapping."""
    if isinstance(expr, relax.Var):
        return env.get(expr, expr)
    elif isinstance(expr, relax.Call):
        new_args = [_remap(a, env) for a in expr.args]
        new_op = expr.op
        if isinstance(new_op, relax.Var):
            new_op = env.get(new_op, new_op)
        # For call_tir / call_dps_packed, keep sinfo_args (they define output shape)
        if isinstance(new_op, ir.Op) and new_op.name in ("relax.call_tir", "relax.call_dps_packed",
                                                          "relax.builtin.alloc_tensor"):
            if len(new_args) >= 2 and isinstance(new_args[1], relax.Tuple):
                new_args[1] = relax.Tuple([env.get(f, f) if isinstance(f, relax.Var) else f
                                           for f in new_args[1].fields])
            return relax.Call(new_op, new_args, expr.attrs, expr.sinfo_args, expr.span)
        # For pure Relax ops (multiply, astype, etc.), drop sinfo_args
        # so BlockBuilder re-infers the output struct info from inputs.
        return relax.Call(new_op, new_args, expr.attrs)
    elif isinstance(expr, relax.Tuple):
        return relax.Tuple([_remap(f, env) for f in expr.fields], expr.span)
    elif isinstance(expr, relax.TupleGetItem):
        return relax.TupleGetItem(_remap(expr.tuple_value, env), expr.index, expr.span)
    elif isinstance(expr, relax.ShapeExpr):
        return expr
    elif isinstance(expr, relax.Constant):
        return expr
    return expr
