"""Fuse elemwise producers with skip connections into reduction groups.

After FuseOps, patterns like RMSNorm produce three separate groups:

    %v1 = call_tir(cast, %x)           # elemwise
    %v2 = call_tir(power, %v1)          # elemwise
    %out = fused_group(%v2, %v1, %w)    # reduction group

v1 has a skip connection: consumed by both power (before reduction) and
multiply (after reduction inside the group). FuseOps can't merge them.

This pass inlines the elemwise producers (cast, power) into the reduction
group, producing a single fused function:

    %out = fused_cast_power_group(%x, %w)

This eliminates separate kernel launches for the inlined ops.
"""

from tilelang import tvm as tvm
from tvm import relax, tir, ir


@tvm.transform.module_pass(opt_level=0, name="FuseSkipReduction")
class FuseSkipReduction:
    """Merge elemwise TIR producers into their reduction-group consumers
    when the producer output is used both directly and transitively."""

    def transform_module(self, mod: tvm.IRModule, _ctx) -> tvm.IRModule:
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

        # Collect all bindings
        bindings = []
        for block in body.blocks:
            for binding in block.bindings:
                if isinstance(binding, relax.VarBinding):
                    bindings.append(binding)

        # Build var → binding index, var → producer info
        var_to_idx = {}  # var_name → binding index
        for i, b in enumerate(bindings):
            var_to_idx[b.var] = i

        # Build consumer map: var → list of binding indices that use it
        var_consumers = {}  # var → set of binding indices
        for i, b in enumerate(bindings):
            for used_var in _collect_used_vars(b.value):
                var_consumers.setdefault(used_var, set()).add(i)

        # Find inlineable patterns
        # Pattern: P1(x) → v1, P2(v1) → v2, C(v2, v1, ...) where C is Primitive
        inline_plan = []  # list of (producers_to_inline, consumer_idx, new_args_mapping)

        for i, b in enumerate(bindings):
            value = b.value
            if not isinstance(value, relax.Call):
                continue
            # Check if this is a call to a Primitive Relax Function
            if not isinstance(value.op, ir.GlobalVar):
                continue
            callee = mod[value.op]
            if not isinstance(callee, relax.Function):
                continue
            if not callee.attrs or not callee.attrs.get("Primitive"):
                continue

            # This is a Primitive group call. Find chains of elemwise
            # producers that can be inlined into this call.
            args = list(value.args)
            chains = _find_inlineable_chains(i, args, bindings, var_to_idx,
                                             var_consumers, mod)
            if chains:
                inline_plan.append((i, chains))

        if not inline_plan:
            return mod

        # Execute the inline plan
        return _apply_inline_plan(mod, main_gvar, main_func, bindings,
                                  inline_plan, body)


def _collect_used_vars(expr):
    """Collect all Var references in a Relax expression."""
    return set(relax.analysis.free_vars(expr))


def _is_standalone_call_tir(value, mod):
    """Check if a value is a call_tir to a standalone elemwise TIR function."""
    if not isinstance(value, relax.Call):
        return False
    if not isinstance(value.op, ir.Op):
        return False
    if value.op.name != "relax.call_tir":
        return False
    # The first arg should be a GlobalVar pointing to a TIR PrimFunc
    if len(value.args) < 1:
        return False
    gvar = value.args[0]
    if not isinstance(gvar, ir.GlobalVar):
        return False
    func = mod[gvar]
    if not isinstance(func, tir.PrimFunc):
        return False
    # Check pattern: must be elemwise or injective (pattern <= 2)
    pattern = func.attrs.get("op_pattern", -1)
    if isinstance(pattern, tvm.tir.IntImm):
        pattern = int(pattern)
    return pattern >= 0 and pattern <= 2  # kElemWise, kBroadcast, kInjective


def _find_inlineable_chains(consumer_idx, consumer_args, bindings,
                            var_to_idx, var_consumers, mod):
    """Find chains of elemwise producers that can be inlined into the consumer.

    Returns list of (arg_idx, [producer_binding_indices_in_order]) chains.
    Each chain is a sequence of bindings whose outputs feed exclusively
    into the consumer (or into each other within the chain).
    """
    chains = []
    visited = set()

    for arg_idx, arg in enumerate(consumer_args):
        if not isinstance(arg, relax.Var):
            continue
        if arg not in var_to_idx:
            continue

        # Walk backward from this arg to find the chain
        chain = []
        current = arg
        while True:
            if current not in var_to_idx:
                break
            prod_idx = var_to_idx[current]
            if prod_idx in visited:
                break
            prod_binding = bindings[prod_idx]
            prod_value = prod_binding.value

            if not _is_standalone_call_tir(prod_value, mod):
                break

            # Check: this producer's output must ONLY be consumed by
            # the consumer call OR by other producers in this chain
            consumers = var_consumers.get(prod_binding.var, set())
            allowed = {consumer_idx} | set(chain)
            if not consumers.issubset(allowed | {prod_idx}):
                break

            chain.append(prod_idx)
            visited.add(prod_idx)

            # Continue walking backward through the producer's inputs
            call_args = prod_value.args[1]  # Tuple of actual inputs
            if isinstance(call_args, relax.Tuple) and len(call_args.fields) == 1:
                inp = call_args.fields[0]
                if isinstance(inp, relax.Var):
                    current = inp
                    continue
            break

        if chain:
            chains.append((arg_idx, list(reversed(chain))))

    return chains


def _apply_inline_plan(mod, main_gvar, main_func, bindings, inline_plan, body):
    """Apply the inline plan: merge producers into consumer functions."""
    bb = relax.BlockBuilder()

    # Collect all producer binding indices to remove
    remove_indices = set()
    # Map: consumer binding index → (new callee gvar, new args)
    rewrites = {}

    for consumer_idx, chains in inline_plan:
        consumer_binding = bindings[consumer_idx]
        consumer_call = consumer_binding.value
        consumer_gvar = consumer_call.op
        consumer_func = mod[consumer_gvar]

        all_prod_indices = []
        for arg_idx, prod_chain in chains:
            all_prod_indices.extend(prod_chain)

        # Create new fused function by inlining producers into the consumer
        new_func, new_arg_map = _inline_producers_into_func(
            mod, consumer_func, consumer_call.args, chains, bindings)

        if new_func is None:
            continue

        # Register new function
        new_name = consumer_gvar.name_hint
        for _, chain in chains:
            for idx in chain:
                prod_name = bindings[idx].value.args[0].name_hint
                new_name = prod_name + "_" + new_name

        new_gvar = bb.add_func(new_func, new_name)

        remove_indices.update(all_prod_indices)
        rewrites[consumer_idx] = (new_gvar, new_arg_map)

    if not rewrites:
        return mod

    # Rebuild the main function with inlined bindings removed
    with bb.function("main", main_func.params):
        with bb.dataflow():
            env = {}
            for p in main_func.params:
                env[p] = p

            output_expr = None
            for i, binding in enumerate(bindings):
                if i in remove_indices:
                    # Skip inlined producers
                    continue

                if i in rewrites:
                    new_gvar, new_arg_map = rewrites[i]
                    # Build new args from the mapping
                    new_args = [env.get(a, a) if isinstance(a, relax.Var) else a
                                for a in new_arg_map]
                    new_call = relax.Call(
                        new_gvar, new_args,
                        sinfo_args=binding.value.sinfo_args if hasattr(binding.value, 'sinfo_args') else [],
                    )
                    var = bb.emit(new_call, name_hint=binding.var.name_hint)
                else:
                    # Remap args through env
                    value = _remap_vars(binding.value, env)
                    var = bb.emit(value, name_hint=binding.var.name_hint)

                env[binding.var] = var

            # Emit output
            output = _remap_vars(body.body, env)
            bb.emit_output(output)

        bb.emit_func_output(output)

    new_mod = bb.get()
    # Copy over all non-main functions
    existing_names = {g.name_hint for g in new_mod.functions}
    for gvar, func in mod.functions.items():
        if gvar.name_hint != "main" and gvar.name_hint not in existing_names:
            new_mod[gvar] = func

    return new_mod


def _inline_producers_into_func(mod, consumer_func, consumer_args, chains, bindings):
    """Create a new Relax function with producers inlined as call_tir.

    The inlined call_tir ops are prepended to the consumer's body so that
    FuseTIR (which runs after this pass) merges everything into one TIR
    PrimFunc.

    Returns (new_function, new_call_args) or (None, None) on failure.
    """
    # Collect all producer indices and map output vars back to them
    all_prod_indices = set()
    for _, chain in chains:
        all_prod_indices.update(chain)

    prod_var_to_idx = {}  # producer output var → producer binding index
    for idx in all_prod_indices:
        prod_var_to_idx[bindings[idx].var] = idx

    # Which consumer args are chain-produced intermediates?
    param_to_prod = {}  # consumer param position → producer binding index
    for i, arg in enumerate(consumer_args):
        if isinstance(arg, relax.Var) and arg in prod_var_to_idx:
            param_to_prod[i] = prod_var_to_idx[arg]

    if not param_to_prod:
        return None, None

    # External inputs: chain inputs not produced within the chain
    chain_produced = set(prod_var_to_idx.keys())
    external_inputs = []
    seen_ext = set()
    for _, chain in chains:
        for prod_idx in chain:
            call = bindings[prod_idx].value
            input_tuple = call.args[1]
            fields = (input_tuple.fields if isinstance(input_tuple, relax.Tuple)
                      else [input_tuple])
            for f in fields:
                if isinstance(f, relax.Var) and f not in chain_produced and f not in seen_ext:
                    external_inputs.append(f)
                    seen_ext.add(f)

    # --- Build new parameter list ---
    # new params = external chain inputs + non-replaced consumer params
    new_params = []
    new_call_args = []  # actual args at the call site in main
    ext_to_param = {}   # original external var → new function param

    for ext_var in external_inputs:
        p = relax.Var("p_" + ext_var.name_hint, ext_var.struct_info_)
        new_params.append(p)
        new_call_args.append(ext_var)
        ext_to_param[ext_var] = p

    cidx_to_param = {}  # consumer param index → new param
    for i, old_param in enumerate(consumer_func.params):
        if i not in param_to_prod:
            p = relax.Var(old_param.name_hint, old_param.struct_info_)
            new_params.append(p)
            new_call_args.append(consumer_args[i])
            cidx_to_param[i] = p

    # --- Build new function body via BlockBuilder ---
    bb = relax.BlockBuilder()
    call_tir_op = tvm.ir.Op.get("relax.call_tir")

    with bb.function("fused_inline", new_params):
        with bb.dataflow():
            # 1) Emit inlined producer call_tir calls
            prod_to_var = {}  # prod binding index → new Var inside function

            for _, chain in chains:
                for prod_idx in chain:
                    call = bindings[prod_idx].value
                    tir_gvar = call.args[0]
                    input_tuple = call.args[1]
                    fields = (input_tuple.fields
                              if isinstance(input_tuple, relax.Tuple)
                              else [input_tuple])

                    # Remap each input field
                    new_fields = []
                    for f in fields:
                        if not isinstance(f, relax.Var):
                            new_fields.append(f)
                        elif f in ext_to_param:
                            new_fields.append(ext_to_param[f])
                        elif f in prod_var_to_idx and prod_var_to_idx[f] in prod_to_var:
                            new_fields.append(prod_to_var[prod_var_to_idx[f]])
                        else:
                            return None, None  # unexpected reference

                    out_sinfo = (call.sinfo_args[0] if call.sinfo_args
                                 else bindings[prod_idx].var.struct_info_)

                    new_call = relax.Call(
                        call_tir_op,
                        [tir_gvar, relax.Tuple(new_fields)],
                        sinfo_args=[out_sinfo],
                    )
                    v = bb.emit(new_call, name_hint=bindings[prod_idx].var.name_hint)
                    prod_to_var[prod_idx] = v

            # 2) Remap consumer params → inlined vars or new params
            param_remap = {}
            for i, old_param in enumerate(consumer_func.params):
                if i in param_to_prod:
                    param_remap[old_param] = prod_to_var[param_to_prod[i]]
                elif i in cidx_to_param:
                    param_remap[old_param] = cidx_to_param[i]

            # 3) Re-emit consumer function body bindings with remapped vars
            env = dict(param_remap)
            for block in consumer_func.body.blocks:
                for binding in block.bindings:
                    if isinstance(binding, relax.VarBinding):
                        remapped = _remap_vars(binding.value, env)
                        v = bb.emit(remapped, name_hint=binding.var.name_hint)
                        env[binding.var] = v

            output = _remap_vars(consumer_func.body.body, env)
            bb.emit_output(output)

        bb.emit_func_output(output)

    new_func = bb.get()["fused_inline"]
    # Preserve Primitive attribute so FuseTIR picks it up
    new_func = new_func.with_attr("Primitive", tvm.tir.IntImm("int32", 1))

    return new_func, new_call_args


def _remap_vars(expr, env):
    """Replace Var references in an expression using the env mapping.

    Only remaps Var nodes in arguments and tuple fields. Preserves
    sinfo_args and attrs untouched (required for call_tir correctness).
    """
    if not env:
        return expr
    if isinstance(expr, relax.Var):
        return env.get(expr, expr)
    if isinstance(expr, relax.Call):
        new_op = env.get(expr.op, expr.op) if isinstance(expr.op, relax.Var) else expr.op
        new_args = [_remap_vars(a, env) for a in expr.args]
        return relax.Call(new_op, new_args, expr.attrs, expr.sinfo_args, expr.span)
    if isinstance(expr, relax.Tuple):
        return relax.Tuple([_remap_vars(f, env) for f in expr.fields], expr.span)
    if isinstance(expr, relax.TupleGetItem):
        return relax.TupleGetItem(_remap_vars(expr.tuple_value, env), expr.index, expr.span)
    if isinstance(expr, relax.ShapeExpr):
        return expr  # shape expressions don't contain Var references
    if isinstance(expr, relax.If):
        return relax.If(
            _remap_vars(expr.cond, env),
            _remap_vars(expr.true_branch, env),
            _remap_vars(expr.false_branch, env), expr.span)
    return expr
