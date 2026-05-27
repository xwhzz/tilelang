"""Relax-level pass: detect chained GEMM call_tir bindings and fuse them.

Takes *unscheduled* Relax IR, schedules both GEMMs with fusion-compatible
tiling (matching the fused_two_gemm2 strategy), then fuses them at the
lowered TIR level so the intermediate tensor stays in shared memory.

Strategy (per fused_two_gemm2):
  gemm1 tile = (block_M, K1, block_K0)  -> N dim (K1) not tiled
  gemm2 tile = (block_M, N, K1)         -> K dim (K1) and N dim not tiled

Both gemms get a 1D grid (1, M // block_M); blocks correspond 1-to-1.
The intermediate lives in a per-block shared buffer [block_M, K1].
"""

from __future__ import annotations

from collections import namedtuple
from tilelang import tvm
from tvm import relax, tir

# ---------------------------------------------------------------------------
# TIR helpers
# ---------------------------------------------------------------------------

def _get_tilelang_root(body: tir.Stmt) -> tir.stmt.Block | None:
    """Find the 'tilelang_root' block recursively."""
    found: list[tir.stmt.Block] = []

    def _visit(node):
        if isinstance(node, tir.stmt.Block) and node.name_hint == "tilelang_root":
            found.append(node)

    tir.stmt_functor.post_order_visit(body, _visit)
    return found[0] if found else None


def _build_sync() -> tir.stmt.Evaluate:
    """``__syncthreads`` for ``shared.dyn``."""
    op = tvm.ir.Op.get("tir.tvm_storage_sync")
    return tir.Evaluate(tir.Call("handle", op, [tir.StringImm("shared.dyn")]))


def _find_copy_dest_shared_buf(copy_stmt: tir.stmt.Evaluate) -> tir.Buffer | None:
    """Return the destination buffer of a ``T.copy``, if it is shared."""
    call = copy_stmt.value
    dest = call.args[1]
    if isinstance(dest, tir.Call) and dest.op.name == "tl.tileop.region":
        bload = dest.args[0]
        if isinstance(bload, tir.BufferLoad):
            buf = bload.buffer
            if "shared" in str(buf.scope()):
                return buf
    return None


def _replace_root_block(body: tir.Stmt, new_root: tir.stmt.Block) -> tir.Stmt:
    """Replace the 'tilelang_root' block inside the outer anonymous block.

    *body* is a ``BlockRealize`` whose inner anonymous block contains a
    nested chain of ``AttrStmt`` (thread bindings) ending in
    ``BlockRealize(tilelang_root)``.

    Returns a new ``BlockRealize`` with the tilelang_root block swapped.
    """
    def _swap_in_block(stmt):
        if isinstance(stmt, tir.stmt.BlockRealize):
            if stmt.block.name_hint == "tilelang_root":
                return tir.stmt.BlockRealize(
                    block=new_root,
                    iter_values=stmt.iter_values,
                    predicate=stmt.predicate,
                )
        if isinstance(stmt, tir.AttrStmt):
            return tir.AttrStmt(
                node=stmt.node,
                attr_key=stmt.attr_key,
                value=stmt.value,
                body=_swap_in_block(stmt.body),
            )
        return stmt

    outer_br = body
    outer_block = outer_br.block
    new_outer_body = _swap_in_block(outer_block.body)
    new_outer_block = tir.stmt.Block(
        iter_vars=outer_block.iter_vars,
        reads=outer_block.reads,
        writes=outer_block.writes,
        name_hint=outer_block.name_hint,
        body=new_outer_body,
        init=outer_block.init,
        alloc_buffers=outer_block.alloc_buffers,
        annotations=outer_block.annotations,
    )
    return tir.stmt.BlockRealize(
        block=new_outer_block,
        iter_values=outer_br.iter_values,
        predicate=outer_br.predicate,
    )


def _get_launch_thread_vars(body: tir.Stmt) -> dict[str, tir.Var]:
    """Map thread_tag -> Var from lowered TIR body (AttrStmt chain)."""
    result: dict[str, tir.Var] = {}
    outer_block = body.block
    stmt = outer_block.body
    while isinstance(stmt, tir.stmt.AttrStmt):
        if stmt.attr_key == "thread_extent" and stmt.node is not None:
            iv = stmt.node
            result[iv.thread_tag] = iv.var
        stmt = stmt.body
    return result


def _lower_for_fusion(func: tir.PrimFunc) -> tir.PrimFunc:
    """Lower a scheduled PrimFunc to post-ReserveRootBlock form."""
    from tilelang.engine.phase import NormalizeScheduledIR as _norm
    mod = tvm.IRModule({"main": func})
    mod = _norm(mod)
    return mod["main"]


def _schedule_matmul(
    func: tir.PrimFunc,
    target: tvm.target.Target,
    tile_override: tuple[int, int, int],
) -> tir.PrimFunc | None:
    """Schedule *func* with the given tile_override."""
    from tilelang.schedule.templates.gpu.matmul import Matmul
    rule = Matmul(tile_override=tile_override)
    result = rule.apply(func, target, False)
    if result is None:
        return None
    if isinstance(result, list):
        schedule = result[0]
    else:
        schedule = result
    return schedule.mod["main"]


# ---------------------------------------------------------------------------
# Chain detection (works on Relax dataflow, same for scheduled / unscheduled)
# ---------------------------------------------------------------------------

ChainInfo = namedtuple("ChainInfo", [
    "first_gv",              # GlobalVar of first matmul
    "second_gv",             # GlobalVar of second matmul
    "intermediate_var",       # Var connecting them (output of first -> input of second)
    "intermediate_arg_idx",   # which argument index of second matmul is the intermediate
    "first_args",             # list of relax.Var arguments to first matmul
    "second_args",            # list of relax.Var arguments to second matmul
    "output_var",             # the output Var of the second matmul
])


def _find_chained_matmul_bindings(mod: tvm.IRModule) -> list[ChainInfo]:
    """Scan the Relax main function for chained matmul call_tir bindings."""
    try:
        main = mod["main"]
    except (KeyError, AttributeError):
        return []

    if not isinstance(main, relax.Function):
        return []

    chains: list[ChainInfo] = []

    for block in main.body.blocks:
        if not isinstance(block, relax.DataflowBlock):
            continue

        call_tir_bindings: dict[relax.Var, tuple[tir.GlobalVar, list[relax.Var]]] = {}

        for binding in block.bindings:
            if not isinstance(binding, relax.VarBinding):
                continue
            val = binding.value
            if not isinstance(val, relax.Call):
                continue
            is_call_tir = (
                isinstance(val.op, relax.ExternFunc) and val.op.global_symbol == "call_tir"
            ) or (
                isinstance(val.op, tvm.ir.Op) and val.op.name == "relax.call_tir"
            )
            if not is_call_tir:
                continue
            gv = val.args[0]
            args_tuple = val.args[1]
            args = [args_tuple[i] for i in range(len(args_tuple))]
            call_tir_bindings[binding.var] = (gv, args)

        for var, (gv, args) in call_tir_bindings.items():
            for consumer_var, (consumer_gv, consumer_args) in call_tir_bindings.items():
                if var in consumer_args:
                    arg_idx = consumer_args.index(var)
                    chains.append(ChainInfo(
                        first_gv=gv,
                        second_gv=consumer_gv,
                        intermediate_var=var,
                        intermediate_arg_idx=arg_idx,
                        first_args=args,
                        second_args=consumer_args,
                        output_var=consumer_var,
                    ))

    return chains


# ---------------------------------------------------------------------------
# TIR fusion for compatibly-tiled gemms
# ---------------------------------------------------------------------------

def _fuse_with_compatible_tiling(
    sched1: tir.PrimFunc,
    sched2: tir.PrimFunc,
    unsched1: tir.PrimFunc,
    unsched2: tir.PrimFunc,
    intermediate_param_idx: int,
) -> tir.PrimFunc | None:
    """Fuse two gemms that were scheduled with fusion-compatible tiling.

    *sched1/sched2* are the scheduled (but not yet lowered) PrimFuncs.
    *unsched1/unsched2* are the original unscheduled PrimFuncs (for buffer lookup).
    """
    lowered1 = _lower_for_fusion(sched1)
    lowered2 = _lower_for_fusion(sched2)

    root1 = _get_tilelang_root(lowered1.body)
    root2 = _get_tilelang_root(lowered2.body)
    if root1 is None or root2 is None:
        return None

    body1 = root1.body
    body2 = root2.body

    # Collect thread binding Var names that body1 may shadow via LetStmts.
    # These LetStmts with e.g. "ax0_ax1_0_fused = T.int32()" shadow the outer
    # launch_thread bindings and must be stripped.
    _tv1 = _get_launch_thread_vars(lowered1.body)
    _thread_var_names = {v.name for v in _tv1.values()}

    # Remap func2's thread/block binding variables to func1's
    var_map1 = _get_launch_thread_vars(lowered1.body)
    var_map2 = _get_launch_thread_vars(lowered2.body)
    remap: dict[tir.Var, tir.PrimExpr] = {}
    for key, v2 in var_map2.items():
        if key in var_map1:
            remap[v2] = var_map1[key]
    if remap:
        body2 = tir.stmt_functor.substitute(body2, remap)

    # Identify buffer data variables from unscheduled funcs.
    # In lowered TIR the buffers are Let-bound aliases; match by .data (Var), not identity.
    c1_out_data = unsched1.buffer_map[unsched1.params[-1]].data
    c1_in_data = unsched2.buffer_map[unsched2.params[intermediate_param_idx]].data

    def _find_copy_reading_by_data(body: tir.Stmt, target_data: tir.Var) -> tir.stmt.Evaluate | None:
        """Match source buffer by .data (Var)."""
        found: list[tir.stmt.Evaluate] = []
        def _visit(node):
            if not isinstance(node, tir.stmt.Evaluate):
                return
            call = node.value
            if not isinstance(call, tir.Call) or call.op.name != "tl.tileop.copy":
                return
            src = call.args[0]
            if isinstance(src, tir.Call) and src.op.name == "tl.tileop.region":
                bload = src.args[0]
                if isinstance(bload, tir.BufferLoad) and bload.buffer.data.same_as(target_data):
                    found.append(node)
        tir.stmt_functor.post_order_visit(body, _visit)
        return found[0] if found else None

    # Step 1 — locate gemm2's read copy: copy(C1_in -> shared)
    read_copy = _find_copy_reading_by_data(body2, c1_in_data)
    if read_copy is None:
        return None
    c1_shared_in = _find_copy_dest_shared_buf(read_copy)
    if c1_shared_in is None:
        return None

    # Create per-block intermediate shared buffer (same shape as gemm2's input shared buf)
    intermediate = tir.decl_buffer(
        [int(s) for s in c1_shared_in.shape],
        dtype=c1_shared_in.dtype,
        name="C1_intermediate",
        scope="shared.dyn",
    )

    # Build a substitution map: blockIdx.* vars → 0, used to transform
    # global write-back indices (by*block_M + local → local) into per-block
    # shared-memory indices for the non-tile-GEMM path.
    _block_offset_map: dict[tir.Var, tir.PrimExpr] = {}
    for key, v in _tv1.items():
        if "blockIdx" in key:
            _block_offset_map[v] = tir.IntImm("int32", 0)

    def _subst_expr(expr: tir.PrimExpr, var_map: dict) -> tir.PrimExpr:
        """Substitute Vars in an expression tree."""
        if isinstance(expr, tir.Var) and expr in var_map:
            return var_map[expr]
        if isinstance(expr, (tir.Add, tir.Sub, tir.Mul, tir.Div,
                             tir.FloorDiv, tir.FloorMod, tir.Mod,
                             tir.Min, tir.Max, tir.EQ, tir.NE,
                             tir.LT, tir.LE, tir.GT, tir.GE)):
            return expr.__class__(_subst_expr(expr.a, var_map),
                                  _subst_expr(expr.b, var_map))
        if isinstance(expr, tir.Cast):
            return tir.Cast(expr.dtype, _subst_expr(expr.value, var_map))
        if isinstance(expr, tir.Select):
            return tir.Select(
                _subst_expr(expr.condition, var_map),
                _subst_expr(expr.true_value, var_map),
                _subst_expr(expr.false_value, var_map))
        if isinstance(expr, tir.Not):
            return tir.Not(_subst_expr(expr.a, var_map))
        return expr

    # Step 2 — redirect gemm1 write-back to intermediate with local indices.
    # The lowered TIR uses chains of LetStmt (not SeqStmt) that PyStmtExprMutator
    # mishandles.  Instead of mutators, walk the LetStmt chain directly and
    # rebuild it with the write-back target replaced.
    # Handles both tile-GEMM (T.copy destination) and non-tile-GEMM (BufferStore
    # through For loops) write-back patterns.
    def _rewrite_writeback(stmt):
        """Walk a LetStmt chain; replace write-back T.copy/BufferStore and strip
        LetStmts that shadow thread-binding variables."""
        if isinstance(stmt, tir.stmt.Evaluate):
            call = stmt.value
            if isinstance(call, tir.Call) and call.op.name == "tl.tileop.copy":
                dst = call.args[1]
                if isinstance(dst, tir.Call) and dst.op.name == "tl.tileop.region":
                    bload = dst.args[0]
                    if isinstance(bload, tir.BufferLoad) and bload.buffer.data.same_as(c1_out_data):
                        new_indices = [tir.IntImm("int32", 0) for _ in bload.indices]
                        new_bload = tir.BufferLoad(intermediate, new_indices)
                        new_region_args = [new_bload] + list(dst.args[1:])
                        new_dst = tir.Call(dst.dtype, dst.op, new_region_args)
                        new_call = tir.Call(call.dtype, call.op, [call.args[0], new_dst])
                        return tir.Evaluate(new_call)
            return stmt
        if isinstance(stmt, tir.stmt.BufferStore):
            # Non-tile-GEMM write-back: BufferStore to global output buffer.
            # Replace buffer and strip block offsets from indices so they
            # become local indices into the per-block intermediate.
            if stmt.buffer.data.same_as(c1_out_data):
                new_indices = [_subst_expr(idx, _block_offset_map) for idx in stmt.indices]
                return tir.BufferStore(intermediate, stmt.value, new_indices)
            return stmt
        if isinstance(stmt, tir.stmt.LetStmt):
            # Strip LetStmts that shadow thread-binding variables (e.g.
            # "ax0_ax1_0_fused = T.int32()") — the outer launch_thread already
            # defines them.  Substitute the shadow Var with the outer Var
            # so references in the body resolve correctly.
            if stmt.var.name in _thread_var_names:
                outer_var = next((v for v in _tv1.values() if v.name == stmt.var.name), None)
                body = _rewrite_writeback(stmt.body)
                if outer_var is not None and outer_var is not stmt.var:
                    body = tir.stmt_functor.substitute(body, {stmt.var: outer_var})
                return body
            body = _rewrite_writeback(stmt.body)
            if body is stmt.body:
                return stmt
            return tir.stmt.LetStmt(stmt.var, stmt.value, body)
        if isinstance(stmt, tir.stmt.SeqStmt):
            new_stmts = [_rewrite_writeback(s) for s in stmt]
            if all(a is b for a, b in zip(new_stmts, stmt)):
                return stmt
            return tir.SeqStmt(new_stmts)
        if isinstance(stmt, tir.stmt.For):
            new_body = _rewrite_writeback(stmt.body)
            if new_body is stmt.body:
                return stmt
            return tir.For(stmt.loop_var, stmt.min, stmt.extent, stmt.kind, new_body,
                          thread_binding=stmt.thread_binding, annotations=stmt.annotations)
        if isinstance(stmt, tir.stmt.AttrStmt):
            new_body = _rewrite_writeback(stmt.body)
            if new_body is stmt.body:
                return stmt
            return tir.AttrStmt(stmt.node, stmt.attr_key, stmt.value, new_body)
        if isinstance(stmt, tir.stmt.IfThenElse):
            new_then = _rewrite_writeback(stmt.then_case)
            new_else = _rewrite_writeback(stmt.else_case) if stmt.else_case else None
            if new_then is stmt.then_case and new_else is stmt.else_case:
                return stmt
            return tir.stmt.IfThenElse(stmt.condition, new_then, new_else)
        if isinstance(stmt, tir.stmt.BlockRealize):
            new_block = _rewrite_writeback(stmt.block)
            if new_block is stmt.block:
                return stmt
            return tir.stmt.BlockRealize(new_block, stmt.iter_values, stmt.predicate)
        if isinstance(stmt, tir.stmt.Block):
            new_body = _rewrite_writeback(stmt.body)
            if new_body is stmt.body:
                return stmt
            return tir.stmt.Block(stmt.iter_vars, stmt.reads, stmt.writes, stmt.name_hint,
                                 new_body, stmt.init, stmt.alloc_buffers, stmt.annotations)
        return stmt

    body1 = _rewrite_writeback(body1)

    # Step 3 — in gemm2: remove global->shared copy, redirect shared buf to intermediate.
    def _replace_buf_in_expr(expr, old_buf, new_buf, by_data=True):
        """Replace buffer references in an expression.  Matches by .data or identity."""
        if isinstance(expr, tir.BufferLoad):
            matches = expr.buffer.data.same_as(old_buf.data) if by_data else expr.buffer.same_as(old_buf)
            if matches:
                return tir.BufferLoad(new_buf, [_replace_buf_in_expr(i, old_buf, new_buf, by_data) for i in expr.indices])
            return tir.BufferLoad(expr.buffer, [_replace_buf_in_expr(i, old_buf, new_buf, by_data) for i in expr.indices])
        if isinstance(expr, tir.BufferStore):
            matches = expr.buffer.data.same_as(old_buf.data) if by_data else expr.buffer.same_as(old_buf)
            if matches:
                return tir.BufferStore(new_buf,
                    _replace_buf_in_expr(expr.value, old_buf, new_buf, by_data),
                    [_replace_buf_in_expr(i, old_buf, new_buf, by_data) for i in expr.indices])
            return tir.BufferStore(expr.buffer,
                _replace_buf_in_expr(expr.value, old_buf, new_buf, by_data),
                [_replace_buf_in_expr(i, old_buf, new_buf, by_data) for i in expr.indices])
        if isinstance(expr, tir.Call):
            return tir.Call(expr.dtype, expr.op, [_replace_buf_in_expr(a, old_buf, new_buf, by_data) for a in expr.args])
        # Recurse into binary/unary ops to find BufferLoad inside Add/Mul/etc.
        if isinstance(expr, (tir.Add, tir.Sub, tir.Mul, tir.Div,
                             tir.FloorDiv, tir.FloorMod, tir.Mod,
                             tir.Min, tir.Max)):
            return expr.__class__(
                _replace_buf_in_expr(expr.a, old_buf, new_buf, by_data),
                _replace_buf_in_expr(expr.b, old_buf, new_buf, by_data))
        if isinstance(expr, tir.Cast):
            return tir.Cast(expr.dtype, _replace_buf_in_expr(expr.value, old_buf, new_buf, by_data))
        if isinstance(expr, tir.Select):
            return tir.Select(
                _replace_buf_in_expr(expr.condition, old_buf, new_buf, by_data),
                _replace_buf_in_expr(expr.true_value, old_buf, new_buf, by_data),
                _replace_buf_in_expr(expr.false_value, old_buf, new_buf, by_data))
        if isinstance(expr, tir.Not):
            return tir.Not(_replace_buf_in_expr(expr.a, old_buf, new_buf, by_data))
        return expr

    def _rewrite_gemm2(stmt):
        """Walk body2: remove read_copy, replace shared buffer refs with intermediate."""
        if stmt is None:
            return None
        if stmt.same_as(read_copy):
            return None
        if isinstance(stmt, tir.stmt.LetStmt):
            body = _rewrite_gemm2(stmt.body)
            if body is None:
                return tir.Evaluate(tir.const(0, "int32"))
            if body is stmt.body:
                return stmt
            return tir.stmt.LetStmt(stmt.var, stmt.value, body)
        if isinstance(stmt, tir.stmt.SeqStmt):
            new_stmts = [_rewrite_gemm2(s) for s in stmt]
            new_stmts = [s for s in new_stmts if s is not None]
            if not new_stmts:
                return tir.Evaluate(tir.const(0, "int32"))
            return tir.SeqStmt(new_stmts) if len(new_stmts) > 1 else new_stmts[0]
        if isinstance(stmt, tir.stmt.For):
            new_body = _rewrite_gemm2(stmt.body)
            if new_body is stmt.body:
                return stmt
            if new_body is None:
                new_body = tir.Evaluate(tir.const(0, "int32"))
            return tir.For(stmt.loop_var, stmt.min, stmt.extent, stmt.kind, new_body,
                          thread_binding=stmt.thread_binding, annotations=stmt.annotations)
        if isinstance(stmt, tir.stmt.AttrStmt):
            new_body = _rewrite_gemm2(stmt.body)
            if new_body is stmt.body:
                return stmt
            if new_body is None:
                new_body = tir.Evaluate(tir.const(0, "int32"))
            return tir.AttrStmt(stmt.node, stmt.attr_key, stmt.value, new_body)
        if isinstance(stmt, tir.stmt.Evaluate):
            # Replace buffer refs inside expressions (T.gemm_py, T.copy regions, etc.)
            # Match by .data to handle LetStmt aliases.
            new_val = _replace_buf_in_expr(stmt.value, c1_shared_in, intermediate, by_data=True)
            new_val = _replace_buf_in_expr(new_val, c1_in_buf_unsched, intermediate, by_data=True)
            if new_val is stmt.value:
                return stmt
            return tir.Evaluate(new_val)
        if isinstance(stmt, tir.BufferStore):
            # Replace c1_shared_in buffer refs in the value expression.
            # Match by .data (Var) because the compute loop may use a LetStmt
            # alias of the shared buffer with different identity.
            new_value = _replace_buf_in_expr(stmt.value, c1_shared_in, intermediate, by_data=True)
            new_value = _replace_buf_in_expr(new_value, c1_in_buf_unsched, intermediate, by_data=True)
            # Also replace if the store target itself is the intermediate buffer
            new_buf = stmt.buffer
            new_indices = list(stmt.indices)
            if stmt.buffer.data.same_as(c1_shared_in.data):
                new_buf = intermediate
                new_indices = [_replace_buf_in_expr(i, c1_shared_in, intermediate, by_data=True) for i in stmt.indices]
            elif stmt.buffer.data.same_as(c1_in_data):
                new_buf = intermediate
                new_indices = [_replace_buf_in_expr(i, c1_in_buf_unsched, intermediate, by_data=True) for i in stmt.indices]
            if new_value is stmt.value and new_buf is stmt.buffer and new_indices == list(stmt.indices):
                return stmt
            return tir.BufferStore(new_buf, new_value, new_indices)
        if isinstance(stmt, tir.stmt.IfThenElse):
            new_then = _rewrite_gemm2(stmt.then_case)
            new_else = _rewrite_gemm2(stmt.else_case) if stmt.else_case else None
            if new_then is stmt.then_case and new_else is stmt.else_case:
                return stmt
            return tir.stmt.IfThenElse(stmt.condition, new_then, new_else)
        if isinstance(stmt, tir.stmt.BlockRealize):
            new_block = _rewrite_gemm2(stmt.block)
            if new_block is stmt.block:
                return stmt
            return tir.stmt.BlockRealize(new_block, stmt.iter_values, stmt.predicate)
        if isinstance(stmt, tir.stmt.Block):
            new_body = _rewrite_gemm2(stmt.body)
            if new_body is stmt.body:
                return stmt
            return tir.stmt.Block(stmt.iter_vars, stmt.reads, stmt.writes, stmt.name_hint,
                                  new_body, stmt.init, stmt.alloc_buffers, stmt.annotations)
        return stmt

    # Need the unscheduled c1_in_buf for .data matching in expression rewrites
    c1_in_buf_unsched = unsched2.buffer_map[unsched2.params[intermediate_param_idx]]
    body2 = _rewrite_gemm2(body2)
    if body2 is None:
        body2 = tir.Evaluate(tir.const(0, "int32"))

    # Step 4 — assemble
    sync = _build_sync()
    fused_body = tir.SeqStmt([body1, sync, body2])

    merged_allocs = list(root1.alloc_buffers)
    merged_allocs.append(intermediate)
    for buf in root2.alloc_buffers:
        if buf.same_as(c1_shared_in):
            continue
        merged_allocs.append(buf)

    new_root = tir.stmt.Block(
        iter_vars=root1.iter_vars,
        reads=root1.reads,
        writes=root2.writes,
        name_hint="tilelang_root",
        body=fused_body,
        init=root1.init,
        alloc_buffers=merged_allocs,
        annotations=root1.annotations,
    )

    fused_body_full = _replace_root_block(lowered1.body, new_root)

    # Build params using unscheduled function Vars (which have unique names).
    # Lowered functions have auto-generated names like "var_args_1" that
    # collide, causing buffer aliasing bugs.  The unscheduled params retain
    # the original distinct names.
    fused_params = list(unsched1.params)[:-1]
    for i, p in enumerate(unsched2.params):
        if i == intermediate_param_idx:
            continue
        fused_params.append(p)

    fused_buffer_map: dict[tir.Var, tir.Buffer] = {}
    for i, p in enumerate(unsched1.params[:-1]):
        fused_buffer_map[p] = lowered1.buffer_map[lowered1.params[i]]
    for i, p in enumerate(unsched2.params):
        if i == intermediate_param_idx:
            continue
        fused_buffer_map[p] = lowered2.buffer_map[lowered2.params[i]]

    return tir.PrimFunc(
        params=fused_params,
        body=fused_body_full,
        ret_type=lowered1.ret_type,
        buffer_map=fused_buffer_map,
        attrs=lowered1.attrs,
    )


# ---------------------------------------------------------------------------
# Relax main function rewriting
# ---------------------------------------------------------------------------

def _rewrite_relax_main(
    mod: tvm.IRModule,
    chain: ChainInfo,
    fused_gv: tir.GlobalVar,
    fused_sinfo: relax.StructInfo,
) -> tvm.IRModule:
    """Replace two call_tir bindings with a single fused call_tir."""
    main = mod["main"]
    if not isinstance(main, relax.Function):
        return mod

    new_blocks = []
    for block in main.body.blocks:
        if not isinstance(block, relax.DataflowBlock):
            new_blocks.append(block)
            continue

        intermediate_var = chain.intermediate_var
        output_var = chain.output_var

        new_bindings = []
        fused_inserted = False

        # Build the fused call_tir
        fused_args_list = list(chain.first_args)
        for i, arg in enumerate(chain.second_args):
            if i == chain.intermediate_arg_idx:
                continue
            fused_args_list.append(arg)
        fused_call = relax.call_tir(
            fused_gv, relax.Tuple(fused_args_list), out_sinfo=fused_sinfo
        )
        fused_binding = relax.VarBinding(output_var, fused_call)

        for binding in block.bindings:
            if not isinstance(binding, relax.VarBinding):
                new_bindings.append(binding)
                continue

            # Skip the two matmul bindings
            if binding.var == intermediate_var:
                continue
            if binding.var == output_var:
                continue

            # Insert fused binding just before the tuple/gv binding that
            # references output_var
            val = binding.value
            if not fused_inserted:
                uses_output = False
                if isinstance(val, relax.Tuple):
                    for field in val.fields:
                        if field == output_var:
                            uses_output = True
                            break
                if uses_output:
                    new_bindings.append(fused_binding)
                    fused_inserted = True

            new_bindings.append(binding)

        if not fused_inserted:
            new_bindings.append(fused_binding)

        new_block = relax.DataflowBlock(new_bindings, block.span)
        new_blocks.append(new_block)

    new_body = relax.SeqExpr(new_blocks, main.body.body)
    new_main = relax.Function(
        params=main.params,
        body=new_body,
        ret_struct_info=main.ret_struct_info,
        attrs=main.attrs,
    )

    mod = mod.clone()
    mod["main"] = new_main
    return mod


# ---------------------------------------------------------------------------
# Module-level pass
# ---------------------------------------------------------------------------

def FuseChainedGemmRelax():
    """Create a Relax module pass that fuses chained GEMMs.

    Runs on *unscheduled* Relax IR.  For each detected chain the pass:

    1. schedules gemm1 with tile ``(block_M, K1, block_K0)``
    2. schedules gemm2 with tile ``(block_M, N, K1)``
    3. lowers both and fuses them at the TIR level
    4. rewrites the Relax dataflow to use the fused kernel

    The fused function is marked ``tir.is_scheduled`` so that downstream
    normal scheduling skips it.
    """
    from tvm.ir.transform import PassContext, module_pass

    @module_pass(opt_level=0)
    class _Pass:
        def transform_module(self, mod: tvm.IRModule, ctx: PassContext) -> tvm.IRModule:
            # Apply Relax operator fusion to convert R.matmul -> call_tir
            seq = tvm.transform.Sequential([
                relax.transform.LegalizeOps(),
                relax.transform.AnnotateTIROpPattern(),
                relax.transform.FoldConstant(),
                relax.transform.FuseOps(),
                relax.transform.FuseTIR(),
            ])
            mod = seq(mod)

            chains = _find_chained_matmul_bindings(mod)
            if not chains:
                return mod

            target = tvm.target.Target.current(allow_none=True)
            if target is None:
                return mod

            for chain in chains:
                if "matmul" not in chain.first_gv.name_hint:
                    continue
                if "matmul" not in chain.second_gv.name_hint:
                    continue

                func1 = mod[chain.first_gv]
                func2 = mod[chain.second_gv]
                if not isinstance(func1, tir.PrimFunc) or not isinstance(func2, tir.PrimFunc):
                    print(f"[FuseChainedGemmRelax]   skip: not PrimFunc", flush=True)
                    continue

                # Infer shapes from TIR buffer_map
                c1_out_buf = func1.buffer_map[func1.params[-1]]
                c2_out_buf = func2.buffer_map[func2.params[-1]]
                c1_shape = [int(s) for s in c1_out_buf.shape]
                c2_shape = [int(s) for s in c2_out_buf.shape]
                M, K1 = c1_shape[0], c1_shape[1]
                N = c2_shape[1]

                # Pick tile sizes using the same logic as the Matmul schedule rule.
                # Prefer larger blocks when the dimension allows, but limit
                # block_M for fp32 to keep pipeline-doubled shared memory in budget.
                from tilelang.schedule.templates.gpu.matmul import _choose_static_tile
                _dtype_bits = tvm.DataType(c1_out_buf.dtype).bits
                _max_m = 128 if _dtype_bits <= 16 else 64
                block_M = _choose_static_tile([c for c in [128, 64, 32] if c <= _max_m], M)
                K0_dim = int(func1.buffer_map[func1.params[0]].shape[1])
                block_K0 = _choose_static_tile([64, 32, 16], K0_dim)

                # Schedule gemm1: tile (block_M, K1, block_K0)
                sched1 = _schedule_matmul(func1, target, (block_M, K1, block_K0))
                if sched1 is None:
                    continue

                # Schedule gemm2: tile (block_M, N, K1)
                sched2 = _schedule_matmul(func2, target, (block_M, N, K1))
                if sched2 is None:
                    continue

                # Fuse
                fused_func = _fuse_with_compatible_tiling(
                    sched1, sched2, func1, func2, chain.intermediate_arg_idx,
                )
                if fused_func is None:
                    continue

                name = f"{chain.first_gv.name_hint}_{chain.second_gv.name_hint}_fused"
                fused_func = fused_func.with_attr("global_symbol", name)
                fused_func = fused_func.with_attr("tir.is_scheduled", True)
                fused_func = fused_func.with_attr("tir.is_lowered", True)

                # Register the fused function using a fresh BlockBuilder so
                # the GlobalVar gets proper struct_info — CallTIRRewrite and
                # RewriteDataflowReshape require it.
                _reg_bb = relax.BlockBuilder()
                fused_gv = _reg_bb.add_func(fused_func, name)

                del mod[chain.first_gv]
                del mod[chain.second_gv]
                mod[fused_gv] = fused_func

                # Output struct info
                fused_sinfo = relax.TensorStructInfo(c2_shape, c2_out_buf.dtype)

                # Rewrite Relax
                mod = _rewrite_relax_main(mod, chain, fused_gv, fused_sinfo)

            return mod

    return _Pass()
