# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# pylint: disable=invalid-name
"""Reduction rule for operators including softmax, layer norm, RMS norm, etc"""

from __future__ import annotations

from functools import reduce

from tvm import tir
from tvm.target import Target

from .. import Schedule as TileSchedule
from tilelang.carver.common_schedules import get_output_blocks
from tvm.dlight import normalize_prim_func, try_inline_contiguous_spatial
from tvm.dlight.analysis import BlockInfo
from .base import GPUScheduleRule
from .reduction import (
    _as_const_int,
    _analyze_reduction_update,
    _choose_num_threads,
    _choose_reduction_step,
    _find_buffer_index,
    _infer_init_value,
    _infer_reduce_dim,
    _is_direct_buffer_load,
)
from .layernorm_like import LayerNormLike


def _collect_input_buffers(rhs: tir.PrimExpr, write_buffer: tir.Buffer) -> list[tir.Buffer]:
    buffers: list[tir.Buffer] = []

    def _collect(expr):
        if (
            isinstance(expr, tir.BufferLoad)
            and (not expr.buffer.same_as(write_buffer))
            and not any(expr.buffer.same_as(buf) for buf in buffers)
        ):
            buffers.append(expr.buffer)

    tir.stmt_functor.post_order_visit(rhs, _collect)
    return buffers


def _contains_exp_like_call(expr: tir.PrimExpr) -> bool:
    has_exp_like = False

    def _collect(node):
        nonlocal has_exp_like
        if has_exp_like:
            return
        if isinstance(node, tir.Call):
            op_name = getattr(node.op, "name", "")
            if op_name in ("tir.exp", "tir.exp2"):
                has_exp_like = True

    tir.stmt_functor.post_order_visit(expr, _collect)
    return has_exp_like


def _is_same_buffer_load(
    lhs: tir.PrimExpr,
    rhs: tir.PrimExpr,
    target_buffer: tir.Buffer | None = None,
) -> bool:
    if not isinstance(lhs, tir.BufferLoad) or not isinstance(rhs, tir.BufferLoad):
        return False
    if not lhs.buffer.same_as(rhs.buffer):
        return False
    if target_buffer is not None and not lhs.buffer.same_as(target_buffer):
        return False
    if len(lhs.indices) != len(rhs.indices):
        return False
    return all(tir.analysis.expr_deep_equal(a, b) for a, b in zip(lhs.indices, rhs.indices))


def _is_square_of_buffer_load(expr: tir.PrimExpr, target_buffer: tir.Buffer) -> bool:
    if not isinstance(expr, tir.Mul):
        return False
    return _is_same_buffer_load(expr.a, expr.b, target_buffer)


def _block_writes_buffer(block: tir.Block, target_buffer: tir.Buffer) -> bool:
    return any(write_region.buffer.same_as(target_buffer) for write_region in block.writes)


def _should_preserve_two_reduction_bridge(
    sch: TileSchedule,
    block_infos: list[BlockInfo],
) -> bool:
    """Keep injective bridge blocks between two reductions when needed.

    If try_inline_contiguous_spatial inlines the bridge, the second stage
    often degrades to a multi-source update that cannot be lowered by
    reduce_at. Preserving the bridge keeps the second reduction single-source.
    """

    reduction_indices = [i for i, info in enumerate(block_infos) if info.is_reduction()]
    if len(reduction_indices) != 2:
        return False

    first_reduction_idx, second_reduction_idx = reduction_indices
    bridge_infos = block_infos[first_reduction_idx + 1 : second_reduction_idx]
    trailing_infos = block_infos[second_reduction_idx + 1 :]
    if len(bridge_infos) == 0:
        return False
    if not all(info.is_injective() for info in bridge_infos):
        return False
    if len(trailing_infos) == 0 or not all(info.is_injective() for info in trailing_infos):
        return False

    output_block_names = {sch.get(output_block).name_hint for output_block in get_output_blocks(sch, block_infos)}
    if sch.get(trailing_infos[-1].block_rv).name_hint not in output_block_names:
        return False

    second_reduction_stmt = sch.get(block_infos[second_reduction_idx].block_rv)
    second_update_info = _analyze_reduction_update(second_reduction_stmt)
    if second_update_info is None or len(second_reduction_stmt.writes) != 1:
        return False
    second_reduce_type, second_rhs_expr = second_update_info
    second_input_buffers = _collect_input_buffers(
        second_rhs_expr,
        second_reduction_stmt.writes[0].buffer,
    )
    if len(second_input_buffers) != 1:
        return False

    bridge_output_buffers: list[tir.Buffer] = []
    bridge_has_exp_like = False
    first_reduction_stmt = sch.get(block_infos[first_reduction_idx].block_rv)
    first_reduce_type_rhs = _analyze_reduction_update(first_reduction_stmt)
    first_output = first_reduction_stmt.writes[0].buffer if len(first_reduction_stmt.writes) == 1 else None
    first_is_max = first_reduce_type_rhs is not None and first_reduce_type_rhs[0] == "max"
    for bridge_info in bridge_infos:
        bridge_stmt = sch.get(bridge_info.block_rv)
        for write_region in bridge_stmt.writes:
            bridge_output_buffers.append(write_region.buffer)
        if (
            isinstance(bridge_stmt.body, tir.BufferStore)
            and _contains_exp_like_call(bridge_stmt.body.value)
            and (
                first_output is None
                or _block_writes_buffer(bridge_stmt, first_output)
                or any(read_region.buffer.same_as(first_output) for read_region in bridge_stmt.reads)
            )
        ):
            bridge_has_exp_like = True

    if not any(second_input_buffers[0].same_as(buf) for buf in bridge_output_buffers):
        return False

    # Softmax-like chains are the high-priority case where preserving the
    # bridge is critical for performance and robust lowering.
    if first_is_max and bridge_has_exp_like:
        return True

    # Norm-like chains (e.g. layernorm variance) often rely on preserving a
    # centered bridge so sum(x*x) remains lowerable to tile-level reduce.
    return bool(second_reduce_type == "sum" and _is_square_of_buffer_load(second_rhs_expr, second_input_buffers[0]))


def _schedule_epilogue_block(
    sch: TileSchedule,
    epilogue_name: str,
    bx: tir.schedule.LoopRV,
    is_output_block: bool = False,
    cache_write_back: bool = True,
) -> None:
    epilogue = sch.get_block(epilogue_name)
    epilogue_stmt = sch.get(epilogue)
    if is_output_block:
        sch.reverse_compute_at(epilogue, bx, preserve_unit_loops=True)
    else:
        sch.compute_at(epilogue, bx, preserve_unit_loops=True)

    epilogue = sch.get_block(epilogue_name)
    loops = sch.get_loops(epilogue)
    if loops and epilogue_stmt.init is None:
        sch.parallel(loops[-1])

    epilogue = sch.get_block(epilogue_name)
    epilogue_stmt = sch.get(epilogue)
    if epilogue_stmt.init is None and isinstance(epilogue_stmt.body, tir.BufferStore):
        _cache_elementwise_block(
            sch,
            epilogue_name,
            bx,
            cache_writes=not is_output_block,
            cache_write_back=cache_write_back,
        )


def _cache_elementwise_block(
    sch: TileSchedule,
    block_name: str,
    bx: tir.schedule.LoopRV,
    cache_reads: bool = True,
    cache_writes: bool = True,
    cache_write_back: bool = True,
) -> None:
    block = sch.get_block(block_name)
    block_stmt = sch.get(block)
    block_writes = [region.buffer for region in block_stmt.writes]

    if cache_reads:
        for read_buffer_index, read_region in enumerate(block_stmt.reads):
            if any(read_region.buffer.same_as(write_buffer) for write_buffer in block_writes):
                continue
            block = sch.get_block(block_name)
            sch.cache_read_at(bx, block, read_buffer_index, "local.fragment")

    if cache_writes:
        block = sch.get_block(block_name)
        for write_buffer_index, _ in enumerate(sch.get(block).writes):
            sch.cache_write_at(
                bx,
                block,
                write_buffer_index,
                "local.fragment",
                write_back=cache_write_back,
            )
            block = sch.get_block(block_name)


def _schedule_prologue_block(
    sch: TileSchedule,
    block_name: str,
    bx: tir.schedule.LoopRV,
) -> None:
    block = sch.get_block(block_name)
    block_stmt = sch.get(block)
    loops = sch.get_loops(block)
    if block_stmt.init is not None and loops and len(block_stmt.iter_vars) == len(loops):
        s_count = sum(1 for iter_var in block_stmt.iter_vars if iter_var.iter_type == tir.IterVar.DataPar)
        r_count = sum(1 for iter_var in block_stmt.iter_vars if iter_var.iter_type == tir.IterVar.CommReduce)
        o_count = len(loops) - s_count - r_count
        if s_count > 0 and r_count > 0:
            s_loops = list(loops[:s_count])
            r_loops = list(loops[s_count : s_count + r_count])
            o_loops = list(loops[s_count + r_count : s_count + r_count + o_count])
            sch.reorder(*s_loops, *r_loops, *o_loops)
            loops = sch.get_loops(block)
            s_loops = list(loops[:s_count])
            tail_loops = list(loops[s_count:])
            s_fused = sch.fuse(*s_loops) if len(s_loops) > 1 else s_loops[0]
            so, si = sch.split(s_fused, factors=[None, 1], preserve_unit_iters=True)
            if tail_loops:
                sch.reorder(so, si, *tail_loops)

    block = sch.get_block(block_name)
    sch.compute_at(block, bx, preserve_unit_loops=True, index=0)

    block = sch.get_block(block_name)
    block_stmt = sch.get(block)
    if block_stmt.init is None and isinstance(block_stmt.body, tir.BufferStore):
        _cache_elementwise_block(sch, block_name, bx)


def _schedule_reduction_stage_at_bx(
    sch: TileSchedule,
    block_name: str,
    bx: tir.schedule.LoopRV,
    target: Target,
    output_block_names: set[str],
    force_explicit_update: bool = False,
    cache_reduce_write_back: bool = True,
) -> tir.PrimExpr | None:
    block = sch.get_block(block_name)
    block_stmt = sch.get(block)
    if len(block_stmt.writes) != 1 or len(block_stmt.reads) < 1:
        return None

    update_info = _analyze_reduction_update(block_stmt)
    if update_info is None:
        return None
    reduce_type, rhs_expr = update_info

    write_buffer = block_stmt.writes[0].buffer
    write_buffer_index = _find_buffer_index(block_stmt.writes, write_buffer)
    if write_buffer_index is None:
        return None

    input_buffers = _collect_input_buffers(rhs_expr, write_buffer)
    if len(input_buffers) == 0:
        return None

    read_buffer_indices: list[int] = []
    for input_buffer in input_buffers:
        read_index = _find_buffer_index(block_stmt.reads, input_buffer)
        if read_index is None:
            return None
        read_buffer_indices.append(read_index)

    is_single_source = len(set(read_buffer_indices)) == 1
    square_single_source = (
        is_single_source and len(input_buffers) == 1 and reduce_type == "sum" and _is_square_of_buffer_load(rhs_expr, input_buffers[0])
    )
    init_value = _infer_init_value(block_stmt, reduce_type)

    reduction_loops = sch.get_loops(block)
    if len(reduction_loops) == 0:
        return None
    r_fused = reduction_loops[-1]

    reduce_step = _choose_reduction_step(target, sch.get(r_fused).extent)
    if reduce_step is not None:
        ro, ri = sch.split(r_fused, factors=[None, reduce_step], preserve_unit_iters=True)
        cache_read_loop = ro
        reduce_loop: tir.schedule.LoopRV | None = ro
        thread_extent_expr = sch.get(ri).extent
    else:
        cache_read_loop = bx
        reduce_loop = None
        thread_extent_expr = sch.get(r_fused).extent

    if is_single_source and not force_explicit_update:
        # ------ Single-source path: cache_reduce_at + reduce_at ------
        block = sch.get_block(block_name)
        if block_name not in output_block_names:
            sch.set_scope(block, write_buffer_index, "local.fragment")

        for read_buffer_index in sorted(set(read_buffer_indices)):
            block = sch.get_block(block_name)
            sch.cache_read_at(cache_read_loop, block, read_buffer_index, "local.fragment")

        block = sch.get_block(block_name)
        sch.cache_reduce_at(
            bx,
            block,
            write_buffer_index,
            "local.fragment",
            init_value,
            write_back=cache_reduce_write_back,
        )

        can_lower_to_tile_reduce = is_single_source and (_is_direct_buffer_load(rhs_expr, input_buffers[0]) or square_single_source)
        if reduce_loop is not None and can_lower_to_tile_reduce:
            read_buffer_index = read_buffer_indices[0]
            reduce_type_for_lower = "sumsq" if square_single_source else reduce_type
            block = sch.get_block(block_name)
            block_stmt = sch.get(block)
            reduce_dim = _infer_reduce_dim(
                block_stmt.reads[read_buffer_index].buffer,
                block_stmt.writes[write_buffer_index].buffer,
            )
            sch.reduce_at(
                reduce_loop,
                block,
                read_buffer_index=read_buffer_index,
                write_buffer_index=write_buffer_index,
                reduce_type=reduce_type_for_lower,
                dim=reduce_dim,
                clear=False,
                replace_loop_body=True,
            )
    else:
        # ------ Multi-source path (e.g. sum(exp(x - max)) in softmax) ------
        # reduce_at only works for single-source reductions.
        # Strategy: decompose_reduction first (while the loop tree is
        # intact), then use fill_at + set_scope + cache_read_at.
        # We deliberately skip cache_reduce_at because it inserts a
        # per-chunk read-copy that resets the accumulator.

        block = sch.get_block(block_name)
        block_stmt = sch.get(block)
        if block_stmt.init is not None:
            decompose_loop = reduce_loop if reduce_loop is not None else r_fused
            sch.decompose_reduction(block, decompose_loop)
            block_name = block_name + "_update"

        block = sch.get_block(block_name)
        orig_name = block_name.removesuffix("_update")
        if orig_name not in output_block_names and block_name not in output_block_names:
            sch.set_scope(block, write_buffer_index, "local.fragment")

        # Tile-level initialization via T.fill (replaces the decomposed init).
        block = sch.get_block(block_name)
        sch.fill_at(bx, block, write_buffer_index, init_value)

        # Re-compute read buffer indices for the update block since the
        # reads array may have changed after decompose + set_scope.
        block = sch.get_block(block_name)
        block_stmt = sch.get(block)
        read_buffer_indices = []
        for input_buffer in input_buffers:
            read_index = _find_buffer_index(block_stmt.reads, input_buffer)
            if read_index is not None:
                read_buffer_indices.append(read_index)

        for read_buffer_index in sorted(set(read_buffer_indices)):
            block = sch.get_block(block_name)
            sch.cache_read_at(cache_read_loop, block, read_buffer_index, "local.fragment")

    return thread_extent_expr


class GeneralReduction(GPUScheduleRule):
    """General Reduction rule for operators including softmax, layer norm, RMS norm, etc"""

    def apply(  # pylint: disable=too-many-locals,too-many-return-statements
        self,
        func: tir.PrimFunc,
        target: Target,
        _: bool,
    ) -> None | tir.Schedule | list[tir.Schedule]:
        if not isinstance(func, tir.PrimFunc) or not self.is_target_available(target):
            return None

        # LayerNorm-like two-reduction chains are handled by a dedicated rule
        # to avoid fragile rewrite interactions in the generic template.
        layernorm_schedule = LayerNormLike().apply(func, target, False)
        if layernorm_schedule is not None:
            return layernorm_schedule

        sch = TileSchedule(func)
        block_infos = normalize_prim_func(sch)
        if block_infos is None:
            return None
        preserve_two_reduction_bridge = _should_preserve_two_reduction_bridge(sch, block_infos)
        if not preserve_two_reduction_bridge:
            block_infos = try_inline_contiguous_spatial(sch, block_infos)
            if block_infos is None or len(block_infos) == 0:
                return None

        reduction_indices = [i for i, info in enumerate(block_infos) if info.is_reduction()]
        if len(reduction_indices) == 0:
            return None
        first_reduction_index = reduction_indices[0]
        reduction_index = reduction_indices[-1]

        # Inline the pure injective prefix before the first reduction stage.
        for info in block_infos[:first_reduction_index]:
            if not info.is_injective():
                return None
            sch.compute_inline(info.block_rv)

        block_infos = normalize_prim_func(sch)
        if block_infos is None:
            return None

        if not preserve_two_reduction_bridge:
            block_infos = try_inline_contiguous_spatial(sch, block_infos)
            if block_infos is None or len(block_infos) == 0:
                return None

        reduction_indices = [i for i, info in enumerate(block_infos) if info.is_reduction()]
        if len(reduction_indices) == 0:
            return None
        output_block_names = {sch.get(output_block).name_hint for output_block in get_output_blocks(sch, block_infos)}

        if len(reduction_indices) > 1:
            if len(reduction_indices) == 2:
                first_reduction_idx, second_reduction_idx = reduction_indices
                trailing_infos = block_infos[second_reduction_idx + 1 :]
                second_reduction_stmt = sch.get(block_infos[second_reduction_idx].block_rv)
                second_update_info = _analyze_reduction_update(second_reduction_stmt)
                if second_update_info is not None and second_update_info[0] == "sum":
                    second_rhs_expr = second_update_info[1]
                    second_input_buffers = (
                        _collect_input_buffers(
                            second_rhs_expr,
                            second_reduction_stmt.writes[0].buffer,
                        )
                        if len(second_reduction_stmt.writes) == 1
                        else []
                    )
                    is_norm_like_second = len(second_input_buffers) == 1 and _is_square_of_buffer_load(
                        second_rhs_expr, second_input_buffers[0]
                    )
                    # Layernorm-like chains currently trigger unstable rewrite/codegen
                    # interactions in the two-reduction template path.
                    if is_norm_like_second and len(trailing_infos) >= 2:
                        return None

            # Specialized two-reduction chains with an optional injective bridge:
            #   reduce0 -> [injective...]* -> reduce1 -> out
            if len(reduction_indices) == 2:
                first_reduction_idx, second_reduction_idx = reduction_indices
                first_reduction_info = block_infos[first_reduction_idx]
                second_reduction_info = block_infos[second_reduction_idx]
                bridge_infos = block_infos[first_reduction_idx + 1 : second_reduction_idx]
                trailing_infos = block_infos[second_reduction_idx + 1 :]
                if (
                    all(info.is_injective() for info in bridge_infos)
                    and len(trailing_infos) >= 1
                    and all(info.is_injective() for info in trailing_infos)
                    and sch.get(trailing_infos[-1].block_rv).name_hint in output_block_names
                ):
                    output_block = trailing_infos[-1].block_rv
                    output_loops = sch.get_loops(output_block)

                    reduction_dom_kind = second_reduction_info.dom_kind()
                    num_leading_s = len(reduction_dom_kind) - len(reduction_dom_kind.lstrip("S"))
                    if num_leading_s <= 0 or len(output_loops) < num_leading_s:
                        return None

                    output_bx_loops = list(output_loops[:num_leading_s])
                    output_inner_loops = list(output_loops[num_leading_s:])

                    first_reduction_name = sch.get(first_reduction_info.block_rv).name_hint
                    second_reduction_name = sch.get(second_reduction_info.block_rv).name_hint
                    trailing_epilogue_names = [sch.get(info.block_rv).name_hint for info in trailing_infos[:-1]]

                    output_s_fused = sch.fuse(*output_bx_loops) if len(output_bx_loops) > 1 else output_bx_loops[0]
                    output_inner_fused = (
                        (sch.fuse(*output_inner_loops) if len(output_inner_loops) > 1 else output_inner_loops[0])
                        if output_inner_loops
                        else None
                    )

                    second_reduction_stmt = sch.get(second_reduction_info.block_rv)
                    second_update_info = _analyze_reduction_update(second_reduction_stmt)
                    is_second_multi_source = False
                    is_second_square_single_source = False
                    second_input_buffers: list[tir.Buffer] = []
                    if second_update_info is not None and len(second_reduction_stmt.writes) == 1:
                        second_reduce_type, second_rhs_expr = second_update_info
                        second_input_buffers = _collect_input_buffers(
                            second_rhs_expr,
                            second_reduction_stmt.writes[0].buffer,
                        )
                        is_second_multi_source = len(second_input_buffers) > 1
                        is_second_square_single_source = (
                            second_reduce_type == "sum"
                            and len(second_input_buffers) == 1
                            and _is_square_of_buffer_load(second_rhs_expr, second_input_buffers[0])
                        )

                    bridge_meta: list[tuple[BlockInfo, str, list[tir.Buffer]]] = []
                    for info in bridge_infos:
                        bridge_stmt = sch.get(info.block_rv)
                        bridge_meta.append(
                            (
                                info,
                                bridge_stmt.name_hint,
                                [write_region.buffer for write_region in bridge_stmt.writes],
                            )
                        )

                    keep_bridge_names: list[str] = []
                    if len(second_input_buffers) == 1:
                        second_src = second_input_buffers[0]
                        for _, name, write_buffers in bridge_meta:
                            if any(second_src.same_as(buf) for buf in write_buffers):
                                keep_bridge_names.append(name)
                    else:
                        keep_bridge_names = [name for _, name, _ in bridge_meta]

                    for info, name, _ in reversed(bridge_meta):
                        if name in keep_bridge_names:
                            continue
                        try:
                            sch.compute_inline(info.block_rv)
                        except Exception:
                            return None

                    bridge_names = keep_bridge_names
                    if output_inner_fused is None:
                        num_threads = 1
                    else:
                        num_threads = _choose_num_threads(target, sch.get(output_inner_fused).extent)
                    if is_second_multi_source:
                        # Multi-source updates currently stay as explicit loops.
                        # Keep single-thread launch to avoid duplicated reduction work.
                        num_threads = 1
                    if num_threads < 1:
                        return None

                    # Keep row tile at 1. Larger row tiles can trigger invalid
                    # copy-back index vars in current cache_reduce_at lowering.
                    bx, output_inner = sch.split(
                        output_s_fused,
                        factors=[None, 1],
                        preserve_unit_iters=True,
                    )
                    if output_inner_fused is not None:
                        sch.reorder(bx, output_inner, output_inner_fused)
                        sch.parallel(output_inner_fused)
                    else:
                        sch.reorder(bx, output_inner)
                        sch.parallel(output_inner)
                    sch.bind(bx, "blockIdx.x")

                    # Move/in-line trailing injective blocks first so the
                    # second reduction can be legally compute_at'd to bx.
                    for epilogue_name in reversed(trailing_epilogue_names):
                        try:
                            _schedule_epilogue_block(
                                sch,
                                epilogue_name,
                                bx,
                                is_output_block=(epilogue_name in output_block_names),
                                cache_write_back=(not is_second_square_single_source),
                            )
                        except Exception:
                            try:
                                sch.compute_inline(sch.get_block(epilogue_name))
                            except Exception:
                                return None

                    if is_second_square_single_source:
                        sch.compute_at(
                            sch.get_block(second_reduction_name),
                            bx,
                            preserve_unit_loops=True,
                            index=0,
                        )
                    else:
                        sch.compute_at(
                            sch.get_block(second_reduction_name),
                            bx,
                            preserve_unit_loops=True,
                        )
                    for bridge_name in reversed(bridge_names):
                        sch.compute_at(
                            sch.get_block(bridge_name),
                            bx,
                            preserve_unit_loops=True,
                            index=0,
                        )
                    sch.compute_at(
                        sch.get_block(first_reduction_name),
                        bx,
                        preserve_unit_loops=True,
                        index=0,
                    )

                    first_thread_extent = _schedule_reduction_stage_at_bx(
                        sch,
                        first_reduction_name,
                        bx,
                        target,
                        output_block_names,
                        force_explicit_update=False,
                        cache_reduce_write_back=(first_reduction_name in output_block_names),
                    )
                    if first_thread_extent is None:
                        return None
                    second_thread_extent = _schedule_reduction_stage_at_bx(
                        sch,
                        second_reduction_name,
                        bx,
                        target,
                        output_block_names,
                        force_explicit_update=False,
                        cache_reduce_write_back=(second_reduction_name in output_block_names),
                    )
                    if second_thread_extent is None:
                        return None

                    for bridge_name in bridge_names:
                        _cache_elementwise_block(
                            sch,
                            bridge_name,
                            bx,
                            cache_write_back=False,
                        )
                        bridge_block = sch.get_block(bridge_name)
                        bridge_loops = sch.get_loops(bridge_block)
                        if bridge_loops:
                            sch.parallel(bridge_loops[-1])

                    launch_threads = num_threads
                    if not is_second_multi_source:
                        launch_threads = max(
                            launch_threads,
                            _choose_num_threads(target, first_thread_extent),
                            _choose_num_threads(target, second_thread_extent),
                        )
                    intermediate_names = [
                        first_reduction_name,
                        second_reduction_name,
                        *bridge_names,
                        *trailing_epilogue_names,
                    ]
                    for block_name in dict.fromkeys(intermediate_names):
                        if block_name in output_block_names:
                            continue
                        try:
                            block = sch.get_block(block_name)
                        except Exception:  # pragma: no cover
                            continue
                        for write_buffer_index, _ in enumerate(sch.get(block).writes):
                            sch.set_scope(block, write_buffer_index, "local.fragment")
                            block = sch.get_block(block_name)
                    sch.launch_thread(sch.get_block("root"), launch_threads)
                    return sch

            # Fallback for unsupported multi-reduction patterns.
            class _FallbackConfig:  # pylint: disable=too-few-public-methods
                block = [1]
                thread = [1]
                reduce_thread = [_choose_num_threads(target, tir.IntImm("int32", 256))]

            return self.sch_mutiple_reductions_with_config(func, _FallbackConfig())

        reduction_index = reduction_indices[-1]

        for info in block_infos[:reduction_index]:
            if (not info.is_reduction()) and (not info.is_injective()):
                return None

        for info in block_infos[reduction_index + 1 :]:
            if not info.is_injective():
                return None
        prologue_names = [sch.get(info.block_rv).name_hint for info in block_infos[:reduction_index]]
        epilogue_names = [sch.get(info.block_rv).name_hint for info in block_infos[reduction_index + 1 :]]
        reduction_info = block_infos[reduction_index]
        reduction_block = reduction_info.block_rv
        reduction_stmt = sch.get(reduction_block)
        if not reduction_info.is_reduction() or len(reduction_stmt.writes) != 1 or len(reduction_stmt.reads) < 1:
            return None

        update_info = _analyze_reduction_update(reduction_stmt)
        if update_info is None:
            return None
        reduce_type, rhs_expr = update_info

        write_buffer = reduction_stmt.writes[0].buffer
        write_buffer_index = _find_buffer_index(reduction_stmt.writes, write_buffer)
        if write_buffer_index is None:
            return None
        input_buffers = _collect_input_buffers(rhs_expr, write_buffer)
        if len(input_buffers) == 0:
            return None

        read_buffer_indices: list[int] = []
        for input_buffer in input_buffers:
            read_index = _find_buffer_index(reduction_stmt.reads, input_buffer)
            if read_index is None:
                return None
            read_buffer_indices.append(read_index)
        single_source_square = (
            len(input_buffers) == 1
            and len(set(read_buffer_indices)) == 1
            and reduce_type == "sum"
            and _is_square_of_buffer_load(rhs_expr, input_buffers[0])
        )

        init_value = _infer_init_value(reduction_stmt, reduce_type)
        reduction_name = reduction_stmt.name_hint

        if any(name in output_block_names for name in prologue_names):
            return None
        if any(name in output_block_names for name in epilogue_names[:-1]):
            # Keep the first tile-template version simple: only support the
            # common case where the trailing block is the output epilogue.
            return None
        reduction_dom_kind = reduction_info.dom_kind()
        num_leading_s = len(reduction_dom_kind) - len(reduction_dom_kind.lstrip("S"))
        has_reduction_prologue = any(info.is_reduction() for info in block_infos[:reduction_index])
        use_output_epilogue_anchor = len(epilogue_names) > 0 and epilogue_names[-1] in output_block_names

        if use_output_epilogue_anchor:
            anchor_info = block_infos[-1]
            anchor_block = anchor_info.block_rv
            anchor_loops = sch.get_loops(anchor_block)
            if num_leading_s <= 0 or len(anchor_loops) < num_leading_s:
                return None
            if not all(iter_info.kind in ("S", "O") for iter_info in anchor_info.iters):
                return None

            anchor_bx_loops = list(anchor_loops[:num_leading_s])
            anchor_inner_loops = list(anchor_loops[num_leading_s:])
            anchor_s_fused = sch.fuse(*anchor_bx_loops) if len(anchor_bx_loops) > 1 else anchor_bx_loops[0]
            anchor_tile = 1
            if has_reduction_prologue:
                lead_extent = _as_const_int(sch.get(anchor_s_fused).extent)
                if lead_extent is not None and lead_extent > 1:
                    anchor_tile = min(8, lead_extent)
                    while anchor_tile > 1 and lead_extent % anchor_tile != 0:
                        anchor_tile -= 1
                    if anchor_tile <= 1:
                        anchor_tile = 1
            bx, anchor_inner = sch.split(
                anchor_s_fused,
                factors=[None, anchor_tile],
                preserve_unit_iters=True,
            )
            if anchor_inner_loops:
                sch.reorder(bx, anchor_inner, *anchor_inner_loops)
                anchor_inner_fused = sch.fuse(*anchor_inner_loops) if len(anchor_inner_loops) > 1 else anchor_inner_loops[0]
                sch.parallel(anchor_inner_fused)
            else:
                sch.parallel(anchor_inner)

            reduction_block = sch.get_block(reduction_name)
            sch.compute_at(reduction_block, bx, preserve_unit_loops=True)

            for epilogue_name in epilogue_names[:-1]:
                _schedule_epilogue_block(
                    sch,
                    epilogue_name,
                    bx,
                    is_output_block=(epilogue_name in output_block_names),
                )
            _cache_elementwise_block(
                sch,
                epilogue_names[-1],
                bx,
                cache_reads=False,
                cache_writes=False,
            )

            reduction_block = sch.get_block(reduction_name)
            reduction_loops = sch.get_loops(reduction_block)
            if len(reduction_loops) == 0:
                return None
            r_fused = reduction_loops[-1]

            reduce_step = _choose_reduction_step(target, sch.get(r_fused).extent)
            if reduce_step is not None:
                ro, ri = sch.split(r_fused, factors=[None, reduce_step], preserve_unit_iters=True)
                cache_read_loop = ro
                reduce_loop: tir.schedule.LoopRV | None = ro
                thread_extent_expr = sch.get(ri).extent
            else:
                cache_read_loop = bx
                reduce_loop = None
                thread_extent_expr = sch.get(r_fused).extent
        else:
            s_loops: list[tir.schedule.LoopRV] = []
            r_loops: list[tir.schedule.LoopRV] = []
            o_loops: list[tir.schedule.LoopRV] = []
            for iter_info in reduction_info.iters:
                if iter_info.kind == "S":
                    s_loops.append(iter_info.loop_rv)
                elif iter_info.kind == "R":
                    r_loops.append(iter_info.loop_rv)
                elif iter_info.kind == "O":
                    o_loops.append(iter_info.loop_rv)
                else:
                    return None

            if not s_loops or not r_loops:
                return None

            sch.reorder(*s_loops, *r_loops, *o_loops)
            s_fused = sch.fuse(*s_loops) if len(s_loops) > 1 else s_loops[0]
            r_fused = sch.fuse(*r_loops) if len(r_loops) > 1 else r_loops[0]

            # One output element per CTA, same strategy as the reduction template.
            bx, inner_s = sch.split(s_fused, factors=[None, 1], preserve_unit_iters=True)
            sch.parallel(inner_s)

            reduce_step = _choose_reduction_step(target, sch.get(r_fused).extent)
            if reduce_step is not None:
                ro, ri = sch.split(r_fused, factors=[None, reduce_step], preserve_unit_iters=True)
                sch.reorder(bx, inner_s, ro, ri, *o_loops)
                cache_read_loop = ro
                reduce_loop = ro
                thread_extent_expr = sch.get(ri).extent
            else:
                sch.reorder(bx, inner_s, r_fused, *o_loops)
                cache_read_loop = bx
                reduce_loop = inner_s
                thread_extent_expr = sch.get(r_fused).extent

        # Stage all input regions used by the reduction update expression.
        for read_buffer_index in sorted(set(read_buffer_indices)):
            reduction_block = sch.get_block(reduction_name)
            sch.cache_read_at(cache_read_loop, reduction_block, read_buffer_index, "local.fragment")

        # Cache + initialize reduction output tile.
        reduction_block = sch.get_block(reduction_name)
        if reduction_name not in output_block_names:
            sch.set_scope(reduction_block, write_buffer_index, "local.fragment")
            reduction_block = sch.get_block(reduction_name)
        sch.cache_reduce_at(
            bx,
            reduction_block,
            write_buffer_index,
            "local.fragment",
            init_value,
            write_back=(reduction_name in output_block_names or not use_output_epilogue_anchor),
        )

        # Lower to tile-level T.reduce only for single-source reductions.
        can_lower_main_reduce = (
            len(read_buffer_indices) == 1
            and reduce_loop is not None
            and (_is_direct_buffer_load(rhs_expr, input_buffers[0]) or single_source_square)
        )
        if can_lower_main_reduce:
            read_buffer_index = read_buffer_indices[0]
            reduce_type_for_lower = "sumsq" if single_source_square else reduce_type
            reduction_block = sch.get_block(reduction_name)
            reduction_stmt = sch.get(reduction_block)
            reduce_dim = _infer_reduce_dim(
                reduction_stmt.reads[read_buffer_index].buffer,
                reduction_stmt.writes[write_buffer_index].buffer,
            )
            sch.reduce_at(
                reduce_loop,
                reduction_block,
                read_buffer_index=read_buffer_index,
                write_buffer_index=write_buffer_index,
                reduce_type=reduce_type_for_lower,
                dim=reduce_dim,
                clear=False,
                replace_loop_body=True,
            )

        if not use_output_epilogue_anchor:
            for epilogue_name in reversed(epilogue_names):
                _schedule_epilogue_block(
                    sch,
                    epilogue_name,
                    bx,
                    is_output_block=(epilogue_name in output_block_names),
                )
        for prologue_name in prologue_names:
            prologue_block = sch.get_block(prologue_name)
            if sch.get(prologue_block).init is not None:
                # Compute-at for reduction prologues may duplicate full-grid work.
                # Keep them at root for now.
                continue
            _schedule_prologue_block(sch, prologue_name, bx)

        # TileLang CUDA codegen does not allow in-kernel global allocations.
        # After placement/caching, force intermediate block writes to
        # local.fragment so remaining temporary buffers are compilable.
        intermediate_names = []
        for name in prologue_names:
            if name not in output_block_names:
                intermediate_names.append(name)
        if reduction_name not in output_block_names:
            intermediate_names.append(reduction_name)
        for name in epilogue_names:
            if name not in output_block_names:
                intermediate_names.append(name)

        for block_name in dict.fromkeys(intermediate_names):
            try:
                block = sch.get_block(block_name)
            except Exception:  # pragma: no cover
                continue
            for write_buffer_index, _ in enumerate(sch.get(block).writes):
                sch.set_scope(block, write_buffer_index, "local.fragment")
                block = sch.get_block(block_name)

        num_threads = _choose_num_threads(target, thread_extent_expr)
        sch.bind(bx, "blockIdx.x")
        launch_block = sch.get_block("root")
        if has_reduction_prologue:
            try:
                launch_block = sch.get_block(reduction_name)
            except Exception:  # pragma: no cover
                launch_block = sch.get_block("root")
        sch.launch_thread(launch_block, num_threads)
        return sch

    def sch_mutiple_reductions_with_config(  # pylint: disable=too-many-locals,too-many-branches,too-many-return-statements
        self,
        func: tir.PrimFunc,
        config,
    ):
        block_factors = config.block
        thread_factors = config.thread
        reduce_therad_factors = config.reduce_thread

        sch = tir.Schedule(func)
        block_infos = normalize_prim_func(sch)
        block_infos = try_inline_contiguous_spatial(sch, block_infos)
        if block_infos is None or len(block_infos) == 0:
            return None

        def prod(iterable):
            return reduce(lambda x, y: x * y, iterable, 1)

        len_tx = prod(thread_factors) * prod(reduce_therad_factors)
        block_factor = prod(block_factors)

        dom_kind = block_infos[0].dom_kind()
        num_leading_s = len(dom_kind) - len(dom_kind.lstrip("S"))
        num_trailing_r = len(dom_kind) - len(dom_kind.rstrip("R"))

        # Align the number of block iters of the last block.
        num_last_block_iter = len(block_infos[-1].dom_kind())
        if num_last_block_iter < len(dom_kind):
            index_map = tir.IndexMap.from_func(
                lambda *iters: ([tir.const(0, iters[0].dtype)] * (len(dom_kind) - num_last_block_iter) + list(iters)),
                ndim=num_last_block_iter,
            )
            sch.transform_block_layout(block_infos[-1].block_rv, index_map)

        try:
            # TODO: fix num_leading_s = 0 case
            assert num_trailing_r > 0
            for block in block_infos[1:-1]:
                assert block.dom_kind() == dom_kind
            assert block_infos[-1].is_injective()
            assert len(block_infos[-1].dom_kind()) <= len(dom_kind)
        except AssertionError:
            return None

        loops = sch.get_loops(block_infos[-1].block_rv)
        bx, _ = sch.split(sch.fuse(*loops[:num_leading_s]), factors=[None, block_factor])
        r_loop, tx = sch.split(loops[-1], [None, len_tx])
        sch.reorder(tx, r_loop)
        sch.bind(bx, "blockIdx.x")
        sch.bind(tx, "threadIdx.x")

        for block in reversed(block_infos[:-1]):
            block = block.block_rv
            for i, _ in enumerate(sch.get(block).writes):
                sch.set_scope(block, buffer_index=i, storage_scope="shared")
            sch.compute_at(block, bx, preserve_unit_loops=True)
            r_loop = sch.fuse(*sch.get_loops(block)[-num_trailing_r:])
            r_loop, tx = sch.split(r_loop, [None, len_tx])
            sch.reorder(tx, r_loop)
            sch.bind(tx, "threadIdx.x")

        return sch
