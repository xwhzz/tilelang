# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# pylint: disable=invalid-name
"""LayerNorm-like reduction schedule using tile-level primitives."""

from __future__ import annotations

from tilelang import tvm

from .. import Schedule as TileSchedule
from tilelang.carver.common_schedules import get_output_blocks
from .base import GPUScheduleRule
from .reduction import (
    _analyze_reduction_update,
    _choose_num_threads,
    _choose_reduction_step,
    _find_buffer_index,
    _infer_init_value,
    _infer_reduce_dim,
)

tir = tvm.tir
Target = tvm.target.Target
normalize_prim_func = tvm.dlight.normalize_prim_func
BlockInfo = tvm.dlight.analysis.BlockInfo


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


def _schedule_single_source_reduction(
    sch: TileSchedule,
    block_name: str,
    bx: tir.schedule.LoopRV,
    target: Target,
    write_back: bool = False,
    force_reduce_type: str | None = None,
) -> tir.PrimExpr | None:
    block = sch.get_block(block_name)
    block_stmt = sch.get(block)
    if len(block_stmt.writes) != 1:
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
    if len(input_buffers) != 1:
        return None
    read_buffer_index = _find_buffer_index(block_stmt.reads, input_buffers[0])
    if read_buffer_index is None:
        return None

    reduction_loops = sch.get_loops(block)
    if len(reduction_loops) == 0:
        return None
    r_fused = reduction_loops[-1]

    reduce_step = _choose_reduction_step(target, sch.get(r_fused).extent)
    if reduce_step is not None:
        ro, ri = sch.split(r_fused, factors=[None, reduce_step], preserve_unit_iters=True)
        cache_read_loop = ro
        reduce_loop = ro
        thread_extent_expr = sch.get(ri).extent
    else:
        cache_read_loop = bx
        reduce_loop = r_fused
        thread_extent_expr = sch.get(r_fused).extent

    sch.cache_read_at(cache_read_loop, block, read_buffer_index, "local.fragment")

    block = sch.get_block(block_name)
    init_value = _infer_init_value(block_stmt, reduce_type)
    sch.cache_reduce_at(
        bx,
        block,
        write_buffer_index,
        "local.fragment",
        init_value,
        write_back=write_back,
    )

    block = sch.get_block(block_name)
    block_stmt = sch.get(block)
    if len(block_stmt.reads) != 1 or len(block_stmt.writes) != 1:
        return None
    reduce_dim = _infer_reduce_dim(
        block_stmt.reads[0].buffer,
        block_stmt.writes[0].buffer,
    )
    reduce_type_for_lower = force_reduce_type if force_reduce_type is not None else reduce_type
    sch.reduce_at(
        reduce_loop,
        block,
        read_buffer_index=0,
        write_buffer_index=0,
        reduce_type=reduce_type_for_lower,
        dim=reduce_dim,
        clear=False,
        replace_loop_body=True,
    )
    return thread_extent_expr


def _schedule_center_bridge(
    sch: TileSchedule,
    block_name: str,
    bx: tir.schedule.LoopRV,
) -> bool:
    block = sch.get_block(block_name)
    block_stmt = sch.get(block)
    if len(block_stmt.writes) != 1:
        return False

    write_buffer = block_stmt.writes[0].buffer
    for read_buffer_index, read_region in enumerate(block_stmt.reads):
        if read_region.buffer.same_as(write_buffer):
            continue
        sch.cache_read_at(bx, block, read_buffer_index, "local.fragment")
        block = sch.get_block(block_name)

    sch.cache_write_at(bx, block, 0, "local.fragment", write_back=False)
    block = sch.get_block(block_name)
    loops = sch.get_loops(block)
    if loops:
        sch.parallel(loops[-1])
    return True


class LayerNormLike(GPUScheduleRule):
    """LayerNorm-like two-reduction schedule rule."""

    def apply(
        self,
        func: tir.PrimFunc,
        target: Target,
        _: bool,
    ) -> None | tir.Schedule | list[tir.Schedule]:
        if not isinstance(func, tir.PrimFunc) or not self.is_target_available(target):
            return None

        sch = TileSchedule(func)
        block_infos = normalize_prim_func(sch)
        if block_infos is None:
            return None

        reduction_indices = [i for i, info in enumerate(block_infos) if info.is_reduction()]
        if len(reduction_indices) != 2:
            return None
        first_reduction_idx = reduction_indices[0]
        for info in block_infos[:first_reduction_idx]:
            if not info.is_injective():
                return None
            sch.compute_inline(info.block_rv)

        block_infos = normalize_prim_func(sch)
        if block_infos is None:
            return None

        reduction_indices = [i for i, info in enumerate(block_infos) if info.is_reduction()]
        if len(reduction_indices) != 2:
            return None
        first_reduction_idx, second_reduction_idx = reduction_indices
        first_reduction_info = block_infos[first_reduction_idx]
        second_reduction_info = block_infos[second_reduction_idx]
        bridge_infos = block_infos[first_reduction_idx + 1 : second_reduction_idx]
        trailing_infos = block_infos[second_reduction_idx + 1 :]
        if len(bridge_infos) == 0 or len(trailing_infos) == 0:
            return None
        if not all(info.is_injective() for info in bridge_infos):
            return None
        if not all(info.is_injective() for info in trailing_infos):
            return None

        output_block_names = {sch.get(output_block).name_hint for output_block in get_output_blocks(sch, block_infos)}
        output_name = sch.get(trailing_infos[-1].block_rv).name_hint
        if output_name not in output_block_names:
            return None

        first_reduction_stmt = sch.get(first_reduction_info.block_rv)
        second_reduction_stmt = sch.get(second_reduction_info.block_rv)
        first_update_info = _analyze_reduction_update(first_reduction_stmt)
        second_update_info = _analyze_reduction_update(second_reduction_stmt)
        if first_update_info is None or second_update_info is None:
            return None
        first_reduce_type, _ = first_update_info
        second_reduce_type, second_rhs_expr = second_update_info
        if first_reduce_type != "sum" or second_reduce_type != "sum":
            return None
        if len(second_reduction_stmt.writes) != 1:
            return None
        second_input_buffers = _collect_input_buffers(
            second_rhs_expr,
            second_reduction_stmt.writes[0].buffer,
        )
        if len(second_input_buffers) != 1:
            return None
        center_buffer = second_input_buffers[0]
        if not _is_square_of_buffer_load(second_rhs_expr, center_buffer):
            return None

        center_name: str | None = None
        for info in bridge_infos:
            block_stmt = sch.get(info.block_rv)
            if _block_writes_buffer(block_stmt, center_buffer):
                center_name = block_stmt.name_hint
                break
        if center_name is None:
            return None

        first_reduction_name = first_reduction_stmt.name_hint
        second_reduction_name = second_reduction_stmt.name_hint

        for info in reversed(bridge_infos):
            name = sch.get(info.block_rv).name_hint
            if name == center_name:
                continue
            sch.compute_inline(info.block_rv)
        for info in reversed(trailing_infos[:-1]):
            sch.compute_inline(info.block_rv)

        block_infos = normalize_prim_func(sch)
        if block_infos is None:
            return None
        block_infos_by_name: dict[str, BlockInfo] = {sch.get(info.block_rv).name_hint: info for info in block_infos}
        if (
            first_reduction_name not in block_infos_by_name
            or second_reduction_name not in block_infos_by_name
            or center_name not in block_infos_by_name
            or output_name not in block_infos_by_name
        ):
            return None

        second_dom_kind = block_infos_by_name[second_reduction_name].dom_kind()
        num_leading_s = len(second_dom_kind) - len(second_dom_kind.lstrip("S"))
        if num_leading_s <= 0:
            return None

        output_block = sch.get_block(output_name)
        output_loops = sch.get_loops(output_block)
        if len(output_loops) < num_leading_s:
            return None

        output_bx_loops = list(output_loops[:num_leading_s])
        output_inner_loops = list(output_loops[num_leading_s:])
        output_s_fused = sch.fuse(*output_bx_loops) if len(output_bx_loops) > 1 else output_bx_loops[0]
        bx, output_inner = sch.split(
            output_s_fused,
            factors=[None, 1],
            preserve_unit_iters=True,
        )
        if output_inner_loops:
            sch.reorder(bx, output_inner, *output_inner_loops)
            output_parallel_loop = sch.fuse(*output_inner_loops) if len(output_inner_loops) > 1 else output_inner_loops[0]
            sch.parallel(output_parallel_loop)
        else:
            sch.reorder(bx, output_inner)
            output_parallel_loop = output_inner
            sch.parallel(output_parallel_loop)
        sch.bind(bx, "blockIdx.x")

        sch.compute_at(
            sch.get_block(second_reduction_name),
            bx,
            preserve_unit_loops=True,
            index=0,
        )
        sch.compute_at(
            sch.get_block(center_name),
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

        first_thread_extent = _schedule_single_source_reduction(
            sch,
            first_reduction_name,
            bx,
            target,
            write_back=False,
        )
        if first_thread_extent is None:
            return None

        if not _schedule_center_bridge(sch, center_name, bx):
            return None

        second_thread_extent = _schedule_single_source_reduction(
            sch,
            second_reduction_name,
            bx,
            target,
            write_back=False,
            force_reduce_type="sumsq",
        )
        if second_thread_extent is None:
            return None

        output_block = sch.get_block(output_name)
        output_stmt = sch.get(output_block)
        output_writes = [region.buffer for region in output_stmt.writes]
        for read_buffer_index, read_region in enumerate(output_stmt.reads):
            if any(read_region.buffer.same_as(write_buffer) for write_buffer in output_writes):
                continue
            sch.cache_read_at(bx, output_block, read_buffer_index, "local.fragment")
            output_block = sch.get_block(output_name)
        output_block = sch.get_block(output_name)
        output_loops = sch.get_loops(output_block)
        if output_loops:
            sch.parallel(output_loops[-1])
            output_parallel_extent = sch.get(output_loops[-1]).extent
        else:
            output_parallel_extent = tir.IntImm("int32", 1)

        num_threads = max(
            _choose_num_threads(target, first_thread_extent),
            _choose_num_threads(target, second_thread_extent),
            _choose_num_threads(target, output_parallel_extent),
        )
        sch.launch_thread(sch.get_block("root"), num_threads)
        return sch
