# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# pylint: disable=invalid-name
"""A tile-primitive-first GEMV schedule rule for TileLang."""

from typing import List, Optional, Union

from tvm import DataType, tir
from tvm.target import Target

from .. import Schedule as TileSchedule
from tvm.dlight import normalize_prim_func, try_inline_contiguous_spatial
from tvm.dlight.analysis import BlockInfo, is_gemv, normalize
from . import utils
from .base import GPUScheduleRule
from .element_wise import _resolve_target_from_config
from .reduction import _find_buffer_index


def _as_const_int(expr: tir.PrimExpr) -> Optional[int]:
    if isinstance(expr, tir.IntImm):
        return int(expr.value)
    return None


def _dtype_bytes(dtype: str) -> int:
    return max(1, DataType(dtype).bits // 8)


def _largest_power_of_two_at_most(value: int) -> int:
    result = 1
    while result << 1 <= value:
        result <<= 1
    return result


def _largest_divisor_not_exceeding(extent: int, cap: int, step: int = 1) -> int:
    step = max(step, 1)
    tile = min(extent, cap)
    if step > 1:
        tile = max(step, tile // step * step)
    while tile > step and extent % tile != 0:
        tile -= step
    if tile <= 0:
        return step if extent >= step else 1
    return tile


def _choose_reduction_tile(
    target: Target,
    reduction_extent: tir.PrimExpr,
    dtype: str,
    prefer_large_tile: bool,
) -> int:
    del target
    dtype_bits = DataType(dtype).bits
    vec_len = max(1, 128 // max(dtype_bits, 8))
    const_extent = _as_const_int(reduction_extent)
    base_tile = vec_len * (64 if prefer_large_tile and const_extent is not None and const_extent >= 8192 else 32)
    if const_extent is None:
        return base_tile
    if const_extent <= 0:
        return 1
    return _largest_divisor_not_exceeding(const_extent, base_tile, vec_len)


def _choose_output_tile(
    target: Target,
    spatial_extent: tir.PrimExpr,
    reduction_tile: int,
    dtype: str,
) -> int:
    max_shared = target.attrs.get("max_shared_memory_per_block", 49152)
    tile_cap = max_shared // max(1, reduction_tile * _dtype_bytes(dtype))
    target_matrix_tile_bytes = 8192
    preferred_rows = target_matrix_tile_bytes // max(1, reduction_tile * _dtype_bytes(dtype))
    tile_cap = max(1, min(tile_cap, 128, max(1, preferred_rows)))
    const_extent = _as_const_int(spatial_extent)
    if const_extent is None:
        return _largest_power_of_two_at_most(max(16, tile_cap))
    if const_extent <= 0:
        return 1
    if const_extent >= 16:
        tile_cap = max(tile_cap, 16)
    tile_cap = min(tile_cap, const_extent)
    tile = _largest_power_of_two_at_most(max(1, tile_cap))
    while tile > 1 and const_extent % tile != 0:
        tile >>= 1
    return max(tile, 1)


def _choose_launch_threads(target: Target, output_tile: int, reduction_tile: int) -> int:
    del reduction_tile
    max_threads = min(int(utils.max_threads_per_block(target)), 256)
    if max_threads <= 0:
        return 1
    return max(1, _largest_power_of_two_at_most(min(max_threads, max(output_tile, 1))))


def _find_epilogue_name(block_infos: List[BlockInfo], sch: TileSchedule) -> Optional[str]:
    if len(block_infos) == 1:
        return None
    if len(block_infos) != 2 or not block_infos[1].is_injective():
        return None
    return sch.get(block_infos[1].block_rv).name_hint


class GEMV(GPUScheduleRule):
    """A tile-based GEMV rule using TileLang cache/tile primitives."""

    def apply(
        self,
        func: tir.PrimFunc,
        target: Target,
        _: bool,
    ) -> Union[None, tir.Schedule, List[tir.Schedule]]:
        if not isinstance(func, tir.PrimFunc) or not self.is_target_available(target):
            return None

        sch = TileSchedule(func)
        block_infos = normalize_prim_func(sch)
        block_infos = try_inline_contiguous_spatial(sch, block_infos)
        if block_infos is None:
            return None

        epilogue_name = _find_epilogue_name(block_infos, sch)
        if epilogue_name is None and len(block_infos) != 1:
            return None

        block_info = block_infos[0]
        if len(block_info.iters) not in [2, 3]:
            return None

        block = block_info.block_rv
        vector_input_buffers = is_gemv(sch, block_info)
        if vector_input_buffers is None or len(vector_input_buffers) != 1:
            return None

        is_inner_reduction = normalize(sch, block_info)
        if is_inner_reduction is None or not is_inner_reduction:
            return None

        block_stmt = sch.get(block)
        if len(block_stmt.writes) != 1:
            return None

        vector_buffer = vector_input_buffers[0]
        matrix_buffer = None
        vector_read_index = None
        matrix_read_index = None
        for read_region in block_stmt.reads:
            if not read_region.buffer.same_as(vector_buffer):
                matrix_buffer = read_region.buffer
                break
        if matrix_buffer is None:
            return None
        vector_read_index = _find_buffer_index(block_stmt.reads, vector_buffer)
        matrix_read_index = _find_buffer_index(block_stmt.reads, matrix_buffer)
        prefer_large_reduction_tile = (
            vector_read_index is not None
            and matrix_read_index is not None
            and matrix_read_index < vector_read_index
        )

        batch, s, r, c = sch.get_loops(block)
        reduction_tile = _choose_reduction_tile(
            target,
            sch.get(r).extent,
            matrix_buffer.dtype,
            prefer_large_reduction_tile,
        )
        output_tile = _choose_output_tile(target, sch.get(s).extent, reduction_tile, matrix_buffer.dtype)

        bo, bi = sch.split(s, factors=[None, output_tile], preserve_unit_iters=True)
        ro, ri = sch.split(r, factors=[None, reduction_tile], preserve_unit_iters=True)
        sch.reorder(batch, bo, ro, bi, ri, c)
        block_outer = sch.fuse(batch, bo)

        c_extent = _as_const_int(sch.get(c).extent)
        if c_extent is not None and c_extent > 1:
            sch.vectorize(c)

        sch.parallel(bi)
        block_name = sch.get(block).name_hint

        if epilogue_name is None:
            sch.decompose_reduction(block, ro)
            update_name = block_name + "_update"
            update_block = sch.get_block(update_name)
            update_stmt = sch.get(update_block)

            matrix_read_index = _find_buffer_index(update_stmt.reads, matrix_buffer)
            vector_read_index = _find_buffer_index(update_stmt.reads, vector_buffer)
            write_buffer_index = _find_buffer_index(
                update_stmt.writes, update_stmt.writes[0].buffer
            )
            if (
                matrix_read_index is None
                or vector_read_index is None
                or write_buffer_index is None
            ):
                return None

            sch.cache_read_at(ro, update_block, matrix_read_index, "local.fragment")
            update_block = sch.get_block(update_name)
            update_stmt = sch.get(update_block)
            vector_read_index = _find_buffer_index(update_stmt.reads, vector_buffer)
            if vector_read_index is None:
                return None
            sch.cache_read_at(ro, update_block, vector_read_index, "local.fragment")

            sch.cache_reduce_at(
                block_outer,
                sch.get_block(update_name),
                write_buffer_index,
                "shared.dyn",
                0.0,
                write_back=True,
            )
        else:
            sch.decompose_reduction(block, ro)
            update_name = block_name + "_update"
            update_block = sch.get_block(update_name)
            update_stmt = sch.get(update_block)

            matrix_read_index = _find_buffer_index(update_stmt.reads, matrix_buffer)
            vector_read_index = _find_buffer_index(update_stmt.reads, vector_buffer)
            if matrix_read_index is None or vector_read_index is None:
                return None

            sch.cache_read_at(ro, update_block, matrix_read_index, "local.fragment")
            update_block = sch.get_block(update_name)
            update_stmt = sch.get(update_block)
            vector_read_index = _find_buffer_index(update_stmt.reads, vector_buffer)
            if vector_read_index is None:
                return None
            sch.cache_read_at(ro, update_block, vector_read_index, "local.fragment")

            epilogue = sch.get_block(epilogue_name)
            sch.reverse_compute_at(epilogue, block_outer, preserve_unit_loops=True)
            epilogue = sch.get_block(epilogue_name)
            epilogue_loops = sch.get_loops(epilogue)
            if epilogue_loops:
                sch.parallel(epilogue_loops[-1])

        sch.bind(block_outer, "blockIdx.x")
        sch.launch_thread(
            sch.get_block("root"),
            _choose_launch_threads(target, output_tile, reduction_tile),
        )
        return sch

    def apply_config(
        self, func: tir.PrimFunc, config
    ) -> Union[None, tir.Schedule, List[tir.Schedule]]:
        target = _resolve_target_from_config(config)
        return self.apply(func, target, False)
