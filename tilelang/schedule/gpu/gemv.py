# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# pylint: disable=invalid-name
"""A GEMV schedule rule for TileLang."""

from __future__ import annotations

from collections.abc import Sequence

import tilelang
from tilelang import tvm

from .. import Schedule as TileSchedule
from . import utils
from .base import GPUScheduleRule
from .element_wise import _resolve_target_from_config

tir = tvm.tir
Target = tvm.target.Target
normalize_prim_func = tvm.dlight.normalize_prim_func
try_inline_contiguous_spatial = tvm.dlight.try_inline_contiguous_spatial
BlockInfo = tvm.dlight.analysis.BlockInfo
is_gemv = tvm.dlight.analysis.is_gemv
normalize = tvm.dlight.analysis.normalize


def _as_const_int(expr: tir.PrimExpr) -> int | None:
    if isinstance(expr, tir.IntImm):
        return int(expr.value)
    return None


def _dtype_bytes(dtype: str) -> int:
    return max(1, tvm.DataType(dtype).bits // 8)


def _is_float_dtype(dtype: str) -> bool:
    return dtype.startswith("float") or dtype.startswith("bfloat")


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


def _max_dynamic_shared_bytes(target: Target) -> int:
    """Return the max dynamic shared memory per block for the target."""
    sm = utils.get_sm_version(target) if target.kind.name == "cuda" else 0
    if sm >= 90:
        return 228 * 1024  # Hopper (H100)
    if sm >= 80:
        return 163 * 1024  # Ampere (A100)
    if sm >= 70:
        return 96 * 1024  # Volta (V100)
    return 48 * 1024


def _choose_accumulator_dtype(
    matrix_buffer: tir.Buffer,
    vector_buffer: tir.Buffer,
    output_buffer: tir.Buffer,
) -> str | None:
    read_dtypes = {matrix_buffer.dtype, vector_buffer.dtype}
    if read_dtypes.issubset({"float16", "bfloat16"}) and output_buffer.dtype in {"float16", "bfloat16"}:
        return "float32"
    return None


def _find_epilogue_name(block_infos: list[BlockInfo], sch: TileSchedule) -> str | None:
    if len(block_infos) == 1:
        return None
    if len(block_infos) != 2 or not block_infos[1].is_injective():
        return None
    return sch.get(block_infos[1].block_rv).name_hint


def _choose_splitk_schedule_params(
    target: Target,
    spatial_extent: tir.PrimExpr,
    reduction_extent: tir.PrimExpr,
    matrix_dtype: str,
    output_dtype: str,
) -> Sequence[int]:
    bits = tvm.DataType(matrix_dtype).bits
    output_bits = tvm.DataType(output_dtype).bits
    max_threads = min(int(utils.max_threads_per_block(target)), 256)
    if max_threads <= 0:
        return 1, 1, 1

    # Hopper mixed-precision GEMV prefers fewer reduction threads (single
    # warp avoids cross-warp sync) with larger vectorization for coalesced
    # 128-bit loads and more work per thread.
    if (
        target.kind.name == "cuda"
        and utils.get_sm_version(target) >= 90
        and bits <= 16
        and output_bits > bits
    ):
        output_candidate = 1
        thread_candidate = 128
        vec_candidate = 2
    elif bits <= 16:
        output_candidate = 4
        thread_candidate = 64
        vec_candidate = 2
    else:
        output_candidate = 2
        thread_candidate = 64
        vec_candidate = 1

    const_spatial = _as_const_int(spatial_extent)
    if const_spatial is None:
        output_tile = output_candidate
    elif const_spatial <= 0:
        output_tile = 1
    else:
        output_tile = max(1, _largest_divisor_not_exceeding(const_spatial, output_candidate))

    while output_tile > 1 and output_tile * thread_candidate > max_threads:
        output_tile //= 2

    reduction_threads_cap = max(1, max_threads // max(output_tile, 1))
    reduction_threads = _largest_power_of_two_at_most(min(thread_candidate, reduction_threads_cap))

    const_reduction = _as_const_int(reduction_extent)
    if const_reduction is None:
        vec = vec_candidate
    elif const_reduction <= 1:
        vec = 1
    elif vec_candidate > 1 and const_reduction % vec_candidate == 0:
        vec = vec_candidate
    else:
        vec = 1

    if const_reduction is not None and const_reduction > 0:
        tiles = max(1, (const_reduction + vec - 1) // vec)
        reduction_threads = max(
            1,
            _largest_power_of_two_at_most(min(reduction_threads, tiles)),
        )

    return output_tile, reduction_threads, vec


def _can_use_splitk_schedule(
    target: Target,
    epilogue_name: str | None,
    matrix_buffer: tir.Buffer,
    vector_buffer: tir.Buffer,
    output_buffer: tir.Buffer,
) -> bool:
    if target.kind.name != "cuda":
        return False
    matrix_dtype = tvm.DataType(matrix_buffer.dtype)
    vector_dtype = tvm.DataType(vector_buffer.dtype)
    output_dtype = tvm.DataType(output_buffer.dtype)
    if not _is_float_dtype(matrix_buffer.dtype) or not _is_float_dtype(vector_buffer.dtype) or not _is_float_dtype(output_buffer.dtype):
        return False
    if matrix_dtype.bits not in (16, 32) or vector_dtype.bits not in (16, 32) or output_dtype.bits not in (16, 32):
        return False
    if output_dtype.bits < max(matrix_dtype.bits, vector_dtype.bits):
        return False
    # When the schedule would need to promote the accumulator dtype (e.g.
    # fp16 in / fp16 out), the recommended approach is to express fp32
    # accumulation directly in the TE expression so this check passes.
    if _choose_accumulator_dtype(matrix_buffer, vector_buffer, output_buffer) is not None:
        return False
    # Epilogues stage the GEMV output in dynamic shared memory.  Since the
    # buffer is not compacted, its full size must fit within the device
    # dynamic shared memory limit.
    if epilogue_name is not None:
        buf_bytes = _dtype_bytes(output_buffer.dtype)
        for dim in output_buffer.shape:
            dim_val = _as_const_int(dim)
            if dim_val is None:
                return False
            buf_bytes *= dim_val
        if buf_bytes > _max_dynamic_shared_bytes(target):
            return False
    if len(matrix_buffer.shape) not in (2, 3):
        return False
    if len(vector_buffer.shape) + 1 != len(matrix_buffer.shape):
        return False
    return len(output_buffer.shape) + 1 == len(matrix_buffer.shape)


class GEMV(GPUScheduleRule):
    """A GEMV schedule rule using a CUDA split-K fast path."""

    def _apply_splitk_fast_path(
        self,
        func: tir.PrimFunc,
        target: Target,
        epilogue_name: str | None,
        matrix_buffer: tir.Buffer,
        vector_buffer: tir.Buffer,
        output_buffer: tir.Buffer,
    ) -> tir.Schedule | None:
        if not _can_use_splitk_schedule(
            target,
            epilogue_name,
            matrix_buffer,
            vector_buffer,
            output_buffer,
        ):
            return None

        try:
            sch = tir.Schedule(func)
            block_infos = normalize_prim_func(sch)
            block_infos = try_inline_contiguous_spatial(sch, block_infos)
            if block_infos is None or len(block_infos) not in (1, 2):
                return None

            epilogue = None
            if len(block_infos) == 2:
                if not block_infos[1].is_injective():
                    return None
                epilogue = block_infos[1].block_rv

            block_info = block_infos[0]
            block = block_info.block_rv
            if normalize(sch, block_info) is not True:
                return None

            batch, s, r, c = sch.get_loops(block)
            output_tile, reduce_threads, vec = _choose_splitk_schedule_params(
                target,
                sch.get(s).extent,
                sch.get(r).extent,
                matrix_buffer.dtype,
                output_buffer.dtype,
            )

            if output_tile > 1:
                bo, bi = sch.split(s, factors=[None, output_tile], preserve_unit_iters=True)
            else:
                bo, bi = s, None
            if vec > 1:
                ro, tx, vec_loop = sch.split(r, factors=[None, reduce_threads, vec], preserve_unit_iters=True)
                if bi is not None:
                    sch.reorder(batch, bo, bi, ro, tx, vec_loop, c)
                else:
                    sch.reorder(batch, bo, ro, tx, vec_loop, c)
                sch.fuse(vec_loop, c, preserve_unit_iters=True)
            else:
                ro, tx = sch.split(r, factors=[None, reduce_threads], preserve_unit_iters=True)
                if bi is not None:
                    sch.reorder(batch, bo, bi, ro, tx, c)
                else:
                    sch.reorder(batch, bo, ro, tx, c)
                tx = sch.fuse(tx, c, preserve_unit_iters=True)

            block_outer = sch.fuse(batch, bo)
            sch.bind(block_outer, "blockIdx.x")
            if bi is not None:
                sch.bind(bi, "threadIdx.y")
            sch.bind(tx, "threadIdx.x")

            if epilogue is not None:
                sch.set_scope(block, 0, "shared.dyn")
                sch.reverse_compute_at(epilogue, block_outer, preserve_unit_loops=True)

            mod = tir.transform.LowerCrossThreadReduction()(sch.mod)
            mod = tir.transform.LowerInitBlock()(mod)
            mod = tir.transform.ConvertBlocksToOpaque()(mod)
            mod = tilelang.transform.ReserveRootBlock()(mod)
            return TileSchedule(mod)
        except Exception:
            return None

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
        if is_inner_reduction is None:
            return None
        if not is_inner_reduction:
            return None

        block_stmt = sch.get(block)
        if len(block_stmt.writes) != 1:
            return None

        vector_buffer = vector_input_buffers[0]
        matrix_buffer = None
        for read_region in block_stmt.reads:
            if not read_region.buffer.same_as(vector_buffer):
                matrix_buffer = read_region.buffer
                break
        if matrix_buffer is None:
            return None
        output_buffer = block_stmt.writes[0].buffer

        return self._apply_splitk_fast_path(
            func,
            target,
            epilogue_name,
            matrix_buffer,
            vector_buffer,
            output_buffer,
        )

    def apply_config(self, func: tir.PrimFunc, config) -> None | tir.Schedule | list[tir.Schedule]:
        target = _resolve_target_from_config(config)
        return self.apply(func, target, False)
