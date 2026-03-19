# Copyright 2018 The apache/tvm Authors. All Rights Reserved.
# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
#
# Modifications Copyright (c) Microsoft.
"""A tile-primitive-first schedule rule for reductions."""

from __future__ import annotations


from tilelang import tvm

from .. import Schedule as TileSchedule
from . import utils
from .base import GPUScheduleRule

ir = tvm.ir
tir = tvm.tir
Target = tvm.target.Target
normalize_prim_func = tvm.dlight.normalize_prim_func
try_inline_contiguous_spatial = tvm.dlight.try_inline_contiguous_spatial


_MIN_ELEMS_PER_THREAD = 4


def _as_const_int(expr: tir.PrimExpr) -> int | None:
    if isinstance(expr, tir.IntImm):
        return int(expr.value)
    return None


def _as_const_float(expr: tir.PrimExpr) -> float | None:
    if isinstance(expr, tir.FloatImm):
        return float(expr.value)
    if isinstance(expr, tir.IntImm):
        return float(expr.value)
    return None


def _is_accumulator_term(store: tir.BufferStore, expr: tir.PrimExpr) -> bool:
    return ir.structural_equal(
        expr,
        tir.BufferLoad(store.buffer, store.indices),
        map_free_vars=True,
    )


def _analyze_reduction_update(block: tir.Block) -> tuple[str, tir.PrimExpr] | None:
    """Infer reduction type and source expression from block update."""
    buffer_store = block.body
    if not isinstance(buffer_store, tir.BufferStore):
        return None

    value = buffer_store.value
    if isinstance(value, tir.Add):
        if _is_accumulator_term(buffer_store, value.a):
            return "sum", value.b
        if _is_accumulator_term(buffer_store, value.b):
            return "sum", value.a
        return None
    if isinstance(value, tir.Max):
        if _is_accumulator_term(buffer_store, value.a):
            return "max", value.b
        if _is_accumulator_term(buffer_store, value.b):
            return "max", value.a
        return None
    if isinstance(value, tir.Min):
        if _is_accumulator_term(buffer_store, value.a):
            return "min", value.b
        if _is_accumulator_term(buffer_store, value.b):
            return "min", value.a
        return None

    value_type = type(value).__name__
    if value_type in ("BitwiseAnd", "BitwiseOr", "BitwiseXor"):
        lhs = value.a
        rhs = value.b
        if _is_accumulator_term(buffer_store, lhs):
            src_expr = rhs
        elif _is_accumulator_term(buffer_store, rhs):
            src_expr = lhs
        else:
            return None
        reduce_type = {
            "BitwiseAnd": "bitand",
            "BitwiseOr": "bitor",
            "BitwiseXor": "bitxor",
        }[value_type]
        return reduce_type, src_expr

    return None


def _extract_single_input_buffer(rhs: tir.PrimExpr, write_buffer: tir.Buffer) -> tir.Buffer | None:
    buffers: list[tir.Buffer] = []

    def _collect(expr):
        if (
            isinstance(expr, tir.BufferLoad)
            and (not expr.buffer.same_as(write_buffer))
            and not any(expr.buffer.same_as(buf) for buf in buffers)
        ):
            buffers.append(expr.buffer)

    tir.stmt_functor.post_order_visit(rhs, _collect)
    if len(buffers) != 1:
        return None
    return buffers[0]


def _is_direct_buffer_load(expr: tir.PrimExpr, target_buffer: tir.Buffer) -> bool:
    """Check whether expr is exactly a direct load from target_buffer."""
    return isinstance(expr, tir.BufferLoad) and expr.buffer.same_as(target_buffer)


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


def _find_buffer_index(regions, target_buffer: tir.Buffer) -> int | None:
    for idx, region in enumerate(regions):
        if region.buffer.same_as(target_buffer):
            return idx
    return None


def _infer_reduce_dim(read_buffer: tir.Buffer, write_buffer: tir.Buffer) -> int:
    src_shape = list(read_buffer.shape)
    dst_shape = list(write_buffer.shape)

    def _shape_equal(lhs, rhs):
        if len(lhs) != len(rhs):
            return False
        return all(tir.analysis.expr_deep_equal(a, b) for a, b in zip(lhs, rhs))

    # Case 1: dst rank is src rank - 1
    if len(dst_shape) == len(src_shape) - 1:
        for dim in range(len(src_shape)):
            if _shape_equal(src_shape[:dim] + src_shape[dim + 1 :], dst_shape):
                return dim

    # Case 2: dst rank equals src rank and reduced axis is kept with extent 1
    if len(dst_shape) == len(src_shape):
        for dim in range(len(src_shape)):
            if not _shape_equal(src_shape[:dim], dst_shape[:dim]):
                continue
            if not _shape_equal(src_shape[dim + 1 :], dst_shape[dim + 1 :]):
                continue
            if _as_const_int(dst_shape[dim]) == 1:
                return dim

    return 0


def _default_init_value(reduce_type: str) -> float:
    if reduce_type in ("sum", "abssum", "bitor", "bitxor"):
        return 0.0
    if reduce_type in ("max", "absmax"):
        return float("-inf")
    if reduce_type == "min":
        return float("inf")
    if reduce_type == "bitand":
        return -1.0
    return 0.0


def _infer_init_value(block: tir.Block, reduce_type: str) -> float:
    if block.init is not None and isinstance(block.init, tir.BufferStore):
        value = _as_const_float(block.init.value)
        if value is not None:
            return value
    return _default_init_value(reduce_type)


def _choose_num_threads(target: Target, reduction_extent: tir.PrimExpr) -> int:
    """Pick cooperative thread count for one-block-per-output reduction.

    Ensures each thread handles at least ``_MIN_ELEMS_PER_THREAD`` elements
    so that local accumulation amortises the cost of the cross-thread tree
    reduction (warp shuffles + shared-memory sync).
    """

    max_threads = int(utils.max_threads_per_block(target))
    max_threads = min(max_threads, 256)
    if max_threads <= 0:
        return 1

    extent = _as_const_int(reduction_extent)
    if extent is None or extent <= 0:
        return max_threads

    # Cap so that each thread has at least 4 elements worth of work.
    effective_max = min(max_threads, max(1, extent // _MIN_ELEMS_PER_THREAD))
    threads = 1
    while (threads << 1) <= effective_max and (threads << 1) <= extent:
        threads <<= 1
    return max(threads, 1)


def _choose_reduction_step(target: Target, reduction_extent: tir.PrimExpr) -> int | None:
    """Choose reduction chunk size for very large K.

    Returns None when extent is dynamic, so the template falls back to
    non-chunked reduction.
    """

    extent = _as_const_int(reduction_extent)
    if extent is None:
        return None
    if extent <= 0:
        return 1

    max_threads = int(utils.max_threads_per_block(target))
    max_threads = min(max_threads, 256)
    # TileLang favors wider tile chunks than thread-level templates.
    # Keep chunk size capped to avoid oversized local fragments while
    # allowing softmax-like K=16384 to stay within a single chunk.
    step_limit = max(1, min(max_threads * 64, 16384))
    step = min(extent, step_limit)

    # Keep static divisibility to avoid tail-copy OOB in cache_read_at.
    while step > 1 and extent % step != 0:
        step >>= 1
    return max(step, 1)


class Reduction(GPUScheduleRule):
    """A reduction schedule rule based on TileLang schedule primitives."""

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
        block_infos = try_inline_contiguous_spatial(sch, block_infos)
        if block_infos is None or len(block_infos) != 1:
            # Start with the single reduction-block pattern. More complex
            # epilogues/prologues can be added incrementally.
            return None

        block_info = block_infos[0]
        block = block_info.block_rv
        block_stmt = sch.get(block)

        if not block_info.is_reduction() or len(block_stmt.writes) != 1 or len(block_stmt.reads) < 1:
            return None

        update_info = _analyze_reduction_update(block_stmt)
        if update_info is None:
            return None
        reduce_type, rhs_expr = update_info
        input_buffer = _extract_single_input_buffer(rhs_expr, block_stmt.body.buffer)

        if input_buffer is None:
            return None

        write_buffer = block_stmt.writes[0].buffer
        read_buffer_index = _find_buffer_index(block_stmt.reads, input_buffer)
        write_buffer_index = _find_buffer_index(block_stmt.writes, write_buffer)
        if read_buffer_index is None or write_buffer_index is None:
            return None

        init_value = _infer_init_value(block_stmt, reduce_type)
        block_name = block_stmt.name_hint

        s_loops: list[tir.schedule.LoopRV] = []
        r_loops: list[tir.schedule.LoopRV] = []
        for iter_info in block_info.iters:
            if iter_info.kind == "S":
                s_loops.append(iter_info.loop_rv)
            elif iter_info.kind == "R":
                r_loops.append(iter_info.loop_rv)
            else:
                return None
        if not s_loops or not r_loops:
            return None

        sch.reorder(*s_loops, *r_loops)
        s_fused = sch.fuse(*s_loops) if len(s_loops) > 1 else s_loops[0]
        r_fused = sch.fuse(*r_loops) if len(r_loops) > 1 else r_loops[0]

        # One output element per CTA (examples/reduction.py style).
        bx, inner_s = sch.split(s_fused, factors=[None, 1], preserve_unit_iters=True)
        reduce_step = _choose_reduction_step(target, sch.get(r_fused).extent)
        if reduce_step is not None:
            ro, ri = sch.split(r_fused, factors=[None, reduce_step], preserve_unit_iters=True)
            sch.reorder(bx, inner_s, ro, ri)
            cache_read_loop = ro
            reduce_loop = ro
            thread_extent_expr = sch.get(ri).extent
        else:
            sch.reorder(bx, inner_s, r_fused)
            cache_read_loop = bx
            reduce_loop = r_fused
            thread_extent_expr = sch.get(r_fused).extent

        # Stage input tile per reduction chunk.
        sch.cache_read_at(cache_read_loop, block, read_buffer_index, "local.fragment")

        # Cache output tile, initialize once per CTA, and write back after reduce.
        block = sch.get_block(block_name)
        sch.cache_reduce_at(bx, block, write_buffer_index, "local.fragment", init_value)

        block = sch.get_block(block_name)
        block_stmt = sch.get(block)
        reduce_dim = _infer_reduce_dim(
            block_stmt.reads[read_buffer_index].buffer,
            block_stmt.writes[write_buffer_index].buffer,
        )

        # Replace the explicit inner reduction loop by tile-level T.reduce
        # when the update can be lowered as a single-source reduction.
        square_single_source = reduce_type == "sum" and _is_square_of_buffer_load(rhs_expr, input_buffer)
        can_lower_to_tile_reduce = _is_direct_buffer_load(rhs_expr, input_buffer) or square_single_source
        if can_lower_to_tile_reduce:
            reduce_type_for_lower = "sumsq" if square_single_source else reduce_type
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

        # Explicit-update reductions do not introduce a thread-bound loop in
        # this template, so keep them single-threaded to avoid duplicating the
        # full reduction in every CTA lane.
        num_threads = _choose_num_threads(target, thread_extent_expr) if can_lower_to_tile_reduce else 1
        sch.bind(bx, "blockIdx.x")
        sch.launch_thread(sch.get_block("root"), num_threads)
        return sch
