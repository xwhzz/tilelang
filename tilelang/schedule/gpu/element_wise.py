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
"""A tile-primitive-first schedule rule for element-wise operators.

Strategy
--------
1.  Flatten all spatial dims and buffers to 1D via ``transform_layout``
    + ``transform_block_layout``.
2.  Split into ``[blockIdx.x, inner=TILE]``.
3.  ``cache_read_at`` / ``cache_write_at`` at the block level so every
    global memory access goes through a ``local.fragment`` with constant
    shape ``(TILE,)``.  This gives vectorised ``float4`` loads/stores
    **and** register reuse for expressions that read the same input
    multiple times (e.g. GELU: ``x * (0.5 + erf(x * 0.707))``).
4.  ``parallel(inner)`` + ``bind(bx, blockIdx.x)`` +
    ``launch_thread(root, NUM_THREADS)``.

The flattened kernel expects 1D buffers; the wrapper produced by
``codegen.py`` calls ``.view(-1)`` on every tensor before dispatch.
"""

from __future__ import annotations

from tilelang import tvm

from .. import Schedule as TileSchedule
from . import utils
from .base import GPUScheduleRule

from tvm import tir
from tvm.target import Target
from tvm.dlight import normalize_prim_func, try_inline


def _as_const_int(expr: tir.PrimExpr) -> int | None:
    if isinstance(expr, tir.IntImm):
        return int(expr.value)
    return None


def _largest_pow2_at_most(n: int) -> int:
    """Return the largest power of 2 that is <= *n*."""
    p = 1
    while (p << 1) <= n:
        p <<= 1
    return p


def _choose_tile_and_threads(
    target: Target,
    extent: tir.PrimExpr,
    is_static: bool = True,
) -> tuple[int, int]:
    """Jointly select (tile_extent, num_threads) for element-wise scheduling.

    Constraints:
    - tile_extent must be a multiple of num_threads (for layout inference).
    - num_threads must be a power of 2.
    - Maximize num_threads for GPU occupancy.

    tile_extent does NOT need to divide the fused extent; the downstream
    ``LegalizeSafeMemoryAccess`` pass inserts OOB guards for the last
    partial tile automatically.
    """
    max_threads = min(int(utils.max_threads_per_block(target)), 1024)
    if not is_static:
        return max_threads * 8, max_threads

    const_extent = _as_const_int(extent)

    if const_extent <= 0:
        return 1, 1

    # Choose thread count and tile to balance SM occupancy and per-thread
    # work.  256 threads with ~16 elems/thread is empirically good for
    # large memory-bound kernels on H100/A100.  For smaller extents,
    # reduce to avoid too few blocks.
    threads = min(256, _largest_pow2_at_most(const_extent))
    threads = max(threads, 32)

    # Target enough blocks for full GPU occupancy (~1k-2k on H100)
    # but cap elements-per-thread at 16.
    elems_per_thread = min(16, max((const_extent // threads + 1023) // 1024, 4))
    tile = threads * elems_per_thread

    # Final adjustment: ensure tile doesn't exceed extent
    if tile > const_extent:
        tile = (const_extent + threads - 1) // threads * threads

    return max(tile, threads), threads


class ElementWise(GPUScheduleRule):
    """Tile schedule rule for injective element-wise kernels.

    Produces a flat 1D kernel with vectorised fragment caching for both
    reads and writes.  Falls back to a non-fragment 1D split when the
    extents are symbolic (dynamic shapes).
    """

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
        block_infos = try_inline(sch, block_infos)
        if block_infos is None or len(block_infos) == 0:
            return None

        # Inline everything except the final output block.
        for bi in block_infos[:-1]:
            if bi.is_reduction() or not bi.is_injective():
                return None
            try:
                sch.compute_inline(bi.block_rv)
            except tir.ScheduleError:
                return None

        block_infos = normalize_prim_func(sch)
        if block_infos is None or len(block_infos) != 1:
            return None

        block_info = block_infos[0]
        if block_info.is_reduction():
            return None

        block = block_info.block_rv
        if len(sch.get_loops(block)) == 0:
            return None

        s_loops = [it.loop_rv for it in block_info.iters if it.kind == "S"]
        o_loops = [it.loop_rv for it in block_info.iters if it.kind == "O"]
        if any(it.kind not in ("S", "O") for it in block_info.iters):
            return None
        if not s_loops:
            s_loops.append(sch.add_unit_loop(block))

        sch.reorder(*s_loops, *o_loops)

        # ── Check whether all spatial extents are static ──────────
        s_extents = [_as_const_int(sch.get(l).extent) for l in s_loops]
        all_static = all(e is not None for e in s_extents)

        if all_static and len(s_loops) >= 2:
            result = self._try_flatten_fragment(func, s_extents, target)
            if result is not None:
                return result
        # ── Fallback: 1D fused split, no fragment ────────────
        s_fused = sch.fuse(*s_loops) if len(s_loops) > 1 else s_loops[0]
        TILE, NUM_THREADS = _choose_tile_and_threads(
            target, sch.get(s_fused).extent, is_static=all_static)
        bx, inner = sch.split(s_fused, factors=[None, TILE],
                              preserve_unit_iters=True)
        if o_loops:
            sch.reorder(bx, inner, *o_loops)

        sch.parallel(inner)
        sch.bind(bx, "blockIdx.x")
        sch.launch_thread(sch.get_block("root"), NUM_THREADS)
        return sch

    def _try_flatten_fragment(self, func, s_extents, target):
        """Attempt the flatten+fragment schedule on a fresh TileSchedule.

        Returns a scheduled TileSchedule on success, None on failure.
        """
        try:
            sch = TileSchedule(func)
            block_infos = normalize_prim_func(sch)
            if block_infos is None:
                return None
            block_infos = try_inline(sch, block_infos)
            if not block_infos:
                return None
            for bi in block_infos[:-1]:
                sch.compute_inline(bi.block_rv)
            block_infos = normalize_prim_func(sch)
            if not block_infos or len(block_infos) != 1:
                return None
            block = block_infos[0].block_rv

            s_loops = [it.loop_rv for it in block_infos[0].iters if it.kind == "S"]
            o_loops = [it.loop_rv for it in block_infos[0].iters if it.kind == "O"]
            if not s_loops:
                return None
            sch.reorder(*s_loops, *o_loops)

            strides = []
            stride = 1
            for e in reversed(s_extents):
                strides.insert(0, stride)
                stride *= e
            ndim = len(s_extents)
            flatten_map = tir.IndexMap.from_func(
                lambda *indices: [sum(idx * s for idx, s in zip(indices, strides))],
                ndim=ndim)
            
            TILE, NUM_THREADS = _choose_tile_and_threads(target, stride)

            block_stmt = sch.get(block)
            for i in range(len(block_stmt.reads)):
                sch.transform_layout(block, ("read", i), flatten_map)
            for i in range(len(block_stmt.writes)):
                sch.transform_layout(block, ("write", i), flatten_map)
            sch.transform_block_layout(block, flatten_map)

            loop = sch.get_loops(block)[0]
            bx, inner = sch.split(loop, factors=[None, TILE],
                                  preserve_unit_iters=True)
            if o_loops:
                sch.reorder(bx, inner, *o_loops)

            block_stmt = sch.get(block)
            for i in range(len(block_stmt.reads)):
                sch.cache_read_at(bx, block, i, "local.fragment")
            sch.cache_write_at(bx, block, 0, "local.fragment")

            sch.parallel(inner)
            sch.bind(bx, "blockIdx.x")
            sch.launch_thread(sch.get_block("root"), NUM_THREADS)
            return sch
        except Exception:
            return None
