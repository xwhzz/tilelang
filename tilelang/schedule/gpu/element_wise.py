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
"""A tile-primitive-first schedule rule for element-wise operators."""

from __future__ import annotations

from typing import Any

from tilelang import tvm

from .. import Schedule as TileSchedule
from . import utils
from .base import GPUScheduleRule

tir = tvm.tir
Target = tvm.target.Target
normalize_prim_func = tvm.dlight.normalize_prim_func
try_inline = tvm.dlight.try_inline


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
    tile_cap = max(max_threads, min(max_threads * 8, 8192))

    const_extent = _as_const_int(extent)
    if const_extent is None:
        return tile_cap, max_threads
    if const_extent <= 0:
        return 1, 1

    # Choose the largest power-of-2 thread count that does not exceed the
    # extent, then round the tile up to a multiple of that thread count.
    threads = _largest_pow2_at_most(min(const_extent, max_threads))
    # Round extent up to the nearest multiple of threads.
    tile = (const_extent + threads - 1) // threads * threads
    tile = min(tile, tile_cap)
    # Ensure tile is still a multiple of threads after capping.
    tile = tile // threads * threads
    return max(tile, threads), max(threads, 1)


def _tile_aligns_with_suffix(s_extents: list[int | None], tile: int) -> bool:
    """Check that tile divides the product of some *proper* suffix of s_extents.

    When fusing dims (d0, d1, ..., dn) and splitting by *tile*, the tile
    boundaries align with the original dimension boundaries only if *tile*
    divides d_{k} * d_{k+1} * ... * d_{n} for some k > 0 (i.e. not all
    dims).  If only the full product (k=0) is divisible, the tile still
    crosses intermediate dimension boundaries and cache_read_at creates a
    fragment with non-affine access patterns.

    For a single spatial dim, divisibility of the extent itself suffices.

    When dynamic extents are present, we check the *static* innermost
    suffix.  If the product of all contiguous static dims starting from
    the innermost already divides by tile, alignment is guaranteed
    regardless of the dynamic dims.  Otherwise we conservatively return
    False — the tile may cross dimension boundaries and produce
    block-dependent fragment shapes that crash LayoutInference.
    """
    if not s_extents:
        return True
    has_dynamic = any(e is None for e in s_extents)
    if not has_dynamic:
        if len(s_extents) <= 1:
            return s_extents[0] % tile == 0
        suffix_product = 1
        for e in reversed(s_extents[1:]):  # skip outermost dim
            suffix_product *= e
            if suffix_product % tile == 0:
                return True
        return False
    # Dynamic case: check the contiguous static innermost suffix.
    # If those known dims already align with the tile, we're safe.
    suffix_product = 1
    for e in reversed(s_extents):
        if e is None:
            break
        suffix_product *= e
        if suffix_product % tile == 0:
            return True
    return False


def _resolve_target_from_config(config: Any) -> Target:
    if config is not None:
        arch = getattr(config, "arch", None)
        if arch is not None and hasattr(arch, "target"):
            if isinstance(arch.target, Target):
                return arch.target
            return Target(arch.target)
        cfg_target = getattr(config, "target", None)
        if cfg_target is not None:
            if isinstance(cfg_target, Target):
                return cfg_target
            return Target(cfg_target)
    return Target("cuda")


class ElementWise(GPUScheduleRule):
    """A tile schedule rule for injective element-wise kernels."""

    def apply(  # pylint: disable=too-many-return-statements
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

        # Keep one output block by inlining leading injective blocks.
        for block_info in block_infos[:-1]:
            if block_info.is_reduction() or not block_info.is_injective():
                return None
            try:
                sch.compute_inline(block_info.block_rv)
            except tir.ScheduleError:
                # Output blocks cannot be inlined. Let the generic fallback
                # handle multi-output injective graphs instead.
                return None

        block_infos = normalize_prim_func(sch)
        if block_infos is None or len(block_infos) != 1:
            return None

        block_info = block_infos[0]
        if block_info.is_reduction():
            return None

        block = block_info.block_rv
        block_stmt = sch.get(block)
        if len(sch.get_loops(block)) == 0:
            return None

        s_loops: list[tir.schedule.LoopRV] = []
        o_loops: list[tir.schedule.LoopRV] = []
        for iter_info in block_info.iters:
            if iter_info.kind == "S":
                s_loops.append(iter_info.loop_rv)
            elif iter_info.kind == "O":
                o_loops.append(iter_info.loop_rv)
            else:
                return None

        if not s_loops:
            s_loops.append(sch.add_unit_loop(block))

        # Capture original spatial extents before fusing (needed for
        # suffix-product divisibility check below).
        s_extents = [_as_const_int(sch.get(l).extent) for l in s_loops]

        sch.reorder(*s_loops, *o_loops)
        s_fused = sch.fuse(*s_loops) if len(s_loops) > 1 else s_loops[0]
        fused_extent = sch.get(s_fused).extent
        tile_extent, num_threads = _choose_tile_and_threads(target, fused_extent)
        bx, inner = sch.split(s_fused, factors=[None, tile_extent], preserve_unit_iters=True)
        if o_loops:
            sch.reorder(bx, inner, *o_loops)

        # Fragment caching: cache_read_at produces a rectangular fragment
        # whose shape matches the original buffer dimensions.  This works
        # only when the tile aligns with some suffix of the fused dims —
        # i.e. tile_extent divides the product of the last k spatial extents
        # for some k.  When it does not, the tile crosses intermediate
        # dimension boundaries and cache_read_at creates a fragment with
        # non-affine access patterns that LayoutInference cannot handle.
        #
        # TODO: teach cache_read_at to produce 1-D (flat) fragments for
        # fused loops so that fragments work for every tile size.
        use_fragment = _tile_aligns_with_suffix(s_extents, tile_extent)
        if use_fragment:
            block_name = block_stmt.name_hint
            write_buffers = [region.buffer for region in block_stmt.writes]
            for read_buffer_index, region in enumerate(block_stmt.reads):
                if any(region.buffer.same_as(write_buffer) for write_buffer in write_buffers):
                    continue
                # Skip 0-dim (scalar) buffers — cache_read_at requires ≥1-D.
                if len(region.buffer.shape) == 0:
                    continue
                block = sch.get_block(block_name)
                sch.cache_read_at(bx, block, read_buffer_index, "local.fragment")

            block = sch.get_block(block_name)
            for write_buffer_index, _ in enumerate(sch.get(block).writes):
                sch.cache_write_at(bx, block, write_buffer_index, "local.fragment")
                block = sch.get_block(block_name)

        sch.parallel(inner)
        sch.bind(bx, "blockIdx.x")
        sch.launch_thread(sch.get_block("root"), num_threads)
        return sch

    def apply_config(self, func: tir.PrimFunc, config) -> None | tir.Schedule | list[tir.Schedule]:
        target = _resolve_target_from_config(config)
        return self.apply(func, target, False)
