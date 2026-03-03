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

from typing import Any, List, Optional, Union

from tvm import tir
from tvm.target import Target

from .. import Schedule as TileSchedule
from ..base import normalize_prim_func, try_inline
from . import utils
from .base import GPUScheduleRule


def _as_const_int(expr: tir.PrimExpr) -> Optional[int]:
    if isinstance(expr, tir.IntImm):
        return int(expr.value)
    return None


def _choose_num_threads(target: Target, extent: tir.PrimExpr) -> int:
    max_threads = int(utils.max_threads_per_block(target))
    max_threads = min(max_threads, 1024)
    if max_threads <= 0:
        return 1

    const_extent = _as_const_int(extent)
    if const_extent is None or const_extent <= 0:
        return max_threads

    threads = 1
    while (threads << 1) <= max_threads and (threads << 1) <= const_extent:
        threads <<= 1
    return threads


def _choose_tile_extent(target: Target, extent: tir.PrimExpr, num_threads: int) -> int:
    const_extent = _as_const_int(extent)
    tile_cap = max(num_threads, min(int(utils.max_threads_per_block(target)) * 8, 8192))
    if const_extent is None:
        return tile_cap
    if const_extent <= 0:
        return 1

    tile = min(const_extent, tile_cap)
    while tile > 1 and const_extent % tile != 0:
        tile >>= 1
    return max(tile, 1)


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
    ) -> Union[None, tir.Schedule, List[tir.Schedule]]:
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
            sch.compute_inline(block_info.block_rv)

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

        s_loops: List[tir.schedule.LoopRV] = []
        o_loops: List[tir.schedule.LoopRV] = []
        for iter_info in block_info.iters:
            if iter_info.kind == "S":
                s_loops.append(iter_info.loop_rv)
            elif iter_info.kind == "O":
                o_loops.append(iter_info.loop_rv)
            else:
                return None

        if not s_loops:
            s_loops.append(sch.add_unit_loop(block))

        sch.reorder(*s_loops, *o_loops)
        s_fused = sch.fuse(*s_loops) if len(s_loops) > 1 else s_loops[0]
        num_threads = _choose_num_threads(target, sch.get(s_fused).extent)
        tile_extent = _choose_tile_extent(target, sch.get(s_fused).extent, num_threads)
        bx, inner = sch.split(s_fused, factors=[None, tile_extent], preserve_unit_iters=True)
        if o_loops:
            sch.reorder(bx, inner, *o_loops)

        block_name = block_stmt.name_hint
        write_buffers = [region.buffer for region in block_stmt.writes]
        for read_buffer_index, region in enumerate(block_stmt.reads):
            if any(region.buffer.same_as(write_buffer) for write_buffer in write_buffers):
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

    def apply_config(self, func: tir.PrimFunc, config) -> Union[None, tir.Schedule, List[tir.Schedule]]:
        target = _resolve_target_from_config(config)
        return self.apply(func, target, False)
