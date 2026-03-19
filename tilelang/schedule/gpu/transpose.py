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
"""Tile-based transpose schedule rule."""

from __future__ import annotations

from tilelang import tvm
from tilelang import _ffi_api as tl_ffi

from .. import Schedule as TileSchedule
from . import utils
from .base import GPUScheduleRule

tir = tvm.tir
Target = tvm.target.Target
normalize_prim_func = tvm.dlight.normalize_prim_func


def _largest_power_of_two_at_most(value: int) -> int:
    result = 1
    while (result << 1) <= value:
        result <<= 1
    return result


class Transpose(GPUScheduleRule):
    """Tile-based transpose schedule using shared-memory staging."""

    @staticmethod
    def _is_transpose(sch, block_rv) -> bool:
        block = sch.get(block_rv)
        if not isinstance(block.body, tir.BufferStore):
            return False
        rhs = block.body.value
        if not isinstance(rhs, tir.BufferLoad):
            return False
        lhs_indices = block.body.indices
        rhs_indices = rhs.indices
        return list(lhs_indices) != list(rhs_indices) and set(lhs_indices) == set(rhs_indices)

    def is_transpose(self, sch, block_rv):
        return Transpose._is_transpose(sch, block_rv)

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

        # Find the last transpose block (scan from end).
        transpose_idx = -1
        for idx, info in reversed(list(enumerate(block_infos))):
            if self._is_transpose(sch, info.block_rv):
                transpose_idx = idx
                break
            if not info.is_injective():
                return None
        if transpose_idx == -1:
            return None

        transpose_block = block_infos[transpose_idx].block_rv
        loops = sch.get_loops(transpose_block)
        if len(loops) != 2:
            return None

        # Inline injective producers before the transpose block.
        for info in block_infos[:transpose_idx]:
            sch.compute_inline(info.block_rv)

        # Inline injective consumers after the transpose block (reversed).
        for info in reversed(block_infos[transpose_idx + 1:]):
            sch.compute_inline(info.block_rv)

        tile_size = 128
        max_threads = min(int(utils.max_threads_per_block(target)), 512)
        num_threads = _largest_power_of_two_at_most(max_threads)

        # Get the read buffer name to construct shared cache buffer name.
        block_stmt = sch.get(transpose_block)
        read_buf = block_stmt.reads[0].buffer
        element_bits = int(tvm.DataType(read_buf.dtype).bits)
        shared_buf_name = read_buf.name + "_shared_dyn"

        i, j = loops
        bi, ii = sch.split(i, factors=[None, tile_size], preserve_unit_iters=True)
        bj, jj = sch.split(j, factors=[None, tile_size], preserve_unit_iters=True)
        sch.reorder(bi, bj, ii, jj)

        # Stage input through shared memory for coalesced global reads;
        # the transposed read from shared avoids uncoalesced global access.
        # SIMT copy with swizzled layout outperforms TMA for transpose.
        sch.cache_read_at(bj, transpose_block, 0, "shared.dyn", disable_tma=True)
        # Stage output through registers.
        sch.cache_write_at(bj, transpose_block, 0, "local.fragment")

        # Annotate the shared buffer with a swizzled layout to avoid
        # bank conflicts on the transposed read.
        swizzle_layout = tl_ffi.make_swizzled_layout(
            tile_size, tile_size, element_bits, True, True,
        )
        sch.annotate_layout(sch.get_block("root"), shared_buf_name, swizzle_layout)

        sch.parallel(ii)
        sch.parallel(jj)
        sch.bind(bi, "blockIdx.x")
        sch.bind(bj, "blockIdx.y")

        sch.launch_thread(sch.get_block("root"), num_threads)
        return sch
