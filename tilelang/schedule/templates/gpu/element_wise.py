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
"""Tile-primitive schedule rule for element-wise (injective) operators.

Strategy (static shapes)
------------------------
1.  Inline all blocks except the final output block.
2.  Flatten all spatial dims to 1-D via ``transform_layout`` +
    ``transform_block_layout``.
3.  Split the flat loop into ``[blockIdx.x, inner=TILE]``.
4.  ``cache_read_at`` / ``cache_write_at`` at the block-index level
    into ``local.fragment`` — produces vectorised ``uint4``
    loads/stores and register reuse.
5.  ``parallel(inner)`` + ``bind(bx)`` + ``launch_thread``.
6.  Restore the original N-D buffer shapes in the ``buffer_map`` so
    that MakePackedAPI's ndim check passes when the graph-pipeline VM
    forwards N-D tensors.

Fallback (dynamic shapes)
-------------------------
Simple 1-D fuse + split without fragment caching.
"""

from __future__ import annotations

from tilelang import tvm

from .. import Schedule as TileSchedule
from . import utils
from .base import GPUScheduleRule

from tvm import tir
from tvm.target import Target
from tvm.dlight import normalize_prim_func, try_inline


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _as_const_int(expr) -> int | None:
    if isinstance(expr, int):
        return expr
    if isinstance(expr, tir.IntImm):
        return int(expr.value)
    return None


def _largest_pow2_at_most(n: int) -> int:
    p = 1
    while (p << 1) <= n:
        p <<= 1
    return p


def _choose_tile_and_threads(total: int) -> tuple[int, int]:
    """Pick (tile, num_threads) for a 1-D elementwise kernel."""
    if total <= 0:
        return 1, 1

    threads = min(256, _largest_pow2_at_most(total))
    threads = max(threads, 32)

    elems_per_thread = min(16, max((total // threads + 1023) // 1024, 4))
    tile = threads * elems_per_thread

    if tile > total:
        tile = (total + threads - 1) // threads * threads

    return max(tile, threads), threads


def _build_flatten_map(s_extents: list[int]) -> tir.IndexMap:
    """Build an int32 IndexMap that flattens N-D → 1-D.

    We construct the map manually with int32 types and an explicit
    inverse because ``IndexMap.from_func`` uses int64 variables which
    clash with int32 iter vars from ``te.create_prim_func``.
    """
    ndim = len(s_extents)

    # Forward: (i0, i1, …) → i0 * s1 * s2 * … + i1 * s2 * … + …
    fwd_vars = [tir.Var(f"i{k}", "int32") for k in range(ndim)]
    flat_expr = fwd_vars[0]
    for k in range(1, ndim):
        flat_expr = flat_expr * tir.const(s_extents[k], "int32") + fwd_vars[k]

    # Inverse: flat → (flat // stride0, flat % stride0 // stride1, …)
    strides = []
    stride = 1
    for e in reversed(s_extents):
        strides.insert(0, stride)
        stride *= e

    inv_var = tir.Var("flat", "int32")
    inv_exprs = []
    rem = inv_var
    for k in range(ndim - 1):
        inv_exprs.append(tir.floordiv(rem, tir.const(strides[k], "int32")))
        rem = tir.floormod(rem, tir.const(strides[k], "int32"))
    inv_exprs.append(rem)

    inv_map = tir.IndexMap([inv_var], inv_exprs, None)
    return tir.IndexMap(fwd_vars, [flat_expr], inv_map)


def _restore_buffer_shapes(
    func: tir.PrimFunc,
    original_func: tir.PrimFunc,
) -> tir.PrimFunc:
    """Swap the buffer_map back to the original N-D shapes.

    After flattening, external buffers are 1-D, but the graph pipeline
    passes N-D tensors.  MakePackedAPI checks ``ndim`` from the
    buffer_map, so we restore the original shapes there.  Internal 1-D
    accesses are unaffected — they share the same ``data`` pointer.
    """
    orig_bufs = {
        original_func.buffer_map[p].data.name: original_func.buffer_map[p]
        for p in original_func.params
    }

    new_params = []
    new_buffer_map = {}
    for param in func.params:
        buf = func.buffer_map[param]
        orig = orig_bufs.get(buf.data.name)
        shape = orig.shape if orig is not None else buf.shape
        name = orig.name if orig is not None else buf.name
        new_param = tir.Var(name + "_handle", "handle")
        new_buf = tir.decl_buffer(shape, buf.dtype, name=name, data=buf.data)
        new_params.append(new_param)
        new_buffer_map[new_param] = new_buf

    return tir.PrimFunc(
        new_params, func.body,
        func.ret_type, new_buffer_map, func.attrs,
    )


def _inline_to_single_block(sch):
    """Inline all blocks except the last output block.

    Returns (block_rv, block_info) on success, None on failure.
    """
    block_infos = normalize_prim_func(sch)
    if block_infos is None:
        return None
    block_infos = try_inline(sch, block_infos)
    if not block_infos:
        return None

    for bi in block_infos[:-1]:
        if bi.is_reduction() or not bi.is_injective():
            return None
        try:
            sch.compute_inline(bi.block_rv)
        except tir.ScheduleError:
            return None

    block_infos = normalize_prim_func(sch)
    if not block_infos or len(block_infos) != 1:
        return None

    info = block_infos[0]
    if info.is_reduction():
        return None
    if len(sch.get_loops(info.block_rv)) == 0:
        return None
    if any(it.kind not in ("S", "O") for it in info.iters):
        return None

    s_iters = [it for it in info.iters if it.kind == "S"]
    if not s_iters:
        return None

    return info


# ---------------------------------------------------------------------------
# Schedule rule
# ---------------------------------------------------------------------------

class _ScheduleResult:
    """Thin wrapper with a ``.mod`` attribute, returned when we need to
    post-process the PrimFunc (buffer shape restoration) outside the
    normal TileSchedule flow."""

    def __init__(self, mod: tvm.IRModule):
        self.mod = mod


class ElementWise(GPUScheduleRule):
    """Tile schedule rule for injective element-wise kernels."""

    def apply(
        self,
        func: tir.PrimFunc,
        target: Target,
        _: bool,
    ) -> None | tir.Schedule | list[tir.Schedule]:
        if not isinstance(func, tir.PrimFunc) or not self.is_target_available(target):
            return None

        # Probe the function to check applicability and get extents.
        sch = TileSchedule(func)
        info = _inline_to_single_block(sch)
        if info is None:
            return None

        s_extents = [_as_const_int(sch.get(it.loop_rv).extent)
                     for it in info.iters if it.kind == "S"]
        all_static = all(e is not None for e in s_extents)

        # Primary path: flatten + fragment caching.
        # Skip for tiny tensors — fragment overhead dominates and
        # LayoutInference may fail to find a valid layout.
        total = 1
        for e in s_extents:
            total *= e
        if all_static and total >= 1024:
            result = self._apply_flatten_fragment(func, s_extents, target)
            if result is not None:
                return [result]

        # Fallback: simple 1-D split, no fragments (dynamic shapes).
        return self._apply_fallback(sch, info, target, all_static)

    # ------------------------------------------------------------------

    def _apply_flatten_fragment(self, func, s_extents, target):
        """Flatten → tile → fragment-cache → restore buffer shapes."""
        try:
            sch = TileSchedule(func)
            info = _inline_to_single_block(sch)
            if info is None:
                return None
            block = info.block_rv
            s_loops = [it.loop_rv for it in info.iters if it.kind == "S"]
            o_loops = [it.loop_rv for it in info.iters if it.kind == "O"]
            sch.reorder(*s_loops, *o_loops)

            total = 1
            for e in s_extents:
                total *= e
            TILE, NUM_THREADS = _choose_tile_and_threads(total)

            # Flatten to 1-D (skip for already-1-D tensors).
            if len(s_extents) > 1:
                flatten_map = _build_flatten_map(s_extents)
                block_stmt = sch.get(block)
                for i in range(len(block_stmt.reads)):
                    sch.transform_layout(block, ("read", i), flatten_map)
                for i in range(len(block_stmt.writes)):
                    sch.transform_layout(block, ("write", i), flatten_map)
                sch.transform_block_layout(block, flatten_map)

            # Tile.
            loop = sch.get_loops(block)[0]
            bx, inner = sch.split(loop, factors=[None, TILE],
                                  preserve_unit_iters=True)
            if o_loops:
                sch.reorder(bx, inner, *o_loops)

            # Fragment caching.
            block_stmt = sch.get(block)
            for i in range(len(block_stmt.reads)):
                sch.cache_read_at(bx, block, i, "local.fragment")
            sch.cache_write_at(bx, block, 0, "local.fragment")

            sch.parallel(inner)
            sch.bind(bx, "blockIdx.x")
            sch.launch_thread(sch.get_block("root"), NUM_THREADS)

            # Restore original buffer shapes for MakePackedAPI.
            new_func = _restore_buffer_shapes(sch.mod["main"], func)
            return _ScheduleResult(tvm.IRModule({"main": new_func}))
        except Exception:
            return None

    # ------------------------------------------------------------------

    @staticmethod
    def _apply_fallback(sch, info, target, is_static):
        """Simple 1-D fuse + split — no flatten, no fragments."""
        block = info.block_rv
        s_loops = [it.loop_rv for it in info.iters if it.kind == "S"]
        o_loops = [it.loop_rv for it in info.iters if it.kind == "O"]
        if not s_loops:
            s_loops.append(sch.add_unit_loop(block))
        sch.reorder(*s_loops, *o_loops)
        s_fused = sch.fuse(*s_loops) if len(s_loops) > 1 else s_loops[0]

        if is_static:
            total = _as_const_int(sch.get(s_fused).extent)
            TILE, NUM_THREADS = _choose_tile_and_threads(total)
        else:
            max_threads = min(int(utils.max_threads_per_block(target)), 1024)
            TILE, NUM_THREADS = max_threads * 8, max_threads

        bx, inner = sch.split(s_fused, factors=[None, TILE],
                              preserve_unit_iters=True)
        if o_loops:
            sch.reorder(bx, inner, *o_loops)
        sch.parallel(inner)
        sch.bind(bx, "blockIdx.x")
        sch.launch_thread(sch.get_block("root"), NUM_THREADS)
        return sch
