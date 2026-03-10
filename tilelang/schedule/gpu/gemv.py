# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""A schedule rule for GEMV using rfactor-based cross-thread reduction.

Adapted from tvm.dlight.gpu.gemv to use standard tir.Schedule with rfactor
for parallel reduction across threads, instead of TileSchedule + cache_reduce_at.
"""

from functools import reduce
from typing import List, Optional, Union

from tvm import tir
from tvm.target import Target

from tvm.dlight import normalize_prim_func, try_inline_contiguous_spatial
from tvm.dlight.analysis import BlockInfo, is_broadcast_epilogue, is_gemv, normalize
from tvm.dlight.base import get_bytes, get_extent
from .base import GPUScheduleRule


class GEMV(GPUScheduleRule):
    """A schedule rule for GEMV using rfactor-based cross-thread reduction.

    Uses standard tir.Schedule with two-level rfactor to split reduction
    across threads, following the proven dlight GEMV pattern.
    """

    def apply(
        self,
        func: tir.PrimFunc,
        target: Target,
        _: bool,
    ) -> Union[None, tir.Schedule, List[tir.Schedule]]:
        if not isinstance(func, tir.PrimFunc) or not self.is_target_available(target):
            return None

        sch = tir.Schedule(func)
        block_infos = normalize_prim_func(sch)
        block_infos = try_inline_contiguous_spatial(sch, block_infos)
        if block_infos is None:
            return None

        if len(block_infos) == 1:
            epilogue = None
        elif len(block_infos) == 2:
            epilogue = block_infos[1]
            if not epilogue.is_injective():
                return None
        else:
            return None

        block_info = block_infos[0]
        if len(block_info.iters) not in [2, 3]:
            return None

        block = block_info.block_rv
        vector_input_buffers = is_gemv(sch, block_info)
        if vector_input_buffers is None:
            return None

        # Normalize block to 4 loops: batch, s, r, c
        is_inner_reduction = normalize(sch, block_info)
        if is_inner_reduction is None:
            return None
        elif is_inner_reduction:
            return self.sch_inner_reduction(sch, target, block, vector_input_buffers, epilogue)
        else:
            # Outer reduction not supported in tilelang yet
            return None

    def sch_inner_reduction(
        self,
        sch: tir.Schedule,
        target: Target,
        block: tir.schedule.BlockRV,
        vector_input_buffers: List[tir.Buffer],
        epilogue_info: Optional[BlockInfo],
    ):
        """Schedule the inner reduction block using rfactor."""

        def get_max_factor(n, factors):
            factors = sorted(factors, reverse=True)
            for factor in factors:
                if n % factor == 0:
                    return factor
            return 1

        def apply(
            sch: tir.Schedule,
            gemv,
            TAG_S,
            TAG_R,
            TS,
            TR,
            TILE_S,
            TILE_R,
            VEC_LOAD,
            VEC_C,
            LOAD_V_SHARED,
            LOAD_V_VEC,
            UNROLL,
            SUPPORT_WARP_SHUFFLE,
        ):
            # rfactor: reduce to tx * vec_c
            _, s, r, c = sch.get_loops(block=gemv)
            s = sch.fuse(_, s)
            r = sch.fuse(r, c)
            bx, ts, tile_s = sch.split(s, factors=[None, TS, TILE_S], preserve_unit_iters=True)
            r, tr, tile_r_vec_n, vec_c = sch.split(
                r, factors=[None, TR, TILE_R // VEC_C, VEC_C], preserve_unit_iters=True
            )
            sch.reorder(r, tile_r_vec_n, tr, vec_c)
            tr_vec_c = sch.fuse(tr, vec_c)
            rf = sch.rfactor(tr_vec_c, 0)

            # rfactor: reduce to tx
            bx, ts, tile_s, tr_vec_c = sch.get_loops(block=gemv)
            tr, vec_c = sch.split(tr_vec_c, factors=[TR, None], preserve_unit_iters=True)
            rf2 = sch.rfactor(tr, 0)

            # bind, vectorize compute
            bx, ts, tile_s, r, tile_r_vec_n, tr_vec_c = sch.get_loops(block=rf)
            tr, vec_c = sch.split(tr_vec_c, factors=[TR, None], preserve_unit_iters=True)
            sch.reorder(bx, ts, tr, r, tile_s, tile_r_vec_n, vec_c)
            sch.bind(bx, "blockIdx.x")
            sch.bind(ts, TAG_S)
            sch.bind(tr, TAG_R)
            sch.vectorize(vec_c)

            shared_mem_usage = 0
            for buf in vector_input_buffers:
                dtype_bytes = get_bytes(buf.dtype)
                buf_size = (
                    reduce(lambda x, y: x * y, buf.shape, tir.IntImm(buf.shape[0].dtype, 1))
                    * dtype_bytes
                )
                shared_mem_usage += buf_size
                if not SUPPORT_WARP_SHUFFLE:
                    shared_mem_usage += TS * TR * dtype_bytes

            try:
                max_shared = target.max_shared_memory_per_block
            except (KeyError, AttributeError):
                max_shared = 49152  # default 48KB
            LOAD_V_SHARED_LOCAL = (
                LOAD_V_SHARED
                and isinstance(shared_mem_usage, tir.IntImm)
                and shared_mem_usage.value <= max_shared
            )

            # Find correct read buffer indices for vector vs matrix in rf block
            rf_stmt = sch.get(rf)
            vector_read_idx = None
            matrix_read_idx = None
            for i, read in enumerate(rf_stmt.reads):
                if any(read.buffer.same_as(vbuf) for vbuf in vector_input_buffers):
                    vector_read_idx = i
                else:
                    matrix_read_idx = i
            # Fallback to dlight defaults if detection fails
            if vector_read_idx is None:
                vector_read_idx = 0
            if matrix_read_idx is None:
                matrix_read_idx = 1

            # vectorize load A (matrix input)
            Aq_local = sch.cache_read(rf, read_buffer_index=matrix_read_idx, storage_scope="local")
            sch.compute_at(Aq_local, r, preserve_unit_loops=True)
            s_local, r_local = sch.get_loops(block=Aq_local)[-2:]
            fused_load = sch.fuse(s_local, r_local)
            aq_vec_len = max(1, VEC_LOAD // get_bytes(sch.get(Aq_local).reads[0].buffer.dtype))
            fused_load, vec_load = sch.split(
                fused_load, factors=[None, aq_vec_len], preserve_unit_iters=True
            )
            sch.vectorize(vec_load)

            # load vector into shared memory
            if LOAD_V_SHARED_LOCAL:
                if len(vector_input_buffers) != 1:
                    return None
                V_shared = sch.cache_read(rf, read_buffer_index=vector_read_idx, storage_scope="shared")
                sch.compute_at(V_shared, tr, preserve_unit_loops=True)
                l = sch.get_loops(block=V_shared)[-1]
                loop: tir.For = sch.get(l)
                if isinstance(loop.extent, tir.IntImm):
                    vec_length = max(
                        min(
                            get_max_factor(
                                (int)(loop.extent),
                                [TS * TR * 1, TS * TR * 2, TS * TR * 4, TS * TR * 8],
                            )
                            // TS
                            // TR,
                            LOAD_V_VEC,
                        ),
                        1,
                    )
                else:
                    vec_length = LOAD_V_VEC
                if TAG_R == "threadIdx.x":
                    _, ty, tx, vec = sch.split(
                        l, factors=[None, TS, TR, vec_length], preserve_unit_iters=True
                    )
                else:
                    _, ty, tx, vec = sch.split(
                        l, factors=[None, TR, TS, vec_length], preserve_unit_iters=True
                    )
                sch.bind(ty, "threadIdx.y")
                sch.bind(tx, "threadIdx.x")
                sch.vectorize(vec)

            # reduce tile_s * tr * vec to tile_s * tr
            sch.reverse_compute_at(rf2, loop=bx, preserve_unit_loops=True)
            tr, vec_c, *ts_tile_s = sch.get_loops(block=rf2)[1:]
            ts_tile_s = sch.fuse(*ts_tile_s)
            ts_o, ts_i, tile_s = sch.split(
                ts_tile_s, factors=[None, TS, TILE_S], preserve_unit_iters=True
            )
            tile_s, vec_s = sch.split(
                tile_s,
                factors=[None, get_max_factor(TILE_S, [1, 2, 4, 8])],
                preserve_unit_iters=True,
            )
            assert sch.get(ts_o).extent.value == 1
            ts = sch.fuse(ts_o, ts_i)
            sch.reorder(ts, tr, tile_s, vec_s, vec_c)
            sch.bind(ts, TAG_S)
            sch.bind(tr, TAG_R)
            sch.vectorize(vec_s)

            # reduce tile_s * tr to tile_s
            sch.reverse_compute_at(gemv, loop=bx, preserve_unit_loops=True)
            tr, *ts_tile_s = sch.get_loops(block=gemv)[1:]
            ts_tile_s = sch.fuse(*ts_tile_s)
            ts_o, ts_i, tile_s = sch.split(
                ts_tile_s, factors=[None, TS, TILE_S], preserve_unit_iters=True
            )
            assert sch.get(ts_o).extent.value == 1
            ts = sch.fuse(ts_o, ts_i)
            sch.reorder(tile_s, ts, tr)
            sch.bind(ts, TAG_S)
            sch.bind(tr, TAG_R)

            sch.decompose_reduction(rf, loop=sch.get_loops(block=rf)[3])
            sch.decompose_reduction(rf2, loop=sch.get_loops(block=rf2)[-1])

            sch.set_scope(rf, buffer_index=0, storage_scope="local")
            sch.set_scope(rf2, buffer_index=0, storage_scope="local")

            unroll_factor = UNROLL

            sch.annotate(
                block_or_loop=sch.get_loops(rf)[3],
                ann_key="pragma_auto_unroll_max_step",
                ann_val=unroll_factor,
            )
            sch.annotate(
                block_or_loop=sch.get_loops(rf)[3], ann_key="pragma_unroll_explicit", ann_val=1
            )

            sch.annotate(
                block_or_loop=sch.get_loops(rf2)[3],
                ann_key="pragma_auto_unroll_max_step",
                ann_val=unroll_factor,
            )
            sch.annotate(
                block_or_loop=sch.get_loops(rf2)[3], ann_key="pragma_unroll_explicit", ann_val=1
            )

            if LOAD_V_SHARED_LOCAL:
                sch.annotate(
                    block_or_loop=sch.get_loops(V_shared)[-4],
                    ann_key="pragma_unroll_explicit",
                    ann_val=unroll_factor,
                )
                sch.annotate(
                    block_or_loop=sch.get_loops(V_shared)[-4],
                    ann_key="pragma_vectorize",
                    ann_val=1,
                )

            # Schedule epilogue
            if epilogue_info is not None:
                epilogue = epilogue_info.block_rv
                TX = TS * TR
                if is_broadcast_epilogue(sch, block, epilogue):
                    sch.reverse_compute_at(epilogue, bx)
                    sch.set_scope(block, 0, "shared")
                    _, _, *s = sch.get_loops(epilogue)
                    _, tx = sch.split(sch.fuse(*s), factors=[None, TX])
                    sch.bind(tx, "threadIdx.x")
                else:
                    sch.reverse_compute_at(epilogue, bx, preserve_unit_loops=True)
                    ts_tile_s = sch.fuse(*sch.get_loops(epilogue)[1:])
                    ts_tile_s = sch.get_loops(epilogue)[-1]
                    ts_o, ts_i, tile_s = sch.split(
                        ts_tile_s, factors=[None, TS, TILE_S], preserve_unit_iters=True
                    )
                    assert sch.get(ts_o).extent.value == 1
                    ts = sch.fuse(ts_o, ts_i)
                    sch.bind(ts, TAG_S)
                    sch.set_scope(block, 0, "local")
            return sch

        # Determine tile sizes based on loop extents and target
        batch, s, r, c = sch.get_loops(block=block)
        len_batch, len_s, len_r, len_c = (
            get_extent(sch, batch),
            get_extent(sch, s),
            get_extent(sch, r),
            get_extent(sch, c),
        )
        len_S = len_batch * len_s
        len_R = len_r * len_c

        TAG_S, TAG_R = "threadIdx.y", "threadIdx.x"
        SUPPORT_WARP_SHUFFLE = False
        VEC_LOAD = 1
        if target.kind.name == "cuda":
            VEC_C = 4
            LOAD_V_SHARED = True
            LOAD_V_VEC = 8
            VEC_LOAD = 16
            UNROLL = 256
            SUPPORT_WARP_SHUFFLE = True
            if isinstance(len_S, int):
                TS, TR = 4, 64
            else:
                TS, TR = 1, 64
        elif target.kind.name == "metal":
            TAG_S, TAG_R = "threadIdx.x", "threadIdx.y"
            VEC_C = 1
            LOAD_V_SHARED = False
            LOAD_V_VEC = -1
            UNROLL = 256
            SUPPORT_WARP_SHUFFLE = True
            if isinstance(len_S, int):
                if len_S > len_R:
                    TS, TR = 4, 16
                else:
                    TS, TR = 2, 64
            else:
                TS, TR = 1, 64
        elif target.kind.name == "rocm":
            VEC_C = 4
            LOAD_V_SHARED = False
            LOAD_V_VEC = 8
            UNROLL = 256
            if isinstance(len_S, int):
                if len_S > len_R:
                    TS, TR = 1, 128
                else:
                    TS, TR = 8, 64
            else:
                TS, TR = 1, 64
        else:
            VEC_C = 1
            LOAD_V_SHARED = False
            LOAD_V_VEC = -1
            UNROLL = 64
            TS, TR = 1, 64

        while TS * TR > target.max_num_threads:
            if TS > 1:
                TS //= 2
            else:
                TR //= 2

        TILE_S, TILE_R = (
            1,
            (
                len_c
                if len_c > 1
                else max(get_max_factor(len_r, [TR * 1, TR * 2, TR * 4, TR * 8]) // TR, 1)
            ),
        )
        VEC_C = min(get_max_factor(TILE_R, [1, 2, 4, 8]), VEC_C)

        return apply(
            sch,
            gemv=block,
            TAG_S=TAG_S,
            TAG_R=TAG_R,
            TS=TS,
            TR=TR,
            TILE_S=TILE_S,
            TILE_R=TILE_R,
            VEC_LOAD=VEC_LOAD,
            VEC_C=VEC_C,
            LOAD_V_SHARED=LOAD_V_SHARED,
            LOAD_V_VEC=LOAD_V_VEC,
            UNROLL=UNROLL,
            SUPPORT_WARP_SHUFFLE=SUPPORT_WARP_SHUFFLE,
        )

    def apply_config(
        self, func: tir.PrimFunc, config
    ) -> Union[None, tir.Schedule, List[tir.Schedule]]:
        from .element_wise import _resolve_target_from_config

        target = _resolve_target_from_config(config)
        return self.apply(func, target, False)
