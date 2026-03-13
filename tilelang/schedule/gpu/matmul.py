# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""A tile-oriented matmul schedule rule for TileLang."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

from tilelang import tvm
from tvm.target import Target

from tilelang.carver.analysis import get_reduction_blocks, get_root_block
from tilelang.carver.matmul_analysis import auto_inline_producers, normalize_to_matmul

from .. import Schedule as TileSchedule
from . import utils
from .base import GPUScheduleRule
from .element_wise import _resolve_target_from_config

tir = tvm.tir


@dataclass(frozen=True)
class _MatmulTileConfig:
    block_m: int
    block_n: int
    block_k: int
    num_stages: int
    num_threads: int


def _largest_power_of_two_at_most(value: int) -> int:
    result = 1
    while (result << 1) <= value:
        result <<= 1
    return result


def _as_static_int(expr: tir.PrimExpr) -> Optional[int]:
    if isinstance(expr, tir.IntImm):
        return int(expr.value)
    return None


def _choose_static_tile(candidates: List[int], extent: Optional[int]) -> int:
    if extent is None:
        return candidates[0]
    for candidate in candidates:
        if extent >= candidate:
            return candidate
    return candidates[-1]


def _choose_tile_config(target: Target, block_stmt: tir.Block) -> _MatmulTileConfig:
    input_bits = max(tvm.DataType(region.buffer.dtype).bits for region in block_stmt.reads)
    max_threads = min(int(utils.max_threads_per_block(target)), 256)
    num_threads = max(1, _largest_power_of_two_at_most(max_threads))
    iter_extents = [_as_static_int(iter_var.dom.extent) for iter_var in block_stmt.iter_vars]
    m_extent = iter_extents[1] if len(iter_extents) >= 4 else None
    n_extent = iter_extents[2] if len(iter_extents) >= 4 else None
    k_extent = iter_extents[3] if len(iter_extents) >= 4 else None
    if input_bits <= 16:
        num_stages = 2 if target.kind.name in {"cuda", "hip", "rocm"} else 0
        return _MatmulTileConfig(
            block_m=_choose_static_tile([64, 32, 16], m_extent),
            block_n=_choose_static_tile([64, 32, 16], n_extent),
            block_k=_choose_static_tile([32, 16], k_extent),
            num_stages=num_stages,
            num_threads=num_threads,
        )
    return _MatmulTileConfig(
        block_m=_choose_static_tile([32, 16, 8], m_extent),
        block_n=_choose_static_tile([32, 16, 8], n_extent),
        block_k=_choose_static_tile([16, 8], k_extent),
        num_stages=0,
        num_threads=num_threads,
    )


def _is_injective_block(block_stmt: tir.Block) -> bool:
    return all(iter_var.iter_type == tir.IterVar.DataPar for iter_var in block_stmt.iter_vars)


def _has_block(sch: TileSchedule, name: str) -> bool:
    try:
        sch.get_block(name)
    except Exception:  # pylint: disable=broad-except
        return False
    return True


def _collect_injective_consumer_chain(
    sch: TileSchedule,
    main_block: tir.schedule.BlockRV,
) -> Optional[List[str]]:
    chain: List[str] = []
    current = main_block
    while True:
        consumers = list(sch.get_consumers(current))
        if not consumers:
            break
        if len(consumers) != 1:
            return None
        consumer = consumers[0]
        if not _is_injective_block(sch.get(consumer)):
            return None
        chain.append(sch.get(consumer).name_hint)
        current = consumer
    return chain


def _expr_uses_var(expr: tir.PrimExpr, var: tir.Var) -> bool:
    used = False

    def _visit(node):
        nonlocal used
        if isinstance(node, tir.Var) and node.same_as(var):
            used = True

    tir.stmt_functor.post_order_visit(expr, _visit)
    return used


def _infer_transpose_flags(block_stmt: tir.Block) -> Tuple[bool, bool]:
    reduce_vars = [
        iter_var.var for iter_var in block_stmt.iter_vars
        if iter_var.iter_type == tir.IterVar.CommReduce
    ]
    if len(reduce_vars) != 1 or len(block_stmt.reads) != 2:
        return False, False

    reduce_var = reduce_vars[0]
    a_region = block_stmt.reads[0].region
    b_region = block_stmt.reads[1].region
    if len(a_region) < 2 or len(b_region) < 2:
        return False, False

    transpose_a = not _expr_uses_var(a_region[-1].min, reduce_var)
    transpose_b = _expr_uses_var(b_region[-1].min, reduce_var)
    return transpose_a, transpose_b


def _can_use_tile_gemm(
    target: Target,
    original_block_stmt: tir.Block,
) -> bool:
    if target.kind.name not in {"cuda", "hip", "rocm"}:
        return False
    input_dtypes = [region.buffer.dtype for region in original_block_stmt.reads]
    output_dtype = original_block_stmt.writes[0].buffer.dtype
    if any(dtype not in {"float16", "bfloat16", "int8", "uint8"} for dtype in input_dtypes):
        return False
    if output_dtype not in {"float16", "float32", "int32"}:
        return False
    return True


def _analyze_original_chain(sch: TileSchedule) -> Optional[Tuple[str, List[str]]]:
    root = get_root_block(sch)
    blocks = list(sch.get_child_blocks(root))
    if not blocks:
        return None

    reduction_blocks = get_reduction_blocks(sch, blocks)
    if reduction_blocks is None or len(reduction_blocks) != 1:
        return None

    main_block = reduction_blocks[0]
    main_stmt = sch.get(main_block)
    if len(main_stmt.reads) != 2 or len(main_stmt.writes) != 1:
        return None

    main_name = main_stmt.name_hint
    main_index = next(
        (idx for idx, block in enumerate(blocks) if sch.get(block).name_hint == main_name),
        None,
    )
    if main_index is None:
        return None

    for block in blocks[:main_index]:
        if not _is_injective_block(sch.get(block)):
            return None

    epilogue_names: List[str] = []
    for block in blocks[main_index + 1 :]:
        if not _is_injective_block(sch.get(block)):
            return None
        epilogue_names.append(sch.get(block).name_hint)

    return main_name, epilogue_names


class Matmul(GPUScheduleRule):
    """Tile-first matmul scheduling with shared staging and fragment accumulation."""

    def apply(
        self,
        func: tir.PrimFunc,
        target: Target,
        _: bool,
    ) -> Union[None, tir.Schedule, List[tir.Schedule]]:
        if not isinstance(func, tir.PrimFunc) or not self.is_target_available(target):
            return None

        try:
            sch = TileSchedule(func)
            chain = _analyze_original_chain(sch)
            if chain is None:
                return None

            main_name, epilogue_names = chain
            main_block = sch.get_block(main_name)
            original_block_stmt = sch.get(main_block)
            transpose_a, transpose_b = _infer_transpose_flags(original_block_stmt)
            use_tile_gemm = _can_use_tile_gemm(target, original_block_stmt)
            sch = normalize_to_matmul(sch, main_block, layout=["a", "a", "n"])
            if sch is None:
                return None

            main_block = sch.get_block(main_name)
            auto_inline_producers(sch, main_block)
            main_block = sch.get_block(main_name)
            block_stmt = sch.get(main_block)
            loops = sch.get_loops(main_block)
            if len(loops) != 4:
                return None

            config = _choose_tile_config(target, block_stmt)

            if epilogue_names:
                for name in epilogue_names[:-1]:
                    if _has_block(sch, name):
                        sch.compute_inline(sch.get_block(name))

            sch.pad_einsum(
                main_block,
                [1, config.block_m, config.block_n, config.block_k],
            )
            main_block = sch.get_block(main_name)
            post_chain = _collect_injective_consumer_chain(sch, main_block)
            if not post_chain:
                return None
            materialize_block_name = post_chain[0]
            output_block_name = post_chain[-1]
            has_epilogue = len(post_chain) > 1
            batch, m_loop, n_loop, k_loop = sch.get_loops(main_block)

            m_outer, m_inner = sch.split(
                m_loop,
                factors=[None, config.block_m],
                preserve_unit_iters=True,
            )
            n_outer, n_inner = sch.split(
                n_loop,
                factors=[None, config.block_n],
                preserve_unit_iters=True,
            )
            k_outer, k_inner = sch.split(
                k_loop,
                factors=[None, config.block_k],
                preserve_unit_iters=True,
            )
            sch.reorder(batch, m_outer, n_outer, k_outer, m_inner, n_inner, k_inner)
            if config.num_stages > 0:
                sch.pipeline(k_outer, config.num_stages)

            block_y = sch.fuse(batch, m_outer)
            sch.bind(block_y, "blockIdx.y")
            sch.bind(n_outer, "blockIdx.x")

            for block_name in post_chain:
                sch.reverse_compute_at(
                    sch.get_block(block_name),
                    n_outer,
                    preserve_unit_loops=True,
                )

            main_block = sch.get_block(main_name)
            sch.cache_write_at(
                n_outer,
                main_block,
                0,
                "local.fragment",
                write_back=False,
            )

            main_block = sch.get_block(main_name)
            if use_tile_gemm:
                sch.fill_at(n_outer, main_block, 0, 0.0)
                main_block = sch.get_block(main_name)
            sch.cache_read_at(k_outer, main_block, 0, "shared")
            main_block = sch.get_block(main_name)
            sch.cache_read_at(k_outer, main_block, 1, "shared")

            output_block = sch.get_block(output_block_name)
            output_loops = sch.get_loops(output_block)
            if output_loops and (has_epilogue or not use_tile_gemm):
                sch.parallel(output_loops[-1])

            if use_tile_gemm:
                main_block = sch.get_block(main_name)
                sch.gemm_at(
                    k_outer,
                    main_block,
                    transpose_a=transpose_a,
                    transpose_b=transpose_b,
                )
                if has_epilogue:
                    bridge_block = sch.get_block(materialize_block_name)
                    sch.cache_write_at(
                        n_outer,
                        bridge_block,
                        0,
                        "shared",
                        write_back=False,
                    )
                    bridge_block = sch.get_block(materialize_block_name)
                    sch.copy_at(n_outer, bridge_block)
                else:
                    sch.copy_at(n_outer, sch.get_block(output_block_name))
            else:
                main_block = sch.get_block(main_name)
                main_loops = sch.get_loops(main_block)
                if len(main_loops) < 3:
                    return None
                inner_work = sch.fuse(main_loops[-3], main_loops[-2])
                sch.parallel(inner_work)

            sch.launch_thread(sch.get_block("root"), config.num_threads)
            return sch
        except Exception:  # pylint: disable=broad-except
            return None

    def apply_config(
        self,
        func: tir.PrimFunc,
        config,
    ) -> Union[None, tir.Schedule, List[tir.Schedule]]:
        target = _resolve_target_from_config(config)
        return self.apply(func, target, False)
