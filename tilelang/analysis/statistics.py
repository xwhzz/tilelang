from __future__ import annotations
from collections import defaultdict
from enum import IntEnum, auto
import tabulate
from tvm import tir, DataType, IRModule
from tvm.tir import (PyStmtExprVisitor, BufferStore, For, Buffer, PrimFunc, BufferLoad, IntImm, Call, AttrStmt)
from tilelang.arch import get_arch_config
from .L2_hit_rate_flow_sim_reuse_distance import L2_hit_rate_flow_sim_reuse_distance

class _StatisticType(IntEnum):
    TC_1 = auto() # 8bit
    TC_2 = auto() # 16bit
    CUDA_2 = auto() # 16bit
    CUDA_4 = auto() # 32bit
    CUDA_8 = auto() # 64bit
    SFU = auto() # sfu
    L0 = auto()
    L1 = auto()
    L2 = auto()
    L3 = auto()

_map = {
    "local.fragment": _StatisticType.L0,
    "shared.dyn": _StatisticType.L1,
    "shared": _StatisticType.L1,
    "global": _StatisticType.L3,
}


@tir.functor.visitor
class _ComputeIOCollector(PyStmtExprVisitor):

    def __init__(self) -> None:
        super().__init__()
        self._info = defaultdict(float)
        self._simt_op = 0
        self._extent = 1

        self._arch = get_arch_config("H100_PCIE", profile="microbench")
        self._l2_hit_rate = L2_hit_rate_flow_sim_reuse_distance(1024, 1024, 1024, 128, 128, 128, self._arch.l2_capacity, self._arch.sm_count, {'in1': [1,1,1,2], 'in2': [1,1,1,2], 'out1': [1,1,1,4]}, row_panel=1)

        self._block_num = 1

    @staticmethod
    def _get_extent(region: Call) -> int:
        extent = 1
        for arg in region.args[2:]:
            extent *= arg
        return extent
    
    @staticmethod
    def _get_scope(region: Call | Buffer) -> _StatisticType:
        if isinstance(region, Buffer):
            return _map[region.scope()]
        return _map[region.args[0].buffer.scope()]
    
    @staticmethod
    def _get_core(dtype: DataType, is_tc: bool = False) -> _StatisticType:
        if is_tc:
            if dtype.bits == 8:
                return _StatisticType.TC_1
            elif dtype.bits == 16:
                return _StatisticType.TC_2
            else:
                raise NotImplementedError(f"Unsupported TC dtype: {dtype}")
        else:
            if dtype.bits == 16:
                return _StatisticType.CUDA_2
            elif dtype.bits == 32:
                return _StatisticType.CUDA_4
            elif dtype.bits == 64:
                return _StatisticType.CUDA_8
            else:
                raise NotImplementedError(f"Unsupported CUDA dtype: {dtype}")
            
    def visit_attr_stmt_(self, op: AttrStmt) -> None:
        if op.attr_key == "thread_extent":
            if "blockIdx" in op.node.thread_tag:
                self._block_num *= int(op.value)
        self.visit_stmt(op.body)
        
    def visit_for_(self, op: For) -> None:
        if isinstance(op.extent, IntImm):
            self._extent *= op.extent.value
            self.visit_stmt(op.body)
            self._extent //= op.extent.value
        else:
            self.visit_stmt(op.body)
    
    def visit_call_(self, op) -> None:
        if op.op == tir.op.Op.get("tl.copy"):
            src_region = op.args[0]
            dst_region = op.args[1]
            # src_dtype = self._get_core(src_region.args[0].buffer.dtype)
            # dst_dtype = self._get_core(dst_region.args[0].buffer.dtype)
            src_extent = self._get_extent(src_region)
            dst_extent = self._get_extent(dst_region)
            assert src_extent == dst_extent, "Mismatched copy extents."
            src_scope = self._get_scope(src_region)
            dst_scope = self._get_scope(dst_region)

            src_dtype = src_region.args[0].buffer.dtype

            # self._info[max(src_scope, dst_scope)] += src_extent * self._extent * src_region.args[0].buffer.dtype.bits // 8
            io = int(src_extent * self._extent * src_dtype.bits // 8)
            if src_scope == _StatisticType.L3:
                # load L2 two partition:
                self._info[_StatisticType.L2] += io * self._l2_hit_rate + io * (1 - self._l2_hit_rate) * 2
                self._info[_StatisticType.L3] += io * (1 - self._l2_hit_rate)
            elif dst_scope == _StatisticType.L3:
                # store
                self._info[_StatisticType.L2] += io * 2
                self._info[_StatisticType.L3] += io

                self._info[_StatisticType.L1] += io
            # TODO: consider the overhead of dtype conversion
            # if src_dtype != dst_dtype:
            #     self._info[max(src_dtype, dst_dtype)] += src_extent * self._extent

        elif op.op == tir.op.Op.get("tl.gemm_py") or op.op == tir.op.Op.get("tl.gemm"):
            left_dtype = op.args[0].args[0].buffer.dtype
            right_dtype = op.args[1].args[0].buffer.dtype
            left_type = self._get_core(left_dtype, True)
            right_type = self._get_core(right_dtype, True)

            assert left_type == right_type, "Mismatched gemm types."

            self._info[left_type] += int(op.args[5] * op.args[6] * op.args[7] * 2 * self._extent)
            if self._get_scope(op.args[0]) == _StatisticType.L1:
                self._info[_StatisticType.L1] += int(self._get_extent(op.args[0]) * self._extent * left_dtype.bits // 8 * 2)
            if self._get_scope(op.args[1]) == _StatisticType.L1:
                self._info[_StatisticType.L1] += int(self._get_extent(op.args[1]) * self._extent * right_dtype.bits // 8 * 2)
            

        
        elif op.op == tir.op.Op.get("tl.fill"):
            dtype = op.args[0].args[0].buffer.dtype
            _type = self._get_core(dtype)
            self._info[_type] += int(self._get_extent(op.args[0]) * self._extent)
            
        elif op.op == tir.op.Op.get("tir.exp2"): # sfu
            self._info[_StatisticType.SFU] += int(1 * self._extent)
        return super().visit_call_(op)

    
    def visit_buffer_store_(self, op: BufferStore):
        self._simt_op = 0
        self._simt_op += 1
        _type = self._get_core(op.buffer.dtype)
        self._info[_type] += int(1 * self._extent)
        self.visit_expr(op.value)
        self._simt_op = 0

    def visit_buffer_load_(self, op: BufferLoad):
        scope = self._get_scope(op.buffer)
        if scope > _StatisticType.L0:
            self._info[scope] += int(self._extent)
    
    def get_info(self) -> dict:
        return self._info
    
    def post_data(self) -> None:
        norm = (self._block_num + self._arch.sm_count - 1) // self._arch.sm_count * self._arch.sm_count


        ## memory time
        l3_time = self._info[_StatisticType.L3] / self._arch.ddr_bandwidth * norm
        l2_time = self._info[_StatisticType.L2] / self._arch.l2_bandwidth * norm
        l1_time = self._info[_StatisticType.L1] / self._arch.smem_bandwidth * norm

        ## compute time
        tc_1_time = self._info[_StatisticType.TC_1] / self._arch.int8_flops * norm
        tc_2_time = self._info[_StatisticType.TC_2] / self._arch.fp16_tensor_flops * norm

        cuda_2_time = self._info[_StatisticType.CUDA_2] / self._arch.fp16_cuda_core_flops * norm
        cuda_4_time = self._info[_StatisticType.CUDA_4] / self._arch.fp32_cuda_core_flops * norm
        cuda_8_time = self._info[_StatisticType.CUDA_8] / self._arch.fp64_cuda_core_flops * norm

        sfu_time = self._info[_StatisticType.SFU] / self._arch.sfu_flops * norm

        # helper formatters
        def _fmt_bytes(b: float) -> str:
            b = float(b)
            for unit in ("B", "KB", "MB", "GB", "TB"):
                if abs(b) < 1024.0:
                    return f"{b:,.2f} {unit}"
                b /= 1024.0
            return f"{b:,.2f} PB"

        def _fmt_time_s(t: float) -> str:
            return f"{t*1000:,.6f} ms"

        def _safe_div(n: float, d: float) -> float:
            return float(n) / d if d and d != 0 else 0.0

        # prepare components
        components = [
            ("L3", _StatisticType.L3, l3_time, "mem"),
            ("L2", _StatisticType.L2, l2_time, "mem"),
            ("L1", _StatisticType.L1, l1_time, "mem"),
            ("TC_1", _StatisticType.TC_1, tc_1_time, "compute"),
            ("TC_2", _StatisticType.TC_2, tc_2_time, "compute"),
            ("CUDA_2", _StatisticType.CUDA_2, cuda_2_time, "compute"),
            ("CUDA_4", _StatisticType.CUDA_4, cuda_4_time, "compute"),
            ("CUDA_8", _StatisticType.CUDA_8, cuda_8_time, "compute"),
            ("SFU", _StatisticType.SFU, sfu_time, "compute"),
        ]

        # compute bound and build enhanced table
        bound_time = 0.0
        for _, _, t, _ in components:
            if t > bound_time:
                bound_time = t

        enhanced = []
        for name, key, t, kind in components:
            pct = _safe_div(t, bound_time) * 100.0 if bound_time > 0 else 0.0
            raw = self._info.get(key, 0) * norm
            if kind == "mem":
                data_str = _fmt_bytes(raw)
                bw = _safe_div(raw, bound_time) / 1e9 if t > 0 else 0.0
                bw_str = f"{bw:,.3f} GB/s" if bw else "-"
            else:
            # compute: raw represents operation counts (ops)
                ops = int(raw)
                data_str = f"{ops:,} ops"
                tp = _safe_div(ops, bound_time) / 1e9 if t > 0 else 0.0
                bw_str = f"{tp:,.3f} GOP/s" if tp else "-"

            enhanced.append([name, _fmt_time_s(t), f"{pct:5.1f}%", data_str, bw_str])

        print(tabulate.tabulate(enhanced, headers=["Component", "Time", "Util", "Data", "Throughput"], tablefmt="github"))

        # summary
        bottlenecks = [name for name, _, t, _ in components if abs(t - bound_time) < 1e-12 or t == bound_time]
        bottleneck_str = ", ".join(bottlenecks) if bottlenecks else "N/A"
        print(f"Estimated bound time: {bound_time * 1000: .6f} ms  (bottleneck: {bottleneck_str})")


def ComputeIOCollector(func: IRModule | PrimFunc) -> None:
    if isinstance(func, IRModule):
        items = func.functions_items()
        assert len(items) == 1, "Temporarily only support single function module"
        func = items[0][1]
    collector = _ComputeIOCollector()
    collector.visit_stmt(func.body)

    collector.post_data()
