from __future__ import annotations
from tvm import tir, IRModule
from tvm.tir import (PyStmtExprMutator, PyStmtExprVisitor, For, PrimFunc, BufferLoad, Var, Evaluate)
from tvm.tir.stmt_functor import pre_order_visit
import math
from numpy.random import permutation
from .trace import TraceEvent, GeneralizedL2Simulator

_OP_TILEOP_COPY = tir.op.Op.get("tl.copy")

_TEMPLATE = """
def _generate_trace(coords_list):
    trace = []
    Stride_M = {sm_count} // {pannel}
    Stride_N = {pannel}
            
    chunk_size = Stride_M * Stride_N
    total_blocks = len(coords_list)
    
    for i in range(0, total_blocks, chunk_size):
        chunk = coords_list[i : min(i + chunk_size, total_blocks)]
        
        permuted_indices = permutation(len(chunk))
        
{code}
    return trace

trace = _generate_trace(coords_list)

simulator = GeneralizedL2Simulator(
    l2_cap_bytes=50*1024*1024,
    num_partitions=2, 
    assoc=8, 
    line_bytes=128
)

simulator.run(trace)
hit_rate, l2_io, ddr_io = simulator.get_io_metrics()
print(hit_rate)
"""

class CoordinateGenerator:
    def __init__(self, sm_count, swizzle_config=None):
        """
        sm_count: Total number of SMs (used for stride calculation)
        swizzle_config: Dict defining which dims are swizzled.
                        e.g., {'major': 'by', 'minor': 'bx', 'panel_width': 1}
                        If None, returns simple linear rasterization.
        """
        self.sm_count = sm_count
        self.swizzle_config = swizzle_config

    def generate(self, dims_config):
        """
        dims_config: Dict defining the grid size for each dimension.
                     e.g., {'bx': 16, 'by': 8, 'bz': 1}
                     Keys can be any string identifier.
        """
        # 1. Identify Swizzle Dimensions vs. Outer Dimensions
        if self.swizzle_config:
            major_name = self.swizzle_config['major']
            minor_name = self.swizzle_config['minor']
            panel_width = self.swizzle_config.get('panel_width', 1)
            
            # Calculate Strides for the swizzled dims
            stride_major = self.sm_count // panel_width
            stride_minor = panel_width
            
            # Extract grid limits for swizzled dims
            grid_major = dims_config[major_name]
            grid_minor = dims_config[minor_name]
        else:
            # Fallback if no swizzling needed
            return self._linear_rasterization(dims_config)

        # 2. Generate Swizzled 2D Plane
        # We generate the list of (major, minor) tuples first
        swizzled_plane = []
        major_start, minor_start = 0, 0
        
        # Note: Using 0-based indexing internally
        while major_start < grid_major and minor_start < grid_minor:
            major_end = min(major_start + stride_major, grid_major)
            minor_end = min(minor_start + stride_minor, grid_minor)
            
            for maj in range(major_start, major_end):
                for min_ in range(minor_start, minor_end):
                    swizzled_plane.append((maj, min_))
            
            major_start = major_end
            if major_start >= grid_major:
                major_start = 0
                minor_start = minor_end

        # 3. Handle Remaining Dimensions (Cartesian Product)
        # Identify dims that are NOT involved in swizzling (e.g., 'bz', 'batch')
        other_dims = [k for k in dims_config if k not in [major_name, minor_name]]
        
        final_coords = []
        
        # Recursive function to combine swizzled 2D plane with other dimensions
        def recurse_dims(current_dict, remaining_dims):
            if not remaining_dims:
                # Base case: All outer dims set, now iterate the swizzled plane
                for (maj_val, min_val) in swizzled_plane:
                    # Create the final point dictionary
                    point = current_dict.copy()
                    point[major_name] = maj_val
                    point[minor_name] = min_val
                    final_coords.append(point)
                return

            # Recursive step: Iterate next outer dimension
            dim_key = remaining_dims[0]
            dim_size = dims_config[dim_key]
            for i in range(dim_size):
                current_dict[dim_key] = i
                recurse_dims(current_dict, remaining_dims[1:])

        recurse_dims({}, other_dims)
        
        return final_coords

    def _linear_rasterization(self, dims_config):
        """Helper for non-swizzled simple iteration"""
        import itertools
        keys = list(dims_config.keys())
        ranges = [range(dims_config[k]) for k in keys]
        
        final_coords = []
        for combination in itertools.product(*ranges):
            point = {k: v for k, v in zip(keys, combination)}
            final_coords.append(point)
        return final_coords

@tir.functor.mutator
class _TIREmitter(PyStmtExprMutator):
    global_tensor: set[Var]

    def __init__(self, buffer_map) -> None:
        super().__init__()
        self.global_tensor = set()
        for _, buf in buffer_map.items():
            if buf.scope() == "global":
                self.global_tensor.add(buf.data)
    
    def visit_call_(self, op):
        if op.op == _OP_TILEOP_COPY:
            is_in_global = False
            def in_global(node):
                nonlocal is_in_global
                if isinstance(node, BufferLoad):
                    if node.buffer.data in self.global_tensor:
                        is_in_global = True
                        return False
                return True
            pre_order_visit(Evaluate(op), in_global)
            if is_in_global:
                return op
            return 0
        return 0
    
@tir.functor.visitor
class _PyPrinter(PyStmtExprVisitor):
    code: str
    var_thread_mapping: dict[Var, int]
    global_tensor: set[Var]

    def __init__(self, buffer_map) -> None:
        super().__init__()
        self.code = ""
        self.var_thread_mapping = {}
        self.global_tensor = set()
        for _, buf in buffer_map.items():
            if buf.scope() == "global":
                self.global_tensor.add(buf.data)
        self.indent = 2
    
    def print_indent(self):
        self.code += " " * self.indent * 4
    
    def begin_scope(self):
        self.indent += 1
    
    def end_scope(self):
        self.indent -= 1
    
    def visit_attr_stmt_(self, op):
        if op.attr_key == "thread_extent":
            if "blockIdx" in op.node.thread_tag:
                _n = int(op.value)
                self.var_thread_mapping[op.node.var] = _n
        self.visit_stmt(op.body)
    
    def visit_for_(self, op: For):
        self.print_indent()
        self.code += f"for {op.loop_var} in range({op.min}, {op.min + op.extent}):\n"
        self.begin_scope()
        self.visit_stmt(op.body)
        self.end_scope()
    
    def visit_call_(self, op):
        src_scope = op.args[0].args[0].buffer.scope()
        dst_scope = op.args[1].args[0].buffer.scope()
        if src_scope == "global":
            global_region = op.args[0]
        elif dst_scope == "global":
            global_region = op.args[1]
        else:
            return
        buf_load = global_region.args[0]
        buf = buf_load.buffer
        index = tuple(buf_load.indices)
        dtype_bytes = buf.dtype.bits // 8
        index_str = str(index)

        for k in self.var_thread_mapping:
            index_str = index_str.replace(str(k), f"pt['{k}']")
        extent = 1
        for dim in global_region.args[2:]:
            extent *= dim

        extent *= dtype_bytes
        self.print_indent()
        self.code += f"for idx in permuted_indices:\n"
        self.begin_scope()
        self.print_indent()
        self.code += f"pt = chunk[idx]\n"
        self.print_indent()
        self.code += f"trace.append(TraceEvent('{buf}', {index_str}, {extent}, is_write={bool(global_region.args[1] == 2)}))\n"

        self.end_scope()


def TraceEmitter(func: IRModule | PrimFunc) -> None:
    """
    Given a TIR PrimFunc or IRModule, emit and execute a trace generation
    function that mimics the hardware execution order with swizzling.
    The emitted code is printed out for inspection.
    """
    if isinstance(func, IRModule):
        items = func.functions_items()
        assert len(items) == 1, "Temporarily only support single function module"
        func = items[0][1]
    emitter = _TIREmitter(func.buffer_map)
    tir_code = emitter.visit_stmt(func.body)

    printer = _PyPrinter(func.buffer_map)
    printer.visit_stmt(tir_code)
    dims = {}
    for k, v in printer.var_thread_mapping.items():
        dims[str(k)] = int(v)

    gen = CoordinateGenerator(
        sm_count=114,
        swizzle_config={
            'major': list(dims.keys())[0],
            'minor': list(dims.keys())[1],
            'panel_width': 1})

    coords_list = gen.generate(dims)
    code = _TEMPLATE.format(
        sm_count = 114,
        pannel = 1,
        code = printer.code,
    )
    exec(code)
