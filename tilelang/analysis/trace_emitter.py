from __future__ import annotations

import itertools
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

from numpy.random import permutation
from tvm import tir, IRModule
from tvm.tir import (
    PyStmtExprMutator,
    PyStmtExprVisitor,
    For,
    PrimFunc,
    BufferLoad,
    Var,
    Evaluate,
)
from tvm.tir.stmt_functor import pre_order_visit
from .trace import TraceEvent, GeneralizedL2Simulator

_OP_TILEOP_COPY = tir.op.Op.get("tl.copy")
@dataclass
class SimulatorConfig:
    """Configuration for the L2 cache simulator."""

    sm_count: int = 114
    panel_width: int = 1
    l2_cap_bytes: int = 50 * 1024 * 1024
    num_partitions: int = 2
    assoc: int = 8
    line_bytes: int = 128


_TEMPLATE = """\
def _generate_trace(coords_list):
    trace = []
    stride_major = {sm_count} // {panel_width}
    stride_minor = {panel_width}

    chunk_size = stride_major * stride_minor
    total_blocks = len(coords_list)

    for i in range(0, total_blocks, chunk_size):
        chunk = coords_list[i : min(i + chunk_size, total_blocks)]

        permuted_indices = permutation(len(chunk))

{code}
    return trace

trace = _generate_trace(coords_list)

simulator = GeneralizedL2Simulator(
    l2_cap_bytes={l2_cap_bytes},
    num_partitions={num_partitions},
    assoc={assoc},
    line_bytes={line_bytes}
)

simulator.run(trace)
hit_rate, l2_io, ddr_io = simulator.get_io_metrics()
_result = (hit_rate, l2_io, ddr_io)
"""


def _collect_global_tensors(buffer_map: dict) -> set[Var]:
    """Extract global tensor data pointers from a buffer map."""
    global_tensors = set()
    for buf in buffer_map.values():
        if buf.scope() == "global":
            global_tensors.add(buf.data)
    return global_tensors


class CoordinateGenerator:
    """Generates coordinates for block execution with optional swizzling.

    Supports both linear rasterization and swizzled patterns that mimic
    GPU hardware execution order.
    """

    def __init__(
        self,
        sm_count: int,
        swizzle_config: dict[str, Any] | None = None,
    ) -> None:
        """Initialize the coordinate generator.

        Args:
            sm_count: Total number of SMs (used for stride calculation).
            swizzle_config: Dict defining which dims are swizzled.
                e.g., {'major': 'by', 'minor': 'bx', 'panel_width': 1}
                If None, returns simple linear rasterization.
        """
        self.sm_count = sm_count
        self.swizzle_config = swizzle_config

    def generate(self, dims_config: dict[str, int]) -> list[dict[str, int]]:
        """Generate coordinates based on dimension configuration.

        Args:
            dims_config: Dict defining the grid size for each dimension.
                e.g., {'bx': 16, 'by': 8, 'bz': 1}

        Returns:
            List of coordinate dictionaries.
        """
        if not self.swizzle_config:
            return self._linear_rasterization(dims_config)

        major_name = self.swizzle_config["major"]
        minor_name = self.swizzle_config["minor"]
        panel_width = self.swizzle_config.get("panel_width", 1)

        stride_major = self.sm_count // panel_width
        stride_minor = panel_width

        grid_major = dims_config[major_name]
        grid_minor = dims_config[minor_name]

        swizzled_plane = self._generate_swizzled_plane(
            grid_major, grid_minor, stride_major, stride_minor
        )

        other_dims = [k for k in dims_config if k not in (major_name, minor_name)]

        return self._combine_with_outer_dims(
            swizzled_plane, major_name, minor_name, other_dims, dims_config
        )

    def _generate_swizzled_plane(
        self,
        grid_major: int,
        grid_minor: int,
        stride_major: int,
        stride_minor: int,
    ) -> list[tuple[int, int]]:
        """Generate swizzled 2D coordinates."""
        swizzled_plane = []
        major_start, minor_start = 0, 0

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

        return swizzled_plane

    def _combine_with_outer_dims(
        self,
        swizzled_plane: list[tuple[int, int]],
        major_name: str,
        minor_name: str,
        other_dims: list[str],
        dims_config: dict[str, int],
    ) -> list[dict[str, int]]:
        """Combine swizzled 2D plane with remaining dimensions via Cartesian product."""
        if not other_dims:
            return [
                {major_name: maj, minor_name: min_} for maj, min_ in swizzled_plane
            ]

        final_coords = []
        other_ranges = [range(dims_config[dim]) for dim in other_dims]

        for combination in itertools.product(*other_ranges):
            outer_dict = dict(zip(other_dims, combination))
            for maj_val, min_val in swizzled_plane:
                point = outer_dict.copy()
                point[major_name] = maj_val
                point[minor_name] = min_val
                final_coords.append(point)

        return final_coords

    def _linear_rasterization(
        self, dims_config: dict[str, int]
    ) -> list[dict[str, int]]:
        """Generate coordinates via simple linear iteration."""
        keys = list(dims_config.keys())
        ranges = [range(dims_config[k]) for k in keys]

        return [
            dict(zip(keys, combination))
            for combination in itertools.product(*ranges)
        ]


class _CodeBuilder:
    """Helper for building indented Python code strings."""

    def __init__(self, base_indent: int = 2) -> None:
        self._lines: list[str] = []
        self._indent = base_indent

    @contextmanager
    def indented(self):
        """Context manager for indented code blocks."""
        self._indent += 1
        try:
            yield
        finally:
            self._indent -= 1

    def add_line(self, line: str) -> None:
        """Add a line with current indentation."""
        self._lines.append(" " * self._indent * 4 + line)

    def get_code(self) -> str:
        """Return the generated code."""
        return "\n".join(self._lines) + "\n" if self._lines else ""


@tir.functor.mutator
class _TIREmitter(PyStmtExprMutator):
    """TIR mutator that filters operations involving global memory."""

    global_tensor: set[Var]

    def __init__(self, buffer_map: dict) -> None:
        super().__init__()
        self.global_tensor = _collect_global_tensors(buffer_map)

    def visit_call_(self, op):
        if op.op != _OP_TILEOP_COPY:
            return 0

        is_in_global = False

        def check_global(node):
            nonlocal is_in_global
            if isinstance(node, BufferLoad):
                if node.buffer.data in self.global_tensor:
                    is_in_global = True
                    return False
            return True

        pre_order_visit(Evaluate(op), check_global)
        return op if is_in_global else 0
    
    def visit_for_(self, op: For) -> For:
        # NOTE: Assume all access for global memory uses T.copy.
        if op.kind == tir.ForKind.PARALLEL:
            return Evaluate(0)
        new_body = self.visit_stmt(op.body)
        return For(op.loop_var, op.min, op.extent, op.kind, new_body)

@tir.functor.visitor
class _PyPrinter(PyStmtExprVisitor):
    """TIR visitor that generates Python trace code."""

    code: str
    var_thread_mapping: dict[Var, int]
    global_tensor: set[Var]

    def __init__(self, buffer_map: dict) -> None:
        super().__init__()
        self._builder = _CodeBuilder(base_indent=2)
        self.var_thread_mapping = {}
        self.global_tensor = _collect_global_tensors(buffer_map)

    @property
    def code(self) -> str:
        return self._builder.get_code()

    def visit_attr_stmt_(self, op) -> None:
        if op.attr_key == "thread_extent":
            if "blockIdx" in op.node.thread_tag:
                self.var_thread_mapping[op.node.var] = int(op.value)
        self.visit_stmt(op.body)

    def visit_for_(self, op: For) -> None:
        self._builder.add_line(
            f"for {op.loop_var} in range({op.min}, {op.min + op.extent}):"
        )
        with self._builder.indented():
            self.visit_stmt(op.body)

    def visit_call_(self, op) -> None:
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
        indices = tuple(buf_load.indices)
        dtype_bytes = buf.dtype.bits // 8

        index_str = str(indices)
        for var in self.var_thread_mapping:
            index_str = index_str.replace(str(var), f"pt['{var}']")

        extent = dtype_bytes
        for dim in global_region.args[2:]:
            extent *= dim

        is_write = bool(global_region.args[1] == 2)

        self._builder.add_line("for idx in permuted_indices:")
        with self._builder.indented():
            self._builder.add_line("pt = chunk[idx]")
            self._builder.add_line(
                f"trace.append(TraceEvent('{buf}', {index_str}, {extent}, "
                f"is_write={is_write}))"
            )

    def visit_let_stmt_(self, op):
        self._builder.add_line(f"{op.var} = {int(op.value)}")
        self.visit_stmt(op.body)

def l2_predict(
    func: IRModule | PrimFunc,
    config: SimulatorConfig | None = None,
) -> tuple[float, float, float]:
    """Emit and execute a trace generation function for hardware simulation.

    Given a TIR PrimFunc or IRModule, this function generates and executes
    Python code that simulates the hardware execution order with swizzling,
    running the trace through an L2 cache simulator.

    Args:
        func: The TIR function or module to analyze.
        config: Simulator configuration. Uses defaults if not provided.

    Returns:
        A tuple of (hit_rate, l2_io, ddr_io) metrics from the simulation.
    """
    if config is None:
        config = SimulatorConfig()

    if isinstance(func, IRModule):
        items = func.functions_items()
        if len(items) != 1:
            raise ValueError(
                f"Expected single function module, got {len(items)} functions"
            )
        func = items[0][1]

    emitter = _TIREmitter(func.buffer_map)
    tir_code = emitter.visit_stmt(func.body)

    printer = _PyPrinter(func.buffer_map)
    printer.visit_stmt(tir_code)

    dims = {str(k): int(v) for k, v in printer.var_thread_mapping.items()}

    if len(dims) < 2:
        raise ValueError(
            f"Expected at least 2 block dimensions, got {len(dims)}: {list(dims.keys())}"
        )

    dim_keys = list(dims.keys())
    gen = CoordinateGenerator(
        sm_count=config.sm_count,
        swizzle_config={
            "major": dim_keys[1],
            "minor": dim_keys[0],
            "panel_width": config.panel_width,
        },
    )

    coords_list = gen.generate(dims)

    code = _TEMPLATE.format(
        sm_count=config.sm_count,
        panel_width=config.panel_width,
        l2_cap_bytes=config.l2_cap_bytes,
        num_partitions=config.num_partitions,
        assoc=config.assoc,
        line_bytes=config.line_bytes,
        code=printer.code,
    )

    exec_globals = {
        "coords_list": coords_list,
        "permutation": permutation,
        "TraceEvent": TraceEvent,
        "GeneralizedL2Simulator": GeneralizedL2Simulator,
    }
    exec(code, exec_globals)

    return exec_globals["_result"]

