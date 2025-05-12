# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""The language interface for tl programs."""

from typing import Optional
# from .parser import *
# now is fully compatible with the upstream
# tir script
# TODO(lei): remove this import once the
# upstream tir script is fully compatible
from tvm.script.parser.tir import *
from .tir import (
    prim_func,  # noqa: F401
)
from .tir.ir import *  # noqa: F401
from tilelang.layout import Layout, Fragment  # noqa: F401
from .proxy import (
    ptr,  # noqa: F401
    make_tensor,  # noqa: F401
    Buffer,  # noqa: F401
    Tensor,  # noqa: F401
    FragmentBuffer,  # noqa: F401
    SharedBuffer,  # noqa: F401
    LocalBuffer,  # noqa: F401
)
from .parallel import Parallel  # noqa: F401
from .pipeline import Pipelined  # noqa: F401
from .frame import has_let_value, get_let_value  # noqa: F401
from .kernel import (
    Kernel,  # noqa: F401
    KernelLaunchFrame,  # noqa: F401
    get_thread_binding,  # noqa: F401
    get_thread_bindings,  # noqa: F401
    get_block_binding,  # noqa: F401
    get_block_bindings,  # noqa: F401
)
from .warpgroup import ws  # noqa: F401
from .allocate import (
    alloc_local,  # noqa: F401
    alloc_shared,  # noqa: F401
    alloc_fragment,  # noqa: F401
    alloc_var,  # noqa: F401
)
from .copy import copy, c2d_im2col  # noqa: F401
from .gemm import GemmWarpPolicy, gemm  # noqa: F401
from .fill import fill, clear  # noqa: F401
from .reduce import (
    reduce,  # noqa: F401
    reduce_max,  # noqa: F401
    reduce_min,  # noqa: F401
    reduce_sum,  # noqa: F401
    reduce_abssum,  # noqa: F401
    reduce_absmax,  # noqa: F401
    cumsum,  # noqa: F401
)
from .print import print  # noqa: F401
from .customize import (
    atomic_add,  # noqa: F401
    atomic_addx2,  # noqa: F401
    dp4a,  # noqa: F401
    clamp,  # noqa: F401
    reshape,  # noqa: F401
    view,  # noqa: F401
)
from .logical import any_of, all_of  # noqa: F401
from .builtin import *  # noqa: F401

from .memscope import *  # noqa: F401


def symbolic(name: str, dtype: str = "int32"):
    return tir.Var(name, dtype)


def use_swizzle(panel_size: int, order: str = "row", enable: bool = True):
    # If order is row, use rasterization2DRow, otherwise use rasterization2DColumn
    # The panel size is the number of threads in a warp
    # Use to improve the L2 Cache Locality
    device_func = ("rasterization2DRow" if order == "row" else "rasterization2DColumn")
    return attr(None, "threadblock_swizzle_pattern",
                f"tl::{device_func}<{panel_size}>") if enable else None


def annotate_layout(layout_map):
    # layout_map is a dictionary of buffer to layout
    layout_map = {buffer.data: layout for buffer, layout in layout_map.items()}
    return block_attr({"layout_map": layout_map})


def import_source(source: Optional[str] = None):
    # source is the source code to be imported
    return block_attr({"pragma_import_c": source}) if source is not None else None
