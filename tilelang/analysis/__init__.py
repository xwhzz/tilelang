"""Tilelang IR analysis & visitors."""
from . import _ffi_api
from tvm import IRModule
from tvm.tir import PrimFunc
from .ast_printer import ASTPrinter  # noqa: F401
from .nested_loop_checker import NestedLoopChecker  # noqa: F401
from .fragment_loop_checker import FragmentLoopChecker  # noqa: F401
from .statistics import ComputeIOCollector  # noqa: F401

def CheckStatic(mod: IRModule | PrimFunc) -> bool:
    """CheckStatic

    Returns
    -------
    bool
        Whether the function is completely static.
    """
    if isinstance(mod, IRModule):
        items = mod.functions_items()
        assert len(items) == 1, "Temporarily only support single function module"
        return _ffi_api.CheckStatic(items[0][1])
    return _ffi_api.CheckStatic(mod)