"""Compatibility shim for graph compiler internals."""

from tilelang.jit.torch_compile.analysis import (  # noqa: F401
    FXLoweringResult,
    GraphCompileTrace,
    KernelTrace,
    _ExternOpInfo,
    _TIRCallRecord,
    _lower_primfunc_for_tilelang,
    _resolve_arch,
    _schedule_relax_module,
    compile_subgraph_direct,
    from_fx_with_fallback,
)

__all__ = [
    "FXLoweringResult",
    "GraphCompileTrace",
    "KernelTrace",
    "_ExternOpInfo",
    "_TIRCallRecord",
    "_lower_primfunc_for_tilelang",
    "_resolve_arch",
    "_schedule_relax_module",
    "compile_subgraph_direct",
    "from_fx_with_fallback",
]

