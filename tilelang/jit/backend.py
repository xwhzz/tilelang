"""Compatibility shim for the TileLang torch.compile backend."""

from tilelang.jit.torch_compile.api import (  # noqa: F401
    clear_compilation_traces,
    clear_disk_cache,
    clear_subgraph_cache,
    get_compilation_traces,
    register_backend,
    tilelang_backend,
)

__all__ = [
    "clear_compilation_traces",
    "clear_disk_cache",
    "clear_subgraph_cache",
    "get_compilation_traces",
    "register_backend",
    "tilelang_backend",
]

