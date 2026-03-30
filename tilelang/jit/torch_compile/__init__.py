"""TileLang torch.compile backend internals.

This package owns the graph compiler used by
``torch.compile(..., backend="tilelang")``.

The root ``tilelang.jit`` package only keeps thin compatibility shims so the
graph compiler implementation does not sprawl across unrelated JIT modules.
"""

from .api import (  # noqa: F401
    clear_compilation_traces,
    clear_disk_cache,
    clear_subgraph_cache,
    get_compilation_traces,
    register_backend,
    tilelang_backend,
)

