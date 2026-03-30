"""TileLang torch.compile backend.

This package owns the graph compiler used by
``torch.compile(..., backend="tilelang")``.
"""

from .api import (  # noqa: F401
    clear_compilation_traces,
    clear_disk_cache,
    clear_subgraph_cache,
    get_compilation_traces,
    register_backend,
    tilelang_backend,
)

