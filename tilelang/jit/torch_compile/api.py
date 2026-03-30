"""Public API for the TileLang torch.compile backend."""

from __future__ import annotations

import logging
import os
import shutil
import threading
from dataclasses import dataclass, field
from typing import Any, Callable

import torch

from .analysis import GraphCompileTrace
from .compiler import compile_subgraph

logger = logging.getLogger(__name__)


@dataclass
class BackendState:
    """Process-local state shared by the torch.compile backend."""

    _compilation_traces: list[GraphCompileTrace] = field(default_factory=list)
    _subgraph_cache: dict[str, Callable] = field(default_factory=dict)
    _trace_lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    _cache_lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def cache_dir(self) -> str:
        return os.path.join(
            os.environ.get(
                "TILELANG_CACHE_DIR",
                os.path.join(os.path.expanduser("~"), ".cache", "tilelang"),
            ),
            "graphs",
        )

    def disk_cache_enabled(self) -> bool:
        return os.environ.get("TILELANG_DISK_CACHE", "0") not in ("0", "false", "no")

    def add_trace(self, trace: GraphCompileTrace) -> None:
        with self._trace_lock:
            self._compilation_traces.append(trace)

    def get_traces(self) -> list[GraphCompileTrace]:
        with self._trace_lock:
            return list(self._compilation_traces)

    def clear_traces(self) -> None:
        with self._trace_lock:
            self._compilation_traces.clear()

    def cache_get(self, key: str) -> Callable | None:
        with self._cache_lock:
            return self._subgraph_cache.get(key)

    def cache_put(self, key: str, runner: Callable) -> None:
        with self._cache_lock:
            self._subgraph_cache[key] = runner

    def clear_cache(self) -> None:
        with self._cache_lock:
            self._subgraph_cache.clear()


_STATE = BackendState()
_REGISTERED = False


def tilelang_backend(
    gm: torch.fx.GraphModule,
    example_inputs: list[Any],
    *,
    options: dict[str, Any] | None = None,
) -> Callable:
    gm.graph.eliminate_dead_code()
    return compile_subgraph(_STATE, gm, example_inputs, options=options)


def get_compilation_traces() -> list[GraphCompileTrace]:
    return _STATE.get_traces()


def clear_compilation_traces() -> None:
    _STATE.clear_traces()


def clear_subgraph_cache() -> None:
    _STATE.clear_cache()


def clear_disk_cache() -> None:
    cache_dir = _STATE.cache_dir()
    if os.path.isdir(cache_dir):
        shutil.rmtree(cache_dir)
        logger.info("Disk cache cleared: %s", cache_dir)


def register_backend() -> None:
    global _REGISTERED
    if _REGISTERED:
        return
    try:
        from torch._dynamo.backends.registry import register_backend as register_dynamo_backend

        register_dynamo_backend(name="tilelang")(tilelang_backend)
        _REGISTERED = True
    except ImportError:
        logger.warning("torch._dynamo not available; 'tilelang' backend not registered.")


register_backend()
