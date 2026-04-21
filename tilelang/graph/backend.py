"""TileLang torch.compile backend entry point."""

import logging
from typing import Callable

import torch
from torch import fx

from tilelang import tvm as tvm
from tvm.target import Target

from tilelang.graph.converter import fx_to_relax
from tilelang.graph import cache as graph_cache
from tilelang.utils.target import determine_target

logger = logging.getLogger(__name__)


class _BackendConfig:
    """Mutable configuration for the ``"tilelang"`` torch.compile backend.

    Set attributes on the singleton :data:`tilelang.graph.backend_config`
    **before** calling ``torch.compile``::

        import tilelang
        from tilelang.graph import backend_config

        backend_config.extern_dispatch = my_dispatch
        compiled = torch.compile(model, backend="tilelang")
    """

    def __init__(self):
        self.extern_dispatch: Callable[..., bool] | None = None
        self.vm_clone_output: bool = True  # Clone VM outputs (False for benchmarking)
        self.use_cuda_graph: bool = False  # Capture static regions as CUDA graphs (WIP)

    def reset(self):
        """Restore all options to defaults."""
        self.extern_dispatch = None
        self.vm_clone_output = True
        self.use_cuda_graph = False


backend_config = _BackendConfig()


def _detect_target(example_inputs: list) -> Target:
    for inp in example_inputs:
        if isinstance(inp, torch.Tensor) and inp.device.type == "cuda":
            return determine_target("auto")
    return Target("llvm")


def tilelang_backend(
    gm: fx.GraphModule,
    example_inputs: list[torch.Tensor],
) -> Callable:
    """torch.compile backend that compiles FX graphs using TileLang.

    Converts FX graph → Relax IR → optimized TIR → Relax VM executable.
    Fallback ops (cuBLAS GEMM, SDPA, etc.) are dispatched via registered
    TVM packed functions at VM runtime.
    """
    dispatch = backend_config.extern_dispatch
    key = graph_cache.graph_cache_key(gm, example_inputs)

    # In-memory cache
    cached = graph_cache.get_memory_cached(key)
    if cached is not None:
        return cached

    # Cold compile
    try:
        target = _detect_target(example_inputs)
        tensor_inputs = [inp for inp in example_inputs if isinstance(inp, torch.Tensor)]
        relax_mod, fallback_calls = fx_to_relax(gm, tensor_inputs, extern_dispatch=dispatch)

        from tilelang.graph.vm_build import build_vm_runner
        wrapper = build_vm_runner(relax_mod, target, fallback_calls=fallback_calls,
                                 clone_output=backend_config.vm_clone_output,
                                 use_cuda_graph=backend_config.use_cuda_graph)
    except Exception:
        logger.error("tilelang_backend compilation failed", exc_info=True)
        raise

    graph_cache.put_memory_cached(key, wrapper)
    return wrapper
