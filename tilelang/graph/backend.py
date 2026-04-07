"""TileLang torch.compile backend entry point."""

import logging
import warnings
from typing import Callable

import torch
from torch import fx

from tilelang import tvm as tvm
from tvm.target import Target

from tilelang.graph.converter import fx_to_relax, extract_fallback_calls
from tilelang.graph import cache as graph_cache
from tilelang.utils.target import determine_target

logger = logging.getLogger(__name__)


class _BackendConfig:
    """Mutable configuration for the ``"tilelang"`` torch.compile backend.

    Set attributes on the singleton :data:`tilelang.graph.backend_config`
    **before** calling ``torch.compile``::

        import tilelang
        from tilelang.graph import backend_config, default_extern_dispatch

        # Example: raise the cuBLAS threshold so more matmuls go through TileLang
        def my_dispatch(node):
            return default_extern_dispatch(node, gemm_threshold=256)

        backend_config.extern_dispatch = my_dispatch

        compiled = torch.compile(model, backend="tilelang")
    """

    def __init__(self):
        self.extern_dispatch: Callable[..., bool] | None = None
        self.use_vm: bool = True  # Relax VM runtime (default; set False for legacy codegen)
        self.vm_clone_output: bool = True  # Clone VM outputs (False for benchmarking)

    def reset(self):
        """Restore all options to defaults."""
        self.extern_dispatch = None
        self.use_vm = True
        self.vm_clone_output = True


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
    """torch.compile backend that compiles FX graphs using TileLang."""
    dispatch = backend_config.extern_dispatch
    key = graph_cache.graph_cache_key(gm, example_inputs)

    # Level 1: in-memory
    cached = graph_cache.get_memory_cached(key)
    if cached is not None:
        return cached

    # Level 2: disk (legacy codegen path only)
    if not backend_config.use_vm:
        optimized_mod = graph_cache.load_relax_mod(key)
        if optimized_mod is not None:
            target = _detect_target(example_inputs)
            from tilelang.graph.compiler import compile_tir_functions
            from tilelang.graph.codegen import generate_wrapper
            compiled_kernels = compile_tir_functions(optimized_mod, target)
            fallback_calls = extract_fallback_calls(gm, extern_dispatch=dispatch)
            wrapper = generate_wrapper(optimized_mod, compiled_kernels, fallback_calls)
            graph_cache.put_memory_cached(key, wrapper)
            return wrapper

    # Level 3: cold compile
    target = _detect_target(example_inputs)
    tensor_inputs = [inp for inp in example_inputs if isinstance(inp, torch.Tensor)]
    relax_mod, fallback_calls = fx_to_relax(gm, tensor_inputs, extern_dispatch=dispatch)

    if backend_config.use_vm:
        from tilelang.graph.vm_build import build_vm_runner
        wrapper = build_vm_runner(relax_mod, target, fallback_calls=fallback_calls,
                                 clone_output=backend_config.vm_clone_output)
    else:
        warnings.warn(
            "The Python/C wrapper codegen path is deprecated. "
            "The Relax VM path (use_vm=True) is now the default.",
            DeprecationWarning, stacklevel=2)
        from tilelang.graph.pipeline import run_pipeline
        from tilelang.graph.compiler import compile_tir_functions
        from tilelang.graph.codegen import generate_wrapper
        optimized_mod = run_pipeline(relax_mod, target)
        compiled_kernels = compile_tir_functions(optimized_mod, target)
        wrapper = generate_wrapper(optimized_mod, compiled_kernels, fallback_calls)
        graph_cache.save_relax_mod(key, optimized_mod)

    graph_cache.put_memory_cached(key, wrapper)
    return wrapper
