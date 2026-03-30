"""Compatibility shim for torch.compile graph runtime."""

from tilelang.jit.torch_compile.runtime import CompiledGraphModule  # noqa: F401

__all__ = ["CompiledGraphModule"]
