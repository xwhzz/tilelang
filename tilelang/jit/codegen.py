"""Compatibility shim for torch.compile wrapper codegen."""

from tilelang.jit.torch_compile.codegen import (  # noqa: F401
    WrapperCodeGen,
    _match_fallback_pattern,
)

__all__ = ["WrapperCodeGen", "_match_fallback_pattern"]

