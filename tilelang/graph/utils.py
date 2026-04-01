"""Shared utilities for the TileLang graph compiler."""

import torch
from tilelang.utils.tensor import map_torch_type


# Reuse existing dtype maps from tilelang.language.dtypes
def torch_dtype_to_tvm(dtype: torch.dtype) -> str:
    """Convert a PyTorch dtype to a TVM dtype string."""
    from tilelang.language.dtypes import _TORCH_DTYPE_TO_STR
    if dtype not in _TORCH_DTYPE_TO_STR:
        raise ValueError(f"Unsupported torch dtype: {dtype}")
    return _TORCH_DTYPE_TO_STR[dtype]


def tvm_dtype_to_torch(dtype_str: str) -> torch.dtype:
    """Convert a TVM dtype string to a PyTorch dtype."""
    return map_torch_type(dtype_str)
