# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# pylint: disable=missing-docstring
"""Utility methods for generic GPU."""

from __future__ import annotations


from tilelang import tvm

from tvm import tir
from tvm.target import Target


def max_threads_per_block(target: Target) -> int:
    """Get the maximum number of threads per block for a given target.

    Parameters
    ----------
    target : Target
        The target to get the maximum number of threads per block for.

    Returns
    -------
    max_threads_per_block : int
        The maximum number of threads per block for the given target.
    """
    for name in ["max_threads_per_block", "max_num_threads"]:
        result = target.attrs.get(name, None)
        if result is not None:
            return result
    if target.kind.name == "cuda":
        return 1024
    return 256

def get_sm_version(target: Target) -> int:
    if target.kind.name != "cuda":
        return -1
    arch = target.arch
    if not arch.startswith("sm_"):
        return -1
    suffix = arch[len("sm_") :]
    digits: list[str] = []
    for char in suffix:
        if not char.isdigit():
            break
        digits.append(char)
    if not digits:
        return -1
    return int("".join(digits))
