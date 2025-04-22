# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""The language interface for tl programs."""

from tilelang import language as T
from tvm.tir import Buffer, BufferRegion
from tvm.ir import Range
from tvm import tir
from typing import Union
from tilelang.utils.language import get_buffer_elems


def any_of(buffer: Union[T.Tensor, BufferRegion]):
    """Check if any element in the buffer is true.
    
    Args:
        buffer: Either a TVM buffer or buffer region to be checked
    
    Returns:
        A TVM intrinsic call that performs the any operation
    """
    return_type: str = "bool"
    if isinstance(buffer, Buffer):
        elems = get_buffer_elems(buffer)
        return T.call_intrin(return_type, tir.op.Op.get("tl.any_of"), T.address_of(buffer), elems)
    elif isinstance(buffer, BufferRegion):
        buffer, region = buffer.buffer, buffer.region
        new_region = []
        extent = 1
        for i, r in enumerate(region):
            extent = r.extent
            if extent == 1:
                new_region.append(r)
            else:
                # check the idx is the last dimension
                if i != len(region) - 1:
                    raise ValueError(
                        "Only support the last dimension to be for T.any currently, please contact us if you need this feature"
                    )
                new_region.append(Range(r.min, 1))
        buffer = BufferRegion(buffer, new_region)
        return T.call_intrin(return_type, tir.op.Op.get("tl.any_of"), T.address_of(buffer), extent)
    else:
        raise ValueError(f"Invalid buffer type: {type(buffer)}")


def all_of(buffer: Union[T.Tensor, BufferRegion]):
    """Check if all elements in the buffer are true.
    
    Args:
        buffer: Either a TVM buffer or buffer region to be checked
    
    Returns:
        A TVM intrinsic call that performs the any operation
    """
    return_type: str = "bool"
    if isinstance(buffer, Buffer):
        elems = get_buffer_elems(buffer)
        return T.call_intrin(return_type, tir.op.Op.get("tl.all_of"), T.address_of(buffer), elems)
    elif isinstance(buffer, BufferRegion):
        buffer, region = buffer.buffer, buffer.region
        new_region = []
        extent = 1
        for i, r in enumerate(region):
            extent = r.extent
            if extent == 1:
                new_region.append(r)
            else:
                # check the idx is the last dimension
                if i != len(region) - 1:
                    raise ValueError(
                        "Only support the last dimension to be for T.any currently, please contact us if you need this feature"
                    )
                new_region.append(Range(r.min, 1))
        buffer = BufferRegion(buffer, new_region)
        return T.call_intrin(return_type, tir.op.Op.get("tl.all_of"), T.address_of(buffer), extent)
    else:
        raise ValueError(f"Invalid buffer type: {type(buffer)}")
