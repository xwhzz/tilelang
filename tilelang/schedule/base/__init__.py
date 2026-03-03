# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Base infra"""
from tvm.dlight import (
    BlockInfo,  # noqa: F401
    IterInfo,  # noqa: F401
    normalize_prim_func,  # noqa: F401
    try_inline,  # noqa: F401
    try_inline_contiguous_spatial,  # noqa: F401
    ScheduleRule,  # noqa: F401
)  # noqa: F401
from tvm.dlight.analysis import (
    collect_block_iter_vars_used_in_access_region,  # noqa: F401
    collect_vars_used_in_prim_expr,  # noqa: F401
    detect_dominant_read,  # noqa: F401
    is_broadcast_epilogue,  # noqa: F401
)

from .common_schedules import get_block, get_output_blocks

