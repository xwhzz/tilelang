# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""
GPU-generic schedule rules.
For CUDA/ROCm/Vulkan/Metal-specific rules, use `tvm.dlight.cuda/rocm/vulkan/metal` instead
"""

from .fallback import Fallback  # noqa: F401
from .element_wise import ElementWise  # noqa: F401
from .reduction import Reduction  # noqa: F401
from .general_reduction import GeneralReduction  # noqa: F401
from .layernorm_like import LayerNormLike  # noqa: F401
from .transpose import Transpose  # noqa: F401

from .gemv import GEMV  # noqa: F401
from .matmul import Matmul  # noqa: F401
# from .gemv_dequantize import GEMVWithDequantizeInfo  # noqa: F401
# from .matmul import (
#     Matmul,  # noqa: F401
#     MatmulTensorizationMMA,  # noqa: F401
#     MatmulTensorizationWMMA,  # noqa: F401
# )
# from .matmul_mma_dequantize import (
#     MatmulTensorizationMMAWithDequantizeInfo,  # noqa: F401
# )
# from .matmul_wmma import MatmulTensorizationLegacy  # noqa: F401

# from tvm.dlight.gpu import Reduction  # noqa: F401


def default_schedule_rules():
    """Return the default TileLang GPU rule order with generic fallback last."""
    return [
        Matmul(),
        GEMV(),
        Reduction(),
        GeneralReduction(),
        Transpose(),
        ElementWise(),
        Fallback(),
    ]
