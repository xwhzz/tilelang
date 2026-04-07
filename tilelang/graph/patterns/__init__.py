"""Built-in pattern rewriters for common LLM kernels.

Importing this module registers all built-in patterns with the framework.
Users can register additional patterns from their own code.
"""

# Each module registers its pattern via register_pattern() on import
# Order: longer/more specific patterns first
from . import residual_rmsnorm  # noqa: F401
from . import fused_rope  # noqa: F401
from . import reshape_transpose  # noqa: F401
# rmsnorm: pattern doesn't match current HF LLaMA IR (diamond on x_cast).
# residual_rmsnorm subsumes it for the add+RMSNorm case; the first
# standalone RMSNorm (on embedding) falls through to FuseOps.
# from . import rmsnorm  # noqa: F401
# Old flat-1D RoPE (superseded by fused_rope which uses te.compute + 4D)
# from . import rope   # noqa: F401
