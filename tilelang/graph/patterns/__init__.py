"""Built-in pattern rewriters for common LLM kernels.

Importing this module registers all built-in patterns with the
PatternRewritePass framework. Users can register additional patterns
via ``register_pattern()`` from their own code.

Active patterns (registration order = matching priority):
  - residual_rmsnorm: add(hidden, residual) + RMSNorm → fused dual-output kernel
  - rmsnorm: standalone RMSNorm → fused te.compute kernel
  - fused_rope: reshape + permute + RoPE → single te.compute
  - reshape_permute: (B,S,H*D) → reshape → permute → (B,H,S,D)
  - permute_reshape: (B,H,S,D) → permute → reshape → (B,S,H*D)
"""

from . import residual_rmsnorm  # noqa: F401
from . import rmsnorm  # noqa: F401
from . import fused_rope  # noqa: F401
from . import reshape_transpose  # noqa: F401
