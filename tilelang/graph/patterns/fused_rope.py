"""Fused RoPE pattern: reshape + permute + rotary position embedding.

Matches (before LegalizeOps):
    x (B,S,H*D) → reshape (B,S,H,D) → permute (B,H,S,D) → RoPE → out (B,H,S,D)

Replaces with a single te.compute that reads (B,S,H*D) and writes
(B,H,S,D) with RoPE applied — reshape+permute+RoPE in one pass.

The permuted tensor q feeds multiply + 2 strided_slices (diamond).
We mark q as an annotation so the externally-used check allows the
strided_slices (which are dead code after rewrite).
"""

from tvm.relax.dpl.pattern import wildcard, is_op
from tvm import tir, te

from tilelang.graph.pattern_rewrite import register_pattern, InputInfo
from tilelang.graph.utils import get_static_shape


def _fused_rope_pattern():
    """Match reshape → permute → RoPE chain."""
    x = wildcard()       # pre-reshape: (B, S, H*D) from linear
    cos = wildcard()     # expanded cos
    sin = wildcard()     # expanded sin

    # reshape + permute → q
    x_reshaped = is_op("relax.reshape")(x, wildcard())
    q = is_op("relax.permute_dims")(x_reshaped)

    # RoPE on q
    q_cos = is_op("relax.multiply")(q, cos)
    q2 = is_op("relax.strided_slice")(q, wildcard(), wildcard(), wildcard(), wildcard())
    q2r = is_op("relax.reshape")(q2, wildcard())
    neg = is_op("relax.negative")(q2r)
    rotated = is_op("relax.concat")(wildcard())
    rot_sin = is_op("relax.multiply")(rotated, sin)
    out = is_op("relax.add")(q_cos, rot_sin)

    # q is annotation → the strided_slices on q (for rotate_half) are
    # allowed to be "external" uses. They become dead code after rewrite.
    return out, {"x": x, "q": q, "cos": cos, "sin": sin}


def _fused_rope_check(matched_bindings, annotations):
    """Validate shapes and extract H, D from x and cos."""
    from tvm import relax

    x_var = annotations.get("x")
    cos_var = annotations.get("cos")
    if x_var is None or cos_var is None:
        return None

    x_shape = get_static_shape(x_var)
    cos_shape = get_static_shape(cos_var)
    if x_shape is None or cos_shape is None:
        return None
    if len(x_shape) < 2:
        return None

    head_dim = cos_shape[-1]
    hidden = x_shape[-1]
    if hidden % head_dim != 0:
        return None
    num_heads = hidden // head_dim
    half_dim = head_dim // 2
    if half_dim == 0 or head_dim % 2 != 0:
        return None

    return {
        "num_heads": num_heads,
        "head_dim": head_dim,
        "half_dim": half_dim,
        "x_shape": x_shape,
        "cos_shape": cos_shape,
        "_opaque": True,  # prevent FuseOps merge (q annotation causes FuseTIR conflict)
    }


def _fused_rope_builder(inputs: dict[str, InputInfo], params: dict):
    """Build te.compute: (B,S,H*D) → reshape+permute+RoPE → (B,H,S,D)."""
    x_info = inputs["x"]
    cos_info = inputs["cos"]
    sin_info = inputs["sin"]

    x_shape = tuple(x_info.shape)       # (B, S, H*D)
    cos_shape = tuple(cos_info.shape)
    H = params["num_heads"]
    D = params["head_dim"]
    half = params["half_dim"]
    dtype = x_info.dtype

    B = x_shape[0]
    S = x_shape[1]
    out_shape = (B, H, S, D)

    x = te.placeholder(x_shape, name="x", dtype=dtype)
    cos_t = te.placeholder(cos_shape, name="cos", dtype=cos_info.dtype)
    sin_t = te.placeholder(cos_shape, name="sin", dtype=sin_info.dtype)

    def _rope(b, h, s, d):
        # Fused reshape+permute: x[b, s, h*D + d]
        x_val = x[b, s, h * D + d].astype("float32")
        paired_d = (d + half) % D
        x_paired = x[b, s, h * D + paired_d].astype("float32")
        sign = tir.const(1, "float32") - tir.const(2, "float32") * (
            d < tir.const(half, d.dtype)).astype("float32")

        if len(cos_shape) == 4:
            c = cos_t[0, 0, s, d].astype("float32")
            sn = sin_t[0, 0, s, d].astype("float32")
        elif len(cos_shape) == 3:
            c = cos_t[0, s, d].astype("float32")
            sn = sin_t[0, s, d].astype("float32")
        else:
            c = cos_t[s, d].astype("float32")
            sn = sin_t[s, d].astype("float32")

        return (x_val * c + sign * x_paired * sn).astype(dtype)

    out = te.compute(out_shape, _rope, name="rope_out")
    return te.create_prim_func([x, cos_t, sin_t, out])


register_pattern("fused_rope", _fused_rope_pattern, _fused_rope_builder,
                 check_fn=_fused_rope_check)
