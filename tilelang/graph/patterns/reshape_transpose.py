"""Fused reshape+transpose patterns for attention head layout conversion.

Pattern 1 (pre-attention): (B,S,H*D) → reshape → permute → (B,H,S,D)
Pattern 2 (post-attention): (B,H,S,D) → permute → reshape → (B,S,H*D)

Replaces with te.compute that does the reorder in one pass.
"""

from tvm.relax.dpl.pattern import wildcard, is_op
from tvm import tir, te, relax

from tilelang.graph.pattern_rewrite import register_pattern, InputInfo


# ---------------------------------------------------------------------------
# Pattern 1: reshape + permute (e.g. V projection: (B,S,H*D) → (B,H,S,D))
# ---------------------------------------------------------------------------

def _reshape_permute_pattern():
    x = wildcard()
    x_reshaped = is_op("relax.reshape")(x, wildcard())
    out = is_op("relax.permute_dims")(x_reshaped)
    return out, {"x": x}


def _reshape_permute_check(matched_bindings, annotations):
    x_var = annotations.get("x")
    if x_var is None:
        return None
    x_si = x_var.struct_info_ if hasattr(x_var, "struct_info_") else None
    if not isinstance(x_si, relax.TensorStructInfo) or x_si.shape is None:
        return None
    x_shape = [int(s) for s in x_si.shape.values if isinstance(s, tir.IntImm)]
    if len(x_shape) != len(x_si.shape.values) or len(x_shape) != 3:
        return None

    # Find the permute_dims output to get the target shape
    for name, b in matched_bindings.items():
        val = b.value
        if isinstance(val, relax.Call) and hasattr(val.op, "name") and "permute_dims" in val.op.name:
            si = b.var.struct_info_
            if isinstance(si, relax.TensorStructInfo) and si.shape:
                out_shape = [int(s) for s in si.shape.values if isinstance(s, tir.IntImm)]
                if len(out_shape) == 4:
                    B, H, S, D = out_shape
                    if x_shape == [B, S, H * D]:
                        return {"B": B, "H": H, "S": S, "D": D, "out_shape": out_shape}
    return None


def _reshape_permute_builder(inputs, params):
    """(B,S,H*D) → (B,H,S,D) via index remapping."""
    x_info = inputs["x"]
    B, H, S, D = params["B"], params["H"], params["S"], params["D"]
    dtype = x_info.dtype

    x = te.placeholder(tuple(x_info.shape), name="x", dtype=dtype)
    out = te.compute(
        (B, H, S, D),
        lambda b, h, s, d: x[b, s, h * D + d],
        name="reshape_permute_out")
    return te.create_prim_func([x, out])


# ---------------------------------------------------------------------------
# Pattern 2: permute + reshape (post-attention: (B,H,S,D) → (B,S,H*D))
# ---------------------------------------------------------------------------

def _permute_reshape_pattern():
    x = wildcard()
    x_permuted = is_op("relax.permute_dims")(x)
    out = is_op("relax.reshape")(x_permuted, wildcard())
    return out, {"x": x}


def _permute_reshape_check(matched_bindings, annotations):
    x_var = annotations.get("x")
    if x_var is None:
        return None
    x_si = x_var.struct_info_ if hasattr(x_var, "struct_info_") else None
    if not isinstance(x_si, relax.TensorStructInfo) or x_si.shape is None:
        return None
    x_shape = [int(s) for s in x_si.shape.values if isinstance(s, tir.IntImm)]
    if len(x_shape) != len(x_si.shape.values) or len(x_shape) != 4:
        return None

    # Find reshape output to get the target shape
    for name, b in matched_bindings.items():
        val = b.value
        if isinstance(val, relax.Call) and hasattr(val.op, "name") and "reshape" in val.op.name:
            si = b.var.struct_info_
            if isinstance(si, relax.TensorStructInfo) and si.shape:
                out_shape = [int(s) for s in si.shape.values if isinstance(s, tir.IntImm)]
                if len(out_shape) == 3:
                    B, H, S, D = x_shape
                    if out_shape == [B, S, H * D]:
                        return {"B": B, "H": H, "S": S, "D": D, "out_shape": out_shape}
    return None


def _permute_reshape_builder(inputs, params):
    """(B,H,S,D) → (B,S,H*D) via index remapping."""
    x_info = inputs["x"]
    B, H, S, D = params["B"], params["H"], params["S"], params["D"]
    dtype = x_info.dtype

    x = te.placeholder(tuple(x_info.shape), name="x", dtype=dtype)
    out = te.compute(
        (B, S, H * D),
        lambda b, s, hd: x[b, hd // D, s, hd % D],
        name="permute_reshape_out")
    return te.create_prim_func([x, out])


# ── Register ──
register_pattern("reshape_permute", _reshape_permute_pattern,
                 _reshape_permute_builder, check_fn=_reshape_permute_check)
register_pattern("permute_reshape", _permute_reshape_pattern,
                 _permute_reshape_builder, check_fn=_permute_reshape_check)
