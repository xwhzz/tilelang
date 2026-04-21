"""Standalone Softmax pattern with built-in fp32 up-cast.

Matches plain ``R.nn.softmax(x, axis=-1)`` on any float dtype and emits a
single te.compute kernel that:

  1. Reads the original fp16/bf16 input.
  2. Upcasts to fp32 inside the kernel for numerically-stable max/exp/sum.
  3. Writes the output in the original dtype.

The up-cast happens inside the pattern builder, NOT in user-visible torch
code.  Users write ``F.softmax(x, dim=-1)`` the natural way — same
semantics as PyTorch's own ``F.softmax`` which also upcasts internally
for fp16/bf16.
"""

from tvm.relax.dpl.pattern import wildcard, is_op
from tvm import tir, te

from tilelang.graph.pattern_rewrite import register_pattern, InputInfo
from tilelang.graph.utils import get_static_shape


# ---------------------------------------------------------------------------
# Pattern
# ---------------------------------------------------------------------------

def _softmax_pattern():
    """Match a plain ``softmax`` call in Relax IR."""
    x = wildcard()
    out = is_op("relax.nn.softmax")(x)
    return out, {"x": x}


# ---------------------------------------------------------------------------
# Check
# ---------------------------------------------------------------------------

def _softmax_check(matched_bindings, annotations):
    """Require last-axis softmax on a float dtype; internal upcast handled
    by the builder."""
    x_var = annotations.get("x")
    if x_var is None:
        return None

    x_shape = get_static_shape(x_var)
    if x_shape is None or len(x_shape) < 1:
        return None

    import tvm
    x_dtype = None
    if hasattr(x_var, "struct_info") and hasattr(x_var.struct_info, "dtype"):
        x_dtype = x_var.struct_info.dtype
    if x_dtype not in ("float16", "bfloat16", "float32"):
        return None

    # Softmax must reduce the last dim — that's what the builder generates.
    for _, bnd in matched_bindings.items():
        val = bnd.value
        if not (isinstance(val, tvm.relax.Call) and hasattr(val.op, "name")):
            continue
        if val.op.name != "relax.nn.softmax":
            continue
        axis = int(val.attrs.axis) if hasattr(val.attrs, "axis") else -1
        rank = len(x_shape)
        norm_axis = axis if axis >= 0 else rank + axis
        if norm_axis != rank - 1:
            return None
        break

    return {
        "x_shape": x_shape,
        "_opaque": True,
    }


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------

def _softmax_builder(inputs: dict[str, InputInfo], params: dict):
    """Build a fused cast→softmax→cast kernel via te.compute.

    Layout (flattened over leading dims M and reduction over the last dim N):

        x        :  (M, N)  in input dtype (bf16/fp16)
        x_fp32   :  (M, N)  cast to fp32                  — inlined
        maxelem  :  (M,)    max over N                    — reduction 1
        exp_val  :  (M, N)  exp(x_fp32 - maxelem)         — bridge
        expsum   :  (M,)    sum over N of exp_val         — reduction 2
        out      :  (M, N)  (exp_val / expsum).astype(in_dtype)

    This mirrors softmax's standard two-pass structure (max then sum) while
    keeping the entire fp32 intermediate inside one te.compute group, so
    the graph backend emits a single fused kernel and no fp32 buffer is
    materialised in DRAM.
    """
    x_info = inputs["x"]
    x_shape = tuple(x_info.shape)
    in_dtype = x_info.dtype

    # Flatten all leading dims into M, reduce over the last dim N.
    M = 1
    for s in x_shape[:-1]:
        M *= s
    N = x_shape[-1]

    x = te.placeholder((M, N), name="x", dtype=in_dtype)

    # Cast to fp32 for numerically-stable max/exp/sum.
    x_fp32 = te.compute(
        (M, N),
        lambda i, j: x[i, j].astype("float32"),
        name="x_fp32",
    )

    # First reduction: row-wise max for numerical stability.
    rk1 = te.reduce_axis((0, N), name="rk1")
    maxelem = te.compute(
        (M,),
        lambda i: te.max(x_fp32[i, rk1], axis=rk1),
        name="softmax_maxelem",
    )

    # Bridge: exp(x - max) — full-row fp32 intermediate, fed to both
    # the sum reduction and the output.  This is the softmax analogue of
    # LayerNorm's `diff` bridge.
    exp_val = te.compute(
        (M, N),
        lambda i, j: tir.exp(x_fp32[i, j] - maxelem[i]),
        name="softmax_exp",
    )

    # Second reduction: row-wise sum of exp values.
    rk2 = te.reduce_axis((0, N), name="rk2")
    expsum = te.compute(
        (M,),
        lambda i: te.sum(exp_val[i, rk2], axis=rk2),
        name="softmax_expsum",
    )

    # Output: normalize and cast back to input dtype.
    out = te.compute(
        (M, N),
        lambda i, j: (exp_val[i, j] / expsum[i]).astype(in_dtype),
        name="softmax_out",
    )

    return te.create_prim_func([x, out])


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

register_pattern(
    "softmax_fp32_roundtrip",
    _softmax_pattern,
    _softmax_builder,
    check_fn=_softmax_check,
)
