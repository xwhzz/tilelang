"""Standalone ``torch.sum`` pattern with built-in fp32 accumulator.

Matches plain ``R.sum(x, axis=-1, keepdims=False)`` on any float dtype
and emits a single te.compute kernel that:

  1. Reads the original fp16/bf16/fp32 input.
  2. Upcasts to fp32 inside the kernel for a numerically-stable
     reduction accumulator.
  3. Writes the output in the original dtype.

The fp32 upcast happens inside the pattern builder — NOT in user code.
Users write ``torch.sum(x, dim=-1)`` the natural way; the emitted kernel
behaves equivalently to ``x.float().sum(dim=-1).to(orig_dtype)`` but
without the intermediate fp32 buffer in DRAM.

Supports only last-axis, non-keepdim reductions today.  Non-last axes
or ``keepdims=True`` fall through to the generic schedule path.
"""

from tvm.relax.dpl.pattern import wildcard, is_op
from tvm import tir, te

from tilelang.graph.pattern_rewrite import register_pattern, InputInfo
from tilelang.graph.utils import get_static_shape


# ---------------------------------------------------------------------------
# Pattern
# ---------------------------------------------------------------------------

def _sum_pattern():
    """Match a plain ``sum`` call in Relax IR."""
    x = wildcard()
    out = is_op("relax.sum")(x)
    return out, {"x": x}


# ---------------------------------------------------------------------------
# Check
# ---------------------------------------------------------------------------

def _sum_check(matched_bindings, annotations):
    """Require last-axis, non-keepdim reduction on a float dtype.

    The internal fp32 upcast is added by the builder regardless of input
    dtype, so bf16/fp16/fp32 inputs all go through the same code path.
    """
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

    # Validate the sum call: last-axis only, keepdims=False only.
    sum_axis_ok = False
    for _, bnd in matched_bindings.items():
        val = bnd.value
        if not (isinstance(val, tvm.relax.Call) and hasattr(val.op, "name")):
            continue
        if val.op.name != "relax.sum":
            continue

        # Keepdims must be False — a (M, 1) output with keepdims=True
        # would need a different te.compute shape contract.
        keepdims = bool(val.attrs.keepdims) if hasattr(val.attrs, "keepdims") else False
        if keepdims:
            return None

        # Axis: accept only a single axis and only the last one.  A
        # scalar ``axis=-1`` is encoded as ``tvm.ir.Array([-1])`` in
        # relax attrs, so normalise to a Python list of ints first.
        axis_attr = val.attrs.axis if hasattr(val.attrs, "axis") else None
        if axis_attr is None:
            return None
        axis_list = list(axis_attr) if hasattr(axis_attr, "__iter__") else [axis_attr]
        axis_list = [int(a) for a in axis_list]
        if len(axis_list) != 1:
            return None

        rank = len(x_shape)
        norm_axis = axis_list[0] if axis_list[0] >= 0 else rank + axis_list[0]
        if norm_axis != rank - 1:
            return None

        sum_axis_ok = True
        break

    if not sum_axis_ok:
        return None

    return {
        "x_shape": x_shape,
        "_opaque": True,
    }


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------

def _sum_builder(inputs: dict[str, InputInfo], params: dict):
    """Build a fused cast→sum→cast kernel via te.compute.

    Layout (flattened over leading dims M; reduction over the last dim N):

        x        :  (M, N)   in input dtype (fp16/bf16/fp32)
        x_fp32   :  (M, N)   cast to fp32                    — inlined
        sum_val  :  (M,)     row-wise sum in fp32            — reduction
        out      :  (M,)     sum_val cast back to input dtype
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

    # te.sum must sit at the top level of a te.compute body, so we cannot
    # fuse the cast-back into the reduction's lambda.  Use two blocks:
    # (1) reduce in fp32, (2) cast back to the input dtype in a trailing
    # injective compute.  GeneralReduction will compute_at / inline the
    # trailing cast onto the reduction's spatial loop.
    rk = te.reduce_axis((0, N), name="rk")
    sum_val = te.compute(
        (M,),
        lambda i: te.sum(x[i, rk].astype("float32"), axis=rk),
        name="sum_val",
    )
    out = te.compute(
        (M,),
        lambda i: sum_val[i].astype(in_dtype),
        name="sum_out",
    )

    return te.create_prim_func([x, out])


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

register_pattern(
    "sum_fp32_accum",
    _sum_pattern,
    _sum_builder,
    check_fn=_sum_check,
)
