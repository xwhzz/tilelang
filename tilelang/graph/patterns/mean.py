"""Standalone ``torch.mean`` pattern with built-in fp32 accumulator.

Matches plain ``R.mean(x, axis=-1, keepdims=False)`` on any float dtype
and emits a single kernel that:

  1. Reads the original fp16/bf16/fp32 input.
  2. Upcasts to fp32 inside the kernel for a numerically-stable
     reduction accumulator.
  3. Divides by N in fp32.
  4. Writes the output in the original dtype.

Without this pattern, legalization of ``R.mean`` produces a reduction
with an fp16/bf16 accumulator — which loses precision for larger
magnitudes.  This pattern forces fp32 accumulation inside the kernel
without materializing an fp32 input buffer in DRAM.

Supports only last-axis, non-keepdim reductions today.  Non-last axes
or ``keepdims=True`` fall through to the generic schedule path.
"""

from tvm.relax.dpl.pattern import wildcard, is_op
from tvm import te

from tilelang.graph.pattern_rewrite import register_pattern, InputInfo
from tilelang.graph.utils import get_static_shape


# ---------------------------------------------------------------------------
# Pattern
# ---------------------------------------------------------------------------

def _mean_pattern():
    """Match a plain ``mean`` call in Relax IR."""
    x = wildcard()
    out = is_op("relax.mean")(x)
    return out, {"x": x}


# ---------------------------------------------------------------------------
# Check
# ---------------------------------------------------------------------------

def _mean_check(matched_bindings, annotations):
    """Require last-axis, non-keepdim reduction on a float dtype."""
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

    # Validate the mean call: last-axis only, keepdims=False only.
    mean_axis_ok = False
    for _, bnd in matched_bindings.items():
        val = bnd.value
        if not (isinstance(val, tvm.relax.Call) and hasattr(val.op, "name")):
            continue
        if val.op.name != "relax.mean":
            continue

        keepdims = bool(val.attrs.keepdims) if hasattr(val.attrs, "keepdims") else False
        if keepdims:
            return None

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

        mean_axis_ok = True
        break

    if not mean_axis_ok:
        return None

    return {
        "x_shape": x_shape,
        "_opaque": True,
    }


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------

def _mean_builder(inputs: dict[str, InputInfo], params: dict):
    """Build a fused cast→sum→divide→cast kernel via te.compute.

    Layout (flattened over leading dims M; reduction over the last dim N):

        x        :  (M, N)   in input dtype (fp16/bf16/fp32)
        sum_val  :  (M,)     row-wise sum in fp32            — reduction
        out      :  (M,)     sum_val / N cast back to input dtype
    """
    x_info = inputs["x"]
    x_shape = tuple(x_info.shape)
    in_dtype = x_info.dtype

    M = 1
    for s in x_shape[:-1]:
        M *= s
    N = x_shape[-1]
    inv_n = 1.0 / float(N)

    x = te.placeholder((M, N), name="x", dtype=in_dtype)

    # Two-block form: (1) fp32 reduction, (2) divide + cast-back.
    # Using a single te.compute with te.sum nested inside a divide does
    # not work — reductions must sit at the top level of a compute body.
    rk = te.reduce_axis((0, N), name="rk")
    sum_val = te.compute(
        (M,),
        lambda i: te.sum(x[i, rk].astype("float32"), axis=rk),
        name="sum_val",
    )
    out = te.compute(
        (M,),
        lambda i: (sum_val[i] * inv_n).astype(in_dtype),
        name="mean_out",
    )

    return te.create_prim_func([x, out])


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

register_pattern(
    "mean_fp32_accum",
    _mean_pattern,
    _mean_builder,
    check_fn=_mean_check,
)
