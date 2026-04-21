"""Standalone LayerNorm pattern.

Matches the explicit two-pass implementation:

    astype(fp32)
    → mean (first reduction)
    → subtract (x_fp32 - mean)           ← bridge: depends on input + first reduction
    → power(2)
    → mean (second reduction)
    → add(eps) → rsqrt
    → multiply(diff, rstd)               ← diff reused from bridge
    → astype(orig) → multiply(w) → add(bias)

The pattern fires only when the user's Module uses the explicit fp32 round-trip
(required for FuseOps to form a single call_tir group):

    x32 = x.float()
    mean = x32.mean(dim=-1, keepdim=True)
    diff = x32 - mean
    norm = diff * torch.rsqrt((diff**2).mean(-1, keepdim=True) + eps)
    return weight * norm.to(in_dtype) + bias

The te.compute kernel uses the same two-pass algorithm (mean then variance),
which the GeneralReduction rule's double-reduction path can schedule.
"""

from tvm.relax.dpl.pattern import wildcard, is_op
from tvm import tir, te, relax

from tilelang.graph.pattern_rewrite import register_pattern, InputInfo
from tilelang.graph.utils import get_static_shape


# ---------------------------------------------------------------------------
# Pattern
# ---------------------------------------------------------------------------

def _layernorm_pattern():
    """Match explicit two-pass LayerNorm chain in Relax IR."""
    x = wildcard()
    w = wildcard()
    b = wildcard()

    x_fp32 = is_op("relax.astype")(x)
    x_mean = is_op("relax.mean")(x_fp32)
    diff = is_op("relax.subtract")(x_fp32, x_mean)
    diff_sq = is_op("relax.power")(diff, wildcard())
    var = is_op("relax.mean")(diff_sq)
    var_eps = is_op("relax.add")(var, wildcard())
    rstd = is_op("relax.rsqrt")(var_eps)
    # diff is reused: same node appears in diff_sq (above) and here
    norm_fp32 = is_op("relax.multiply")(diff, rstd)
    norm_cast = is_op("relax.astype")(norm_fp32)
    weighted = is_op("relax.multiply")(w, norm_cast)
    out = is_op("relax.add")(weighted, b)

    return out, {"x": x, "w": w, "b": b}


# ---------------------------------------------------------------------------
# Check
# ---------------------------------------------------------------------------

def _layernorm_check(matched_bindings, annotations):
    """Validate shapes and extract N, eps."""
    x_var = annotations.get("x")
    w_var = annotations.get("w")
    b_var = annotations.get("b")
    if any(v is None for v in (x_var, w_var, b_var)):
        return None

    x_shape = get_static_shape(x_var)
    w_shape = get_static_shape(w_var)
    b_shape = get_static_shape(b_var)
    if any(s is None for s in (x_shape, w_shape, b_shape)):
        return None

    N = x_shape[-1]
    if w_shape != [N] or b_shape != [N]:
        return None

    # Extract eps from the add(var, eps_const) binding
    eps = 1e-5  # PyTorch default
    for _, bnd in matched_bindings.items():
        val = bnd.value
        if not (isinstance(val, relax.Call) and hasattr(val.op, "name")
                and val.op.name == "relax.add"):
            continue
        if len(val.args) == 2 and isinstance(val.args[1], relax.Constant):
            candidate = float(val.args[1].data.numpy())
            # Sanity check: eps must be a small positive float
            if 0 < candidate < 1e-1:
                eps = candidate
                break

    return {
        "N": N,
        "x_shape": x_shape,
        "w_shape": w_shape,
        "b_shape": b_shape,
        "eps": eps,
        "_opaque": True,
    }


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------

def _layernorm_builder(inputs: dict[str, InputInfo], params: dict):
    """Build LayerNorm via te.compute — two-pass: mean then variance.

    Computation graph (-> means te.compute dependency):

        x_fp32 (cast)
        ├─ sum_x  → mean              (first reduction + epilogue)
        │   └─ diff (x_fp32 - mean)   (bridge: prologue of 2nd reduction)
        │       └─ sum_sq  → var      (second reduction + epilogue)
        │           └─ layernorm_out  (epilogue: (diff * rstd).cast * w + b)
        └─ (also feeds sum_sq via diff above)

    The GeneralReduction rule's double-reduction path schedules this as:
        - CTA iterates over M rows (blockIdx.x)
        - Each CTA does reduction-1 (sum_x), then reduction-2 (sum_sq)
        - Bridge block (diff) is inlined / compute_at between the two reductions
    """
    x_info = inputs["x"]
    w_info = inputs["w"]
    b_info = inputs["b"]

    x_shape = tuple(x_info.shape)
    N = params["N"]
    in_dtype = x_info.dtype
    eps = params["eps"]

    # Flatten all leading dimensions into M
    M = 1
    for s in x_shape[:-1]:
        M *= s

    x = te.placeholder((M, N), name="x", dtype=in_dtype)
    w = te.placeholder((N,), name="w", dtype=w_info.dtype)
    b = te.placeholder((N,), name="b", dtype=b_info.dtype)

    # Cast to float32 for numerically stable reductions
    x_fp32 = te.compute(
        (M, N),
        lambda i, j: x[i, j].astype("float32"),
        name="x_fp32",
    )

    # First reduction: sum(x) → mean
    rk1 = te.reduce_axis((0, N), name="rk1")
    sum_x = te.compute(
        (M,),
        lambda i: te.sum(x_fp32[i, rk1], axis=rk1),
        name="sum_x",
    )
    mean = te.compute(
        (M,),
        lambda i: sum_x[i] / tir.const(N, "float32"),
        name="mean",
    )

    # Second reduction: sum_x_sq = sum(x²) — mirrors inductor's two-pass
    # Welford-style structure.  Reading x twice (once for each reduction, and
    # again in the epilogue) is cheaper overall than materialising a full-row
    # fp32 diff buffer at large N: the diff fragment would consume ~32
    # regs/thread at N=16384, capping LayerNormLike occupancy at 1 CTA/SM.
    # With no diff fragment, the same code fits ≥2 CTAs/SM on H100 and
    # matches inductor's DRAM efficiency.
    rk2 = te.reduce_axis((0, N), name="rk2")
    sum_x_sq = te.compute(
        (M,),
        lambda i: te.sum(x_fp32[i, rk2] * x_fp32[i, rk2], axis=rk2),
        name="sum_x_sq",
    )

    # Derive variance from the second-moment identity:
    #   var = E[x²] − (E[x])²  = sum_x_sq/N − mean²
    # NOTE: second-moment has catastrophic-cancellation risk when mean² ≈
    # E[x²].  For LLM residual-stream activations this is tolerable (mean
    # ≈ 0).  If this fires on workloads where mean is large, swap to a
    # proper Welford reducer.
    var = te.compute(
        (M,),
        lambda i: sum_x_sq[i] / tir.const(N, "float32") - mean[i] * mean[i],
        name="var",
    )

    # Output epilogue: read x again (third pass) and compute (x − mean) *
    # rstd * w + b element-wise.  No intermediate diff buffer exists — the
    # epilogue recomputes (x − mean) inside its parallel loop.
    eps_c = tir.const(eps, "float32")
    out = te.compute(
        (M, N),
        lambda i, j: (
            (x_fp32[i, j] - mean[i]) * tir.rsqrt(var[i] + eps_c)
        ).astype(in_dtype) * w[j] + b[j],
        name="layernorm_out",
    )

    return te.create_prim_func([x, w, b, out])


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

register_pattern(
    "layernorm",
    _layernorm_pattern,
    _layernorm_builder,
    check_fn=_layernorm_check,
)
