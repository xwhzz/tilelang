"""Test FuseSkipReduction pass on RMSNorm-like patterns."""

import pytest
from tilelang import tvm as tvm
from tvm import relax, tir, ir
from tvm.script import relax as R, tir as T

from tilelang.graph.fuse_skip_reduction import FuseSkipReduction


def _make_rmsnorm_module():
    """Build a Relax module with the cross-reduction skip pattern:

        lv1 = call_tir(cast, (x,))          # fp16→fp32, elemwise
        lv2 = call_tir(power, (lv1,))        # x^2, elemwise
        out  = fused_group(lv2, lv1, w)      # Primitive: mean→rsqrt→mul→mul
    """

    @T.prim_func
    def cast_func(A: T.Buffer((4096,), "float16"), B: T.Buffer((4096,), "float32")):
        T.func_attr({"op_pattern": 0})  # kElemWise
        for i in range(4096):
            B[i] = T.Cast("float32", A[i])

    @T.prim_func
    def power_func(A: T.Buffer((4096,), "float32"), B: T.Buffer((4096,), "float32")):
        T.func_attr({"op_pattern": 0})  # kElemWise
        for i in range(4096):
            B[i] = A[i] * A[i]

    @T.prim_func
    def mean_func(A: T.Buffer((4096,), "float32"), B: T.Buffer((1,), "float32")):
        T.func_attr({"op_pattern": 3})  # kCommReduce
        B[0] = T.float32(0)
        for i in range(4096):
            B[0] = B[0] + A[i]
        B[0] = B[0] / T.float32(4096)

    @T.prim_func
    def rsqrt_func(A: T.Buffer((1,), "float32"), B: T.Buffer((1,), "float32")):
        T.func_attr({"op_pattern": 0})
        B[0] = T.rsqrt(A[0] + T.float32(1e-6))

    @T.prim_func
    def mul_func(
        A: T.Buffer((4096,), "float32"),
        B: T.Buffer((1,), "float32"),
        C: T.Buffer((4096,), "float32"),
    ):
        T.func_attr({"op_pattern": 1})  # kBroadcast
        for i in range(4096):
            C[i] = A[i] * B[0]

    @T.prim_func
    def mul_weight_func(
        A: T.Buffer((4096,), "float32"),
        W: T.Buffer((4096,), "float16"),
        C: T.Buffer((4096,), "float16"),
    ):
        T.func_attr({"op_pattern": 0})
        for i in range(4096):
            C[i] = T.Cast("float16", A[i] * T.Cast("float32", W[i]))

    # Build Relax module manually
    bb = relax.BlockBuilder()

    # Add TIR functions
    cast_gv = bb.add_func(cast_func, "cast")
    power_gv = bb.add_func(power_func, "power")
    mean_gv = bb.add_func(mean_func, "mean")
    rsqrt_gv = bb.add_func(rsqrt_func, "rsqrt")
    mul_gv = bb.add_func(mul_func, "mul")
    mul_w_gv = bb.add_func(mul_weight_func, "mul_weight")

    # Build the Primitive fused function: fused_group(lv2, lv1, w)
    p0 = relax.Var("lv2", R.Tensor((4096,), "float32"))
    p1 = relax.Var("lv1", R.Tensor((4096,), "float32"))
    p2 = relax.Var("w", R.Tensor((4096,), "float16"))

    call_tir_op = ir.Op.get("relax.call_tir")

    with bb.function("fused_group", [p0, p1, p2]):
        with bb.dataflow():
            m = bb.emit(relax.Call(call_tir_op, [mean_gv, relax.Tuple([p0])],
                                  sinfo_args=[R.Tensor((1,), "float32")]),
                        name_hint="mean_out")
            rs = bb.emit(relax.Call(call_tir_op, [rsqrt_gv, relax.Tuple([m])],
                                   sinfo_args=[R.Tensor((1,), "float32")]),
                         name_hint="rsqrt_out")
            scaled = bb.emit(relax.Call(call_tir_op, [mul_gv, relax.Tuple([p1, rs])],
                                       sinfo_args=[R.Tensor((4096,), "float32")]),
                             name_hint="scaled")
            out = bb.emit(relax.Call(call_tir_op, [mul_w_gv, relax.Tuple([scaled, p2])],
                                    sinfo_args=[R.Tensor((4096,), "float16")]),
                          name_hint="out")
            bb.emit_output(out)
        bb.emit_func_output(out)

    fused_gv = None
    tmp_mod = bb.get()
    for gv in tmp_mod.functions:
        if gv.name_hint == "fused_group":
            fused_gv = gv
            break
    # Mark as Primitive
    fused_func = tmp_mod[fused_gv].with_attr("Primitive", tvm.tir.IntImm("int32", 1))

    # Rebuild with main function
    bb2 = relax.BlockBuilder()
    for gv, func in tmp_mod.functions.items():
        if gv.name_hint == "fused_group":
            fused_gv2 = bb2.add_func(fused_func, "fused_group")
        elif isinstance(func, tir.PrimFunc):
            bb2.add_func(func, gv.name_hint)

    # Re-lookup TIR gvars in new module
    partial = bb2.get()
    cast_gv2 = None
    power_gv2 = None
    fused_gv_final = None
    for gv in partial.functions:
        if gv.name_hint == "cast":
            cast_gv2 = gv
        elif gv.name_hint == "power":
            power_gv2 = gv
        elif gv.name_hint == "fused_group":
            fused_gv_final = gv

    # Build main
    x = relax.Var("x", R.Tensor((4096,), "float16"))
    w = relax.Var("w", R.Tensor((4096,), "float16"))

    with bb2.function("main", [x, w]):
        with bb2.dataflow():
            lv1 = bb2.emit(
                relax.Call(call_tir_op, [cast_gv2, relax.Tuple([x])],
                           sinfo_args=[R.Tensor((4096,), "float32")]),
                name_hint="lv1")
            lv2 = bb2.emit(
                relax.Call(call_tir_op, [power_gv2, relax.Tuple([lv1])],
                           sinfo_args=[R.Tensor((4096,), "float32")]),
                name_hint="lv2")
            out = bb2.emit(
                relax.Call(fused_gv_final, [lv2, lv1, w],
                           sinfo_args=[R.Tensor((4096,), "float16")]),
                name_hint="result")
            bb2.emit_output(out)
        bb2.emit_func_output(out)

    return bb2.get()


def test_fuse_skip_reduction_merges_producers():
    """After the pass, cast and power should be inlined into the fused group."""
    mod = _make_rmsnorm_module()

    # Count standalone call_tir bindings in main before
    main_before = mod["main"]
    bindings_before = []
    for block in main_before.body.blocks:
        bindings_before.extend(block.bindings)
    # Should have 4 bindings: cast, power, fused_group call, dataflow output
    assert len(bindings_before) == 4, f"Expected 4 bindings, got {len(bindings_before)}"

    # Apply pass
    mod_after = FuseSkipReduction().transform_module(mod, None)

    # Check main after: should have 1 real binding (the merged call)
    # plus dataflow output bindings
    main_after = mod_after["main"]
    call_bindings = []
    for block in main_after.body.blocks:
        for b in block.bindings:
            if isinstance(b, relax.VarBinding) and isinstance(b.value, relax.Call):
                call_bindings.append(b)
    assert len(call_bindings) == 1, (
        f"Expected 1 call binding after fusion, got {len(call_bindings)}")

    # The merged function should contain cast + power + original group ops
    merged_call = call_bindings[0].value
    assert isinstance(merged_call.op, ir.GlobalVar), "Expected call to merged function"
    merged_func = mod_after[merged_call.op]
    assert isinstance(merged_func, relax.Function), "Expected Relax Function"
    assert merged_func.attrs and merged_func.attrs.get("Primitive"), \
        "Merged function should be Primitive"

    # Count call_tir inside the merged function
    inner_calls = []
    for block in merged_func.body.blocks:
        for b in block.bindings:
            if isinstance(b, relax.VarBinding) and isinstance(b.value, relax.Call):
                if isinstance(b.value.op, ir.Op) and b.value.op.name == "relax.call_tir":
                    inner_calls.append(b)
    # cast + power + mean + rsqrt + mul + mul_weight = 6 call_tir
    assert len(inner_calls) == 6, (
        f"Expected 6 call_tir (2 inlined + 4 original), got {len(inner_calls)}")

    # The merged function should take 2 params (x, w) instead of 3 (lv2, lv1, w)
    assert len(merged_func.params) == 2, (
        f"Expected 2 params, got {len(merged_func.params)}")


def test_no_change_when_no_pattern():
    """Pass should be no-op when there are no inlineable patterns."""
    bb = relax.BlockBuilder()

    @T.prim_func
    def add_func(A: T.Buffer((16,), "float32"), B: T.Buffer((16,), "float32"),
                 C: T.Buffer((16,), "float32")):
        T.func_attr({"op_pattern": 0})
        for i in range(16):
            C[i] = A[i] + B[i]

    add_gv = bb.add_func(add_func, "add")
    call_tir_op = ir.Op.get("relax.call_tir")

    x = relax.Var("x", R.Tensor((16,), "float32"))
    y = relax.Var("y", R.Tensor((16,), "float32"))

    with bb.function("main", [x, y]):
        with bb.dataflow():
            out = bb.emit(
                relax.Call(call_tir_op, [add_gv, relax.Tuple([x, y])],
                           sinfo_args=[R.Tensor((16,), "float32")]),
                name_hint="out")
            bb.emit_output(out)
        bb.emit_func_output(out)

    mod = bb.get()
    mod_after = FuseSkipReduction().transform_module(mod, None)

    # Should be unchanged - still 1 call_tir
    main = mod_after["main"]
    calls = []
    for block in main.body.blocks:
        for b in block.bindings:
            if isinstance(b, relax.VarBinding) and isinstance(b.value, relax.Call):
                if isinstance(b.value.op, ir.Op):
                    calls.append(b)
    assert len(calls) == 1


if __name__ == "__main__":
    test_fuse_skip_reduction_merges_producers()
    print("PASS: test_fuse_skip_reduction_merges_producers")
    test_no_change_when_no_pattern()
    print("PASS: test_no_change_when_no_pattern")
