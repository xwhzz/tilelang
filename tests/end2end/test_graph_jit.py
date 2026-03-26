"""Tests for torch.compile(backend="tilelang") end-to-end graph compilation."""

from __future__ import annotations

import torch
import torch.nn.functional as F
import torch._dynamo

import tilelang  # noqa: F401  (triggers backend registration)
from tilelang.profiler import do_bench  # noqa: F401


# ---------------------------------------------------------------------------
# Tolerance helpers
# ---------------------------------------------------------------------------

def _dtype_tolerance(dtype: str, accum_dtype: str) -> tuple[float, float]:
    if dtype == "float16" and accum_dtype == "float32":
        return 2e-2, 2e-2
    if accum_dtype == "float32":
        return 2e-3, 2e-3
    if accum_dtype == "float16":
        return 1e-1, 1e-1
    return 1e-3, 1e-3


# ---------------------------------------------------------------------------
# Test 1: basic MLP (matmul + relu + matmul)
# ---------------------------------------------------------------------------

def test_graph_jit_mlp():
    if not torch.cuda.is_available():
        print("CUDA not available, skipping.")
        return

    torch._dynamo.reset()

    dim = 4096
    dtype = "float16"
    accum_dtype = "float32"
    torch_dtype = getattr(torch, dtype)
    torch_accum = getattr(torch, accum_dtype)

    torch.manual_seed(0)
    x = torch.randn((dim,), device="cuda", dtype=torch_dtype)
    w1 = torch.randn((dim, dim), device="cuda", dtype=torch_dtype)
    w2 = torch.randn((dim, dim), device="cuda", dtype=torch_dtype)

    # --- graph-mode via torch.compile ---
    @torch.compile(backend="tilelang")
    def mlp(x_in: torch.Tensor, w1_in: torch.Tensor, w2_in: torch.Tensor) -> torch.Tensor:
        x_acc = x_in.to(torch_accum)
        w1_acc = w1_in.to(torch_accum)
        w2_acc = w2_in.to(torch_accum)
        return w2_acc @ F.relu(w1_acc @ x_acc)

    tilelang_out = mlp(x, w1, w2)

    # --- reference ---
    @torch.compile()
    def ref(x_in, w1_in, w2_in):
        x_acc = x_in.to(torch_accum)
        w1_acc = w1_in.to(torch_accum)
        w2_acc = w2_in.to(torch_accum)
        return w2_acc @ F.relu(w1_acc @ x_acc)

    ref_out = ref(x, w1, w2)

    rtol, atol = _dtype_tolerance(dtype, accum_dtype)
    torch.testing.assert_close(tilelang_out, ref_out, rtol=rtol, atol=atol)
    print("\033[92mtest_graph_jit_mlp: correctness check passed.\033[0m")

    # --- benchmark ---
    tl_time = do_bench(lambda: mlp(x, w1, w2), backend="event")
    tc_time = do_bench(lambda: ref(x, w1, w2), backend="event")
    print(f"  tilelang backend time: {tl_time:.6f} ms, torch.compile time: {tc_time:.6f} ms")


# ---------------------------------------------------------------------------
# Test 2: shape re-specialization (different input shape → recompile)
# ---------------------------------------------------------------------------

def test_graph_jit_shape_cache():
    if not torch.cuda.is_available():
        print("CUDA not available, skipping.")
        return

    torch._dynamo.reset()

    @torch.compile(backend="tilelang")
    def add_relu(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return F.relu(a + b)

    a1 = torch.randn(128, device="cuda")
    b1 = torch.randn(128, device="cuda")
    out1 = add_relu(a1, b1)
    ref1 = F.relu(a1 + b1)
    torch.testing.assert_close(out1, ref1, rtol=1e-3, atol=1e-3)

    a2 = torch.randn(256, device="cuda")
    b2 = torch.randn(256, device="cuda")
    out2 = add_relu(a2, b2)
    ref2 = F.relu(a2 + b2)
    torch.testing.assert_close(out2, ref2, rtol=1e-3, atol=1e-3)

    print("\033[92mtest_graph_jit_shape_cache: passed.\033[0m")


# ---------------------------------------------------------------------------
# Test 3: float32 path
# ---------------------------------------------------------------------------

def test_graph_jit_float32():
    if not torch.cuda.is_available():
        print("CUDA not available, skipping.")
        return

    torch._dynamo.reset()

    dim = 1024

    torch.manual_seed(42)
    x = torch.randn((dim,), device="cuda", dtype=torch.float32)
    w1 = torch.randn((dim, dim), device="cuda", dtype=torch.float32)
    w2 = torch.randn((dim, dim), device="cuda", dtype=torch.float32)

    @torch.compile(backend="tilelang")
    def mlp_f32(x_in, w1_in, w2_in):
        return w2_in @ F.relu(w1_in @ x_in)

    out = mlp_f32(x, w1, w2)
    ref = w2 @ F.relu(w1 @ x)
    torch.testing.assert_close(out, ref, rtol=2e-3, atol=2e-3)
    print("\033[92mtest_graph_jit_float32: passed.\033[0m")


# ---------------------------------------------------------------------------
# Test 4: user-registered torch.library custom op (transparent handling)
# ---------------------------------------------------------------------------

_test4_lib = None

def test_graph_jit_torch_library_custom_op():
    if not torch.cuda.is_available():
        print("CUDA not available, skipping.")
        return

    torch._dynamo.reset()

    global _test4_lib
    _test4_lib = torch.library.Library("test_tl_graph", "DEF")
    _test4_lib.define("my_double(Tensor x) -> Tensor")

    def _cuda_impl(x):
        return x * 2.0

    def _meta_impl(x):
        return torch.empty_like(x)

    _test4_lib.impl("my_double", _cuda_impl, "CUDA")
    _test4_lib.impl("my_double", _meta_impl, "Meta")

    my_double = torch.ops.test_tl_graph.my_double

    @torch.compile(backend="tilelang")
    def fn(a: torch.Tensor) -> torch.Tensor:
        return my_double(a)

    a = torch.randn(256, device="cuda", dtype=torch.float32)
    out = fn(a)
    ref = a * 2.0
    torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-5)
    print("\033[92mtest_graph_jit_torch_library_custom_op: passed.\033[0m")


# ---------------------------------------------------------------------------
# Test 5: torch.library custom op mixed with standard ops
# ---------------------------------------------------------------------------

_test5_lib = None

def test_graph_jit_torch_library_mixed():
    if not torch.cuda.is_available():
        print("CUDA not available, skipping.")
        return

    torch._dynamo.reset()

    global _test5_lib
    _test5_lib = torch.library.Library("test_tl_graph2", "DEF")
    _test5_lib.define("my_scale(Tensor x) -> Tensor")

    def _cuda_impl(x):
        return x * 3.0

    def _meta_impl(x):
        return torch.empty_like(x)

    _test5_lib.impl("my_scale", _cuda_impl, "CUDA")
    _test5_lib.impl("my_scale", _meta_impl, "Meta")

    my_scale = torch.ops.test_tl_graph2.my_scale

    @torch.compile(backend="tilelang")
    def fn(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        s = a + b           # standard op
        s = my_scale(s)     # custom torch op
        return F.relu(s)    # standard op

    a = torch.randn(256, device="cuda", dtype=torch.float32)
    b = torch.randn(256, device="cuda", dtype=torch.float32)
    out = fn(a, b)
    ref = F.relu((a + b) * 3.0)
    torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-5)
    print("\033[92mtest_graph_jit_torch_library_mixed: passed.\033[0m")


# ---------------------------------------------------------------------------
# Test 7: trace inspection API
# ---------------------------------------------------------------------------

def test_graph_jit_trace():
    if not torch.cuda.is_available():
        print("CUDA not available, skipping.")
        return

    torch._dynamo.reset()
    from tilelang.jit.backend import clear_compilation_traces, get_compilation_traces

    clear_compilation_traces()

    @torch.compile(backend="tilelang")
    def add_fn(a, b):
        return a + b

    a = torch.randn(256, device="cuda")
    b = torch.randn(256, device="cuda")
    add_fn(a, b)

    traces = get_compilation_traces()
    assert len(traces) > 0, "Expected at least one compilation trace"
    print(f"  Got {len(traces)} trace(s)")
    for t in traces:
        print(f"    {t.summary()}")
    print("\033[92mtest_graph_jit_trace: passed.\033[0m")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    test_graph_jit_mlp()
    test_graph_jit_shape_cache()
    test_graph_jit_float32()
    test_graph_jit_torch_library_custom_op()
    test_graph_jit_torch_library_mixed()
    test_graph_jit_trace()
    print("\n\033[92mAll graph-mode JIT tests passed.\033[0m")
