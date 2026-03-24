"""Tests for tilelang.jit(mode="graph") end-to-end graph compilation."""

from __future__ import annotations

import torch
import torch.nn.functional as F

import tilelang
from tilelang.profiler import do_bench


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

    dim = 4096
    dtype = "float16"
    accum_dtype = "float32"
    torch_dtype = getattr(torch, dtype)
    torch_accum = getattr(torch, accum_dtype)

    torch.manual_seed(0)
    x = torch.randn((dim,), device="cuda", dtype=torch_dtype)
    w1 = torch.randn((dim, dim), device="cuda", dtype=torch_dtype)
    w2 = torch.randn((dim, dim), device="cuda", dtype=torch_dtype)

    # --- graph-mode JIT ---
    @tilelang.jit(mode="graph")
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
    print(f"  tilelang.jit(graph) time: {tl_time:.6f} ms, torch.compile time: {tc_time:.6f} ms")


# ---------------------------------------------------------------------------
# Test 2: shape re-specialization (different input shape → recompile)
# ---------------------------------------------------------------------------

def test_graph_jit_shape_cache():
    if not torch.cuda.is_available():
        print("CUDA not available, skipping.")
        return

    @tilelang.jit(mode="graph")
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

    # Check that both shapes are cached
    assert len(add_relu._cache) == 2, f"Expected 2 cached runners, got {len(add_relu._cache)}"
    print("\033[92mtest_graph_jit_shape_cache: passed.\033[0m")


# ---------------------------------------------------------------------------
# Test 3: explicit compile() to inspect the runner
# ---------------------------------------------------------------------------

def test_graph_jit_explicit_compile():
    if not torch.cuda.is_available():
        print("CUDA not available, skipping.")
        return

    @tilelang.jit(mode="graph")
    def simple_matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return a @ b

    M, N, K = 512, 512, 512
    a = torch.randn(M, K, device="cuda")
    b = torch.randn(K, N, device="cuda")

    runner = simple_matmul.compile(a, b)
    assert hasattr(runner, "kernels"), "GraphRunner should expose .kernels"
    assert hasattr(runner, "calls"), "GraphRunner should expose .calls"
    assert len(runner.calls) > 0, "Expected at least one kernel call"

    out = runner(a, b)
    ref = a @ b
    torch.testing.assert_close(out, ref, rtol=1e-3, atol=1e-3)
    print("\033[92mtest_graph_jit_explicit_compile: passed.\033[0m")


# ---------------------------------------------------------------------------
# Test 4: float32 path
# ---------------------------------------------------------------------------

def test_graph_jit_float32():
    if not torch.cuda.is_available():
        print("CUDA not available, skipping.")
        return

    dim = 1024

    torch.manual_seed(42)
    x = torch.randn((dim,), device="cuda", dtype=torch.float32)
    w1 = torch.randn((dim, dim), device="cuda", dtype=torch.float32)
    w2 = torch.randn((dim, dim), device="cuda", dtype=torch.float32)

    @tilelang.jit(mode="graph")
    def mlp_f32(x_in, w1_in, w2_in):
        return w2_in @ F.relu(w1_in @ x_in)

    out = mlp_f32(x, w1, w2)
    ref = w2 @ F.relu(w1 @ x)
    torch.testing.assert_close(out, ref, rtol=2e-3, atol=2e-3)
    print("\033[92mtest_graph_jit_float32: passed.\033[0m")


# ---------------------------------------------------------------------------
# Test 5: CUDA graph capture via enable_cuda_graph()
# ---------------------------------------------------------------------------

def test_graph_jit_cuda_graph_manual():
    if not torch.cuda.is_available():
        print("CUDA not available, skipping.")
        return

    dim = 1024
    torch.manual_seed(0)
    x = torch.randn((dim,), device="cuda", dtype=torch.float16)
    w1 = torch.randn((dim, dim), device="cuda", dtype=torch.float16)
    w2 = torch.randn((dim, dim), device="cuda", dtype=torch.float16)

    @tilelang.jit(mode="graph")
    def mlp(x_in, w1_in, w2_in):
        return w2_in @ F.relu(w1_in @ x_in)

    # Compile and enable CUDA graph manually
    runner = mlp.compile(x, w1, w2)
    runner.enable_cuda_graph(warmup_iters=3)

    # First call triggers capture; subsequent calls replay.
    out1 = runner(x, w1, w2)
    ref1 = w2.float() @ F.relu(w1.float() @ x.float())
    rtol, atol = _dtype_tolerance("float16", "float32")
    torch.testing.assert_close(out1.float(), ref1, rtol=rtol, atol=atol)

    # Second call with different data — must still be correct.
    x2 = torch.randn((dim,), device="cuda", dtype=torch.float16)
    out2 = runner(x2, w1, w2)
    ref2 = w2.float() @ F.relu(w1.float() @ x2.float())
    torch.testing.assert_close(out2.float(), ref2, rtol=rtol, atol=atol)

    assert runner._cuda_graph is not None, "CUDA graph should be captured"
    print("\033[92mtest_graph_jit_cuda_graph_manual: passed.\033[0m")


# ---------------------------------------------------------------------------
# Test 6: CUDA graph via decorator (cuda_graph=True)
# ---------------------------------------------------------------------------

def test_graph_jit_cuda_graph_decorator():
    if not torch.cuda.is_available():
        print("CUDA not available, skipping.")
        return

    dim = 1024
    torch.manual_seed(42)
    x = torch.randn((dim,), device="cuda", dtype=torch.float16)
    w1 = torch.randn((dim, dim), device="cuda", dtype=torch.float16)
    w2 = torch.randn((dim, dim), device="cuda", dtype=torch.float16)

    @tilelang.jit(mode="graph", cuda_graph=True)
    def mlp_cg(x_in, w1_in, w2_in):
        return w2_in @ F.relu(w1_in @ x_in)

    # First call: compile + capture + return result
    out = mlp_cg(x, w1, w2)
    ref = w2.float() @ F.relu(w1.float() @ x.float())
    rtol, atol = _dtype_tolerance("float16", "float32")
    torch.testing.assert_close(out.float(), ref, rtol=rtol, atol=atol)

    # Verify graph was captured
    runner = mlp_cg._cache[list(mlp_cg._cache.keys())[0]]
    assert runner._cuda_graph is not None, "cuda_graph=True should auto-capture"

    # Second call with new data — replays captured graph
    x2 = torch.randn((dim,), device="cuda", dtype=torch.float16)
    out2 = mlp_cg(x2, w1, w2)
    ref2 = w2.float() @ F.relu(w1.float() @ x2.float())
    torch.testing.assert_close(out2.float(), ref2, rtol=rtol, atol=atol)
    print("\033[92mtest_graph_jit_cuda_graph_decorator: passed.\033[0m")


# ---------------------------------------------------------------------------
# Test 7: CUDA graph benchmark
# ---------------------------------------------------------------------------

def test_graph_jit_cuda_graph_benchmark():
    if not torch.cuda.is_available():
        print("CUDA not available, skipping.")
        return

    dim = 4096
    torch.manual_seed(0)
    x = torch.randn((dim,), device="cuda", dtype=torch.float16)
    w1 = torch.randn((dim, dim), device="cuda", dtype=torch.float16)
    w2 = torch.randn((dim, dim), device="cuda", dtype=torch.float16)

    # Normal graph runner (no CUDA graph)
    @tilelang.jit(mode="graph")
    def mlp_normal(x_in, w1_in, w2_in):
        return w2_in @ F.relu(w1_in @ x_in)

    # CUDA graph runner
    @tilelang.jit(mode="graph", cuda_graph=True)
    def mlp_cudagraph(x_in, w1_in, w2_in):
        return w2_in @ F.relu(w1_in @ x_in)

    # Warm up both
    mlp_normal(x, w1, w2)
    mlp_cudagraph(x, w1, w2)

    normal_time = do_bench(lambda: mlp_normal(x, w1, w2), backend="event")
    cg_time = do_bench(lambda: mlp_cudagraph(x, w1, w2), backend="event")
    print(f"\033[92mtest_graph_jit_cuda_graph_benchmark:\033[0m "
          f"normal={normal_time:.6f} ms, cuda_graph={cg_time:.6f} ms")


# ---------------------------------------------------------------------------
# Test 8: native C++ dispatch via enable_native_dispatch()
# ---------------------------------------------------------------------------

def test_graph_jit_native_manual():
    if not torch.cuda.is_available():
        print("CUDA not available, skipping.")
        return

    dim = 1024
    torch.manual_seed(0)
    x = torch.randn((dim,), device="cuda", dtype=torch.float16)
    w1 = torch.randn((dim, dim), device="cuda", dtype=torch.float16)
    w2 = torch.randn((dim, dim), device="cuda", dtype=torch.float16)

    @tilelang.jit(mode="graph")
    def mlp(x_in, w1_in, w2_in):
        return w2_in @ F.relu(w1_in @ x_in)

    runner = mlp.compile(x, w1, w2)
    runner.enable_native_dispatch()

    out = runner(x, w1, w2)
    ref = w2.float() @ F.relu(w1.float() @ x.float())
    rtol, atol = _dtype_tolerance("float16", "float32")
    torch.testing.assert_close(out.float(), ref, rtol=rtol, atol=atol)

    # Different data
    x2 = torch.randn((dim,), device="cuda", dtype=torch.float16)
    out2 = runner(x2, w1, w2)
    ref2 = w2.float() @ F.relu(w1.float() @ x2.float())
    torch.testing.assert_close(out2.float(), ref2, rtol=rtol, atol=atol)

    assert runner._native_func is not None
    print("\033[92mtest_graph_jit_native_manual: passed.\033[0m")


# ---------------------------------------------------------------------------
# Test 9: native dispatch via decorator (native=True)
# ---------------------------------------------------------------------------

def test_graph_jit_native_decorator():
    if not torch.cuda.is_available():
        print("CUDA not available, skipping.")
        return

    dim = 1024
    torch.manual_seed(42)
    x = torch.randn((dim,), device="cuda", dtype=torch.float16)
    w1 = torch.randn((dim, dim), device="cuda", dtype=torch.float16)
    w2 = torch.randn((dim, dim), device="cuda", dtype=torch.float16)

    @tilelang.jit(mode="graph", native=True)
    def mlp_native(x_in, w1_in, w2_in):
        return w2_in @ F.relu(w1_in @ x_in)

    out = mlp_native(x, w1, w2)
    ref = w2.float() @ F.relu(w1.float() @ x.float())
    rtol, atol = _dtype_tolerance("float16", "float32")
    torch.testing.assert_close(out.float(), ref, rtol=rtol, atol=atol)

    runner = mlp_native._cache[list(mlp_native._cache.keys())[0]]
    assert runner._native_func is not None, "native=True should build C dispatch"
    print("\033[92mtest_graph_jit_native_decorator: passed.\033[0m")


# ---------------------------------------------------------------------------
# Test 10: native dispatch benchmark
# ---------------------------------------------------------------------------

def test_graph_jit_native_benchmark():
    if not torch.cuda.is_available():
        print("CUDA not available, skipping.")
        return

    dim = 4096
    torch.manual_seed(0)
    x = torch.randn((dim,), device="cuda", dtype=torch.float16)
    w1 = torch.randn((dim, dim), device="cuda", dtype=torch.float16)
    w2 = torch.randn((dim, dim), device="cuda", dtype=torch.float16)

    @tilelang.jit(mode="graph")
    def mlp_py(x_in, w1_in, w2_in):
        return w2_in @ F.relu(w1_in @ x_in)

    @tilelang.jit(mode="graph", native=True)
    def mlp_native(x_in, w1_in, w2_in):
        return w2_in @ F.relu(w1_in @ x_in)

    mlp_py(x, w1, w2)
    mlp_native(x, w1, w2)

    py_time = do_bench(lambda: mlp_py(x, w1, w2), backend="event")
    native_time = do_bench(lambda: mlp_native(x, w1, w2), backend="event")
    print(f"\033[92mtest_graph_jit_native_benchmark:\033[0m "
          f"python={py_time:.6f} ms, native={native_time:.6f} ms")


# ---------------------------------------------------------------------------
# Test 11: JITKernel auto-detection – custom add kernel via global reference
# ---------------------------------------------------------------------------

def test_graph_jit_detect_global_kernel():
    if not torch.cuda.is_available():
        print("CUDA not available, skipping.")
        return

    from tilelang import language as T

    dim = 128

    @T.prim_func
    def add_kernel(
        A: T.Tensor((dim,), "float32"),
        B: T.Tensor((dim,), "float32"),
        C: T.Tensor((dim,), "float32"),
    ):
        with T.Kernel(1, threads=128) as bx:
            for i in T.Parallel(dim):
                C[i] = A[i] + B[i]

    # Pre-compile the kernel
    my_add = tilelang.compile(add_kernel)

    # Use the pre-compiled kernel inside a graph-mode function
    @tilelang.jit(mode="graph")
    def fn(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return my_add(a, b)

    a = torch.randn(dim, device="cuda", dtype=torch.float32)
    b = torch.randn(dim, device="cuda", dtype=torch.float32)
    out = fn(a, b)
    ref = a + b
    torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-5)
    print("\033[92mtest_graph_jit_detect_global_kernel: passed.\033[0m")


# ---------------------------------------------------------------------------
# Test 12: JITKernel auto-detection – closure variable
# ---------------------------------------------------------------------------

def test_graph_jit_detect_closure_kernel():
    if not torch.cuda.is_available():
        print("CUDA not available, skipping.")
        return

    from tilelang import language as T

    dim = 128

    @T.prim_func
    def mul_kernel(
        A: T.Tensor((dim,), "float32"),
        B: T.Tensor((dim,), "float32"),
        C: T.Tensor((dim,), "float32"),
    ):
        with T.Kernel(1, threads=128) as bx:
            for i in T.Parallel(dim):
                C[i] = A[i] * B[i]

    my_mul = tilelang.compile(mul_kernel)

    def make_fn():
        # my_mul captured via closure
        @tilelang.jit(mode="graph")
        def fn(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            return my_mul(a, b)
        return fn

    fn = make_fn()
    a = torch.randn(dim, device="cuda", dtype=torch.float32)
    b = torch.randn(dim, device="cuda", dtype=torch.float32)
    out = fn(a, b)
    ref = a * b
    torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-5)
    print("\033[92mtest_graph_jit_detect_closure_kernel: passed.\033[0m")


# ---------------------------------------------------------------------------
# Test 13: JITKernel mixed with standard ops
# ---------------------------------------------------------------------------

def test_graph_jit_mixed_custom_and_standard():
    if not torch.cuda.is_available():
        print("CUDA not available, skipping.")
        return

    from tilelang import language as T

    dim = 256

    @T.prim_func
    def custom_add(
        A: T.Tensor((dim,), "float32"),
        B: T.Tensor((dim,), "float32"),
        C: T.Tensor((dim,), "float32"),
    ):
        with T.Kernel(1, threads=128) as bx:
            for i in T.Parallel(dim):
                C[i] = A[i] + B[i]

    my_add = tilelang.compile(custom_add)

    @tilelang.jit(mode="graph")
    def fn(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        # Custom kernel for add, standard relu auto-scheduled
        s = my_add(a, b)
        return F.relu(s)

    a = torch.randn(dim, device="cuda", dtype=torch.float32)
    b = torch.randn(dim, device="cuda", dtype=torch.float32)
    out = fn(a, b)
    ref = F.relu(a + b)
    torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-5)

    # Check that the runner has at least 2 kernels (custom add + auto relu)
    runner = fn._cache[list(fn._cache.keys())[0]]
    assert len(runner.calls) >= 2, f"Expected >=2 calls, got {len(runner.calls)}"
    print("\033[92mtest_graph_jit_mixed_custom_and_standard: passed.\033[0m")


# ---------------------------------------------------------------------------
# Test 14: JITKernel with CUDA graph capture
# ---------------------------------------------------------------------------

def test_graph_jit_detect_kernel_cuda_graph():
    if not torch.cuda.is_available():
        print("CUDA not available, skipping.")
        return

    from tilelang import language as T

    dim = 128

    @T.prim_func
    def add_kernel(
        A: T.Tensor((dim,), "float32"),
        B: T.Tensor((dim,), "float32"),
        C: T.Tensor((dim,), "float32"),
    ):
        with T.Kernel(1, threads=128) as bx:
            for i in T.Parallel(dim):
                C[i] = A[i] + B[i]

    my_add = tilelang.compile(add_kernel)

    @tilelang.jit(mode="graph", cuda_graph=True)
    def fn(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return my_add(a, b)

    a = torch.randn(dim, device="cuda", dtype=torch.float32)
    b = torch.randn(dim, device="cuda", dtype=torch.float32)
    out = fn(a, b)
    ref = a + b
    torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-5)

    # Verify CUDA graph was captured
    runner = fn._cache[list(fn._cache.keys())[0]]
    assert runner._cuda_graph is not None, "CUDA graph should be captured"

    # Second call with different data
    a2 = torch.randn(dim, device="cuda", dtype=torch.float32)
    b2 = torch.randn(dim, device="cuda", dtype=torch.float32)
    out2 = fn(a2, b2)
    ref2 = a2 + b2
    torch.testing.assert_close(out2, ref2, rtol=1e-5, atol=1e-5)
    print("\033[92mtest_graph_jit_detect_kernel_cuda_graph: passed.\033[0m")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    test_graph_jit_mlp()
    test_graph_jit_shape_cache()
    test_graph_jit_explicit_compile()
    test_graph_jit_float32()
    test_graph_jit_cuda_graph_manual()
    test_graph_jit_cuda_graph_decorator()
    test_graph_jit_cuda_graph_benchmark()
    test_graph_jit_native_manual()
    test_graph_jit_native_decorator()
    test_graph_jit_native_benchmark()
    test_graph_jit_detect_global_kernel()
    test_graph_jit_detect_closure_kernel()
    test_graph_jit_mixed_custom_and_standard()
    test_graph_jit_detect_kernel_cuda_graph()
    print("\n\033[92mAll graph-mode JIT tests passed.\033[0m")
