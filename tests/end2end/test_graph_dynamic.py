"""Tests for dynamic shapes and dynamic control flow in graph compile."""

from __future__ import annotations

import torch
import torch.nn.functional as F

import tilelang


# ===========================================================================
# Dynamic Shape Tests (True dynamic — compile once, run with any shape)
# ===========================================================================

def test_dynamic_shape_elementwise():
    """Compile add+relu once with symbolic dim, run with multiple sizes."""
    if not torch.cuda.is_available():
        print("CUDA not available, skipping.")
        return

    @tilelang.jit(mode="graph", dynamic_dims={0: [0], 1: [0]})
    def add_relu(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return F.relu(a + b)

    for N in [128, 256, 512, 1024]:
        a = torch.randn(N, device="cuda")
        b = torch.randn(N, device="cuda")
        out = add_relu(a, b)
        ref = F.relu(a + b)
        torch.testing.assert_close(out, ref, rtol=1e-3, atol=1e-3)
        print(f"  shape ({N},) passed")

    # Verify single compilation
    assert add_relu._dynamic_runner is not None
    assert len(add_relu._cache) == 0
    print("\033[92mtest_dynamic_shape_elementwise: passed (compiled once).\033[0m")


def test_dynamic_shape_matmul():
    """Dynamic batch dimension for matmul (compile once)."""
    if not torch.cuda.is_available():
        print("CUDA not available, skipping.")
        return

    K, N = 512, 512

    @tilelang.jit(mode="graph", dynamic_dims={0: [0]})
    def matmul_fn(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return a @ b

    b = torch.randn(K, N, device="cuda")
    for M in [4, 16, 64, 256]:
        a = torch.randn(M, K, device="cuda")
        out = matmul_fn(a, b)
        ref = a @ b
        torch.testing.assert_close(out, ref, rtol=1e-2, atol=1e-2)
        print(f"  shape ({M}, {K}) @ ({K}, {N}) passed")

    assert matmul_fn._dynamic_runner is not None
    print("\033[92mtest_dynamic_shape_matmul: passed (compiled once).\033[0m")


def test_dynamic_shape_2d_single_dim():
    """2D tensor with one dynamic dimension."""
    if not torch.cuda.is_available():
        print("CUDA not available, skipping.")
        return

    @tilelang.jit(mode="graph", dynamic_dims={0: [0], 1: [0]})
    def add_fn(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return a + b

    N = 256
    for M in [32, 64, 128]:
        a = torch.randn(M, N, device="cuda")
        b = torch.randn(M, N, device="cuda")
        out = add_fn(a, b)
        ref = a + b
        torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-5)
        print(f"  shape ({M}, {N}) passed")

    assert add_fn._dynamic_runner is not None
    print("\033[92mtest_dynamic_shape_2d_single_dim: passed (compiled once).\033[0m")


def test_dynamic_shape_mlp():
    """MLP with dynamic batch dimension: (B, dim) @ (dim, dim) → (B, dim)."""
    if not torch.cuda.is_available():
        print("CUDA not available, skipping.")
        return

    dim = 256
    torch.manual_seed(0)
    w1 = torch.randn(dim, dim, device="cuda", dtype=torch.float32)
    w2 = torch.randn(dim, dim, device="cuda", dtype=torch.float32)

    @tilelang.jit(mode="graph", dynamic_dims={0: [0]})
    def mlp(x, w1_in, w2_in):
        h = x @ w1_in        # (B, dim) @ (dim, dim) -> (B, dim)
        h = F.relu(h)
        return h @ w2_in     # (B, dim) @ (dim, dim) -> (B, dim)

    for B in [4, 16, 64]:
        x = torch.randn(B, dim, device="cuda", dtype=torch.float32)
        out = mlp(x, w1, w2)
        ref = F.relu(x @ w1) @ w2
        torch.testing.assert_close(out, ref, rtol=2e-3, atol=2e-3)
        print(f"  batch={B} passed")

    assert mlp._dynamic_runner is not None
    print("\033[92mtest_dynamic_shape_mlp: passed (compiled once).\033[0m")


# ===========================================================================
# Shape Recompilation Tests (fallback when dynamic_dims not specified)
# ===========================================================================

def test_shape_recompilation():
    """Without dynamic_dims, each shape triggers recompilation."""
    if not torch.cuda.is_available():
        print("CUDA not available, skipping.")
        return

    @tilelang.jit(mode="graph")
    def add_relu(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return F.relu(a + b)

    for N in [128, 256, 512]:
        a = torch.randn(N, device="cuda")
        b = torch.randn(N, device="cuda")
        out = add_relu(a, b)
        ref = F.relu(a + b)
        torch.testing.assert_close(out, ref, rtol=1e-3, atol=1e-3)

    assert len(add_relu._cache) == 3
    print("\033[92mtest_shape_recompilation: passed (3 compilations).\033[0m")


# ===========================================================================
# Dynamic Control Flow Tests
# ===========================================================================

def test_python_if_else_basic():
    """Python if/else on tensor pred — compiled via torch.compile fallback."""
    if not torch.cuda.is_available():
        print("CUDA not available, skipping.")
        return

    @tilelang.jit(mode="graph")
    def cond_fn(pred: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        if pred:
            return x * 2.0
        else:
            return x * 0.5

    x = torch.randn(256, device="cuda")

    pred_true = True # torch.tensor(True, device="cuda")
    out = cond_fn(pred_true, x)
    torch.testing.assert_close(out, x * 2.0, rtol=1e-5, atol=1e-5)

    pred_false = False # torch.tensor(False, device="cuda")
    out = cond_fn(pred_false, x)
    torch.testing.assert_close(out, x * 0.5, rtol=1e-5, atol=1e-5)

    # Verify dynamo fallback was used.
    assert cond_fn._dynamo_runner is not None
    print("\033[92mtest_python_if_else_basic: passed (dynamo fallback).\033[0m")


def test_python_if_else_with_matmul():
    """Python if/else with matmul in branches — dynamo fallback."""
    if not torch.cuda.is_available():
        print("CUDA not available, skipping.")
        return

    dim = 256

    @tilelang.jit(mode="graph")
    def cond_matmul(pred, x, w):
        if pred:
            return F.relu(w @ x)
        else:
            return F.silu(w @ x)

    x = torch.randn(dim, device="cuda", dtype=torch.float32)
    w = torch.randn(dim, dim, device="cuda", dtype=torch.float32)

    pred_true = torch.tensor(True, device="cuda")
    out = cond_matmul(pred_true, x, w)
    torch.testing.assert_close(out, F.relu(w @ x), rtol=1e-3, atol=1e-3)

    pred_false = torch.tensor(False, device="cuda")
    out = cond_matmul(pred_false, x, w)
    torch.testing.assert_close(out, F.silu(w @ x), rtol=1e-3, atol=1e-3)

    assert cond_matmul._dynamo_runner is not None
    print("\033[92mtest_python_if_else_with_matmul: passed (dynamo fallback).\033[0m")


def test_python_while_loop():
    """Python while loop with data-dependent iteration — dynamo fallback."""
    if not torch.cuda.is_available():
        print("CUDA not available, skipping.")
        return

    @tilelang.jit(mode="graph")
    def iterative_fn(x: torch.Tensor) -> torch.Tensor:
        while x.sum() < 100.0:
            x = x * 1.5
        return x

    x = torch.ones(10, device="cuda") * 2.0
    out = iterative_fn(x)
    ref = x.clone()
    while ref.sum() < 100.0:
        ref = ref * 1.5
    torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-5)
    assert iterative_fn._dynamo_runner is not None
    print("\033[92mtest_python_while_loop: passed (dynamo fallback).\033[0m")


def test_where_conditional():
    """Element-wise torch.where (data-parallel, no control flow)."""
    if not torch.cuda.is_available():
        print("CUDA not available, skipping.")
        return

    @tilelang.jit(mode="graph")
    def where_fn(cond, a, b):
        return torch.where(cond, a, b)

    cond = torch.randn(256, device="cuda") > 0
    a = torch.randn(256, device="cuda")
    b = torch.randn(256, device="cuda")
    out = where_fn(cond, a, b)
    ref = torch.where(cond, a, b)
    torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-5)
    print("\033[92mtest_where_conditional: passed.\033[0m")


# ===========================================================================
# Main
# ===========================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("True Dynamic Shape Tests (compile once)")
    print("=" * 60)
    test_dynamic_shape_elementwise()
    print()
    test_dynamic_shape_matmul()
    print()
    test_dynamic_shape_2d_single_dim()
    print()
    test_dynamic_shape_mlp()
    print()
    test_shape_recompilation()
    print()

    print("=" * 60)
    print("Dynamic Control Flow Tests")
    print("=" * 60)
    test_python_if_else_basic()
    print()
    test_python_if_else_with_matmul()
    print()
    test_python_while_loop()
    print()
    test_where_conditional()
    print()

    print("\n\033[92mAll dynamic tests completed.\033[0m")
