"""Tests for dynamic shapes and dynamic control flow with torch.compile(backend="tilelang")."""

from __future__ import annotations

import torch
import torch.nn.functional as F
import torch._dynamo

import tilelang  # noqa: F401  (triggers backend registration)
from tilelang.torch_compile.api import (
    clear_compilation_traces,
    clear_subgraph_cache,
    get_compilation_traces,
)


# ===========================================================================
# Shape Recompilation Tests (Dynamo handles guard-based recompilation)
# ===========================================================================

def test_shape_recompilation():
    """Different shapes trigger Dynamo recompilation automatically."""
    if not torch.cuda.is_available():
        print("CUDA not available, skipping.")
        return

    torch._dynamo.reset()

    @torch.compile(backend="tilelang")
    def add_relu(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return F.relu(a + b)

    for N in [128, 256, 512]:
        a = torch.randn(N, device="cuda")
        b = torch.randn(N, device="cuda")
        out = add_relu(a, b)
        ref = F.relu(a + b)
        torch.testing.assert_close(out, ref, rtol=1e-3, atol=1e-3)
        print(f"  shape ({N},) passed")

    print("\033[92mtest_shape_recompilation: passed.\033[0m")


def test_shape_recompilation_matmul():
    """Different batch sizes with matmul."""
    if not torch.cuda.is_available():
        print("CUDA not available, skipping.")
        return

    torch._dynamo.reset()

    K, N = 512, 512

    @torch.compile(backend="tilelang")
    def matmul_fn(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return a @ b

    b = torch.randn(K, N, device="cuda")
    for M in [4, 16, 64, 256]:
        a = torch.randn(M, K, device="cuda")
        out = matmul_fn(a, b)
        ref = a @ b
        torch.testing.assert_close(out, ref, rtol=1e-2, atol=1e-2)
        print(f"  shape ({M}, {K}) @ ({K}, {N}) passed")

    print("\033[92mtest_shape_recompilation_matmul: passed.\033[0m")


def test_shape_recompilation_mlp():
    """MLP with different batch dimensions."""
    if not torch.cuda.is_available():
        print("CUDA not available, skipping.")
        return

    torch._dynamo.reset()

    dim = 256
    torch.manual_seed(0)
    w1 = torch.randn(dim, dim, device="cuda", dtype=torch.float32)
    w2 = torch.randn(dim, dim, device="cuda", dtype=torch.float32)

    @torch.compile(backend="tilelang")
    def mlp(x, w1_in, w2_in):
        h = x @ w1_in
        h = F.relu(h)
        return h @ w2_in

    for B in [4, 16, 64]:
        x = torch.randn(B, dim, device="cuda", dtype=torch.float32)
        out = mlp(x, w1, w2)
        ref = F.relu(x @ w1) @ w2
        torch.testing.assert_close(out, ref, rtol=2e-3, atol=2e-3)
        print(f"  batch={B} passed")

    print("\033[92mtest_shape_recompilation_mlp: passed.\033[0m")


# ===========================================================================
# Symbolic Shape Tests (compile once, run with any shape)
# ===========================================================================

def test_symbolic_shapes_add():
    """torch.compile(dynamic=True) should compile once and reuse for multiple sizes."""
    if not torch.cuda.is_available():
        print("CUDA not available, skipping.")
        return

    torch._dynamo.reset()
    clear_subgraph_cache()
    clear_compilation_traces()

    @torch.compile(backend="tilelang", dynamic=True)
    def add_fn(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return a + b

    for N in [128, 256, 512, 1024]:
        a = torch.randn(N, device="cuda")
        b = torch.randn(N, device="cuda")
        out = add_fn(a, b)
        ref = a + b
        torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-5)
        print(f"  shape ({N},) passed")

    paths = [trace.compilation_path for trace in get_compilation_traces()]
    assert paths == ["dynamo_symbolic"], paths
    print("\033[92mtest_symbolic_shapes_add: passed.\033[0m")


def test_symbolic_shapes_mlp():
    """MLP with explicit dynamic=True — should use symbolic runner."""
    if not torch.cuda.is_available():
        print("CUDA not available, skipping.")
        return

    torch._dynamo.reset()

    dim = 256
    torch.manual_seed(42)
    w1 = torch.randn(dim, dim, device="cuda", dtype=torch.float32)
    w2 = torch.randn(dim, dim, device="cuda", dtype=torch.float32)

    @torch.compile(backend="tilelang", dynamic=True)
    def mlp(x, w1_in, w2_in):
        h = x @ w1_in
        h = F.relu(h)
        return h @ w2_in

    for B in [1, 4, 16, 64, 128]:
        x = torch.randn(B, dim, device="cuda", dtype=torch.float32)
        out = mlp(x, w1, w2)
        ref = F.relu(x @ w1) @ w2
        torch.testing.assert_close(out, ref, rtol=2e-3, atol=2e-3)
        print(f"  batch={B} passed")

    print("\033[92mtest_symbolic_shapes_mlp: passed.\033[0m")


# ===========================================================================
# Dynamic Control Flow Tests
# ===========================================================================

def test_python_if_else_basic():
    """Python if/else on tensor pred — Dynamo handles via graph breaks."""
    if not torch.cuda.is_available():
        print("CUDA not available, skipping.")
        return

    torch._dynamo.reset()

    @torch.compile(backend="tilelang")
    def cond_fn(pred, x: torch.Tensor) -> torch.Tensor:
        if pred:
            return x * 2.0
        else:
            return x * 0.5

    x = torch.randn(256, device="cuda")

    out = cond_fn(True, x)
    torch.testing.assert_close(out, x * 2.0, rtol=1e-5, atol=1e-5)

    out = cond_fn(False, x)
    torch.testing.assert_close(out, x * 0.5, rtol=1e-5, atol=1e-5)

    print("\033[92mtest_python_if_else_basic: passed.\033[0m")


def test_where_conditional():
    """Element-wise torch.where (data-parallel, no control flow)."""
    if not torch.cuda.is_available():
        print("CUDA not available, skipping.")
        return

    torch._dynamo.reset()

    @torch.compile(backend="tilelang")
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
    print("Shape Recompilation Tests")
    print("=" * 60)
    test_shape_recompilation()
    print()
    test_shape_recompilation_matmul()
    print()
    test_shape_recompilation_mlp()
    print()

    print("=" * 60)
    print("Symbolic Shape Tests")
    print("=" * 60)
    test_symbolic_shapes_add()
    print()
    test_symbolic_shapes_mlp()
    print()

    print("=" * 60)
    print("Dynamic Control Flow Tests")
    print("=" * 60)
    test_python_if_else_basic()
    print()
    test_where_conditional()
    print()

    print("\n\033[92mAll dynamic tests completed.\033[0m")
