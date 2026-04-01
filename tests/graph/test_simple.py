"""Tests for TileLang torch.compile backend."""

import torch
import torch._dynamo
import tilelang  # noqa: F401 — triggers backend registration


def test_elementwise_add():
    """Element-wise addition compiled via TileLang."""

    def fn(x, y):
        return x + y

    x = torch.randn(512, 1024, device="cuda", dtype=torch.float16)
    y = torch.randn(512, 1024, device="cuda", dtype=torch.float16)

    torch._dynamo.reset()
    compiled = torch.compile(fn, backend="tilelang")
    out = compiled(x, y)
    torch.testing.assert_close(out, fn(x, y), atol=1e-3, rtol=1e-3)


def test_matmul():
    """Matmul compiled via TileLang."""

    def fn(x, w):
        return torch.matmul(x, w)

    x = torch.randn(512, 1024, device="cuda", dtype=torch.float16)
    w = torch.randn(1024, 2048, device="cuda", dtype=torch.float16)

    torch._dynamo.reset()
    out = torch.compile(fn, backend="tilelang")(x, w)
    torch.testing.assert_close(out, fn(x, w), atol=1e-2, rtol=1e-2)


def test_gelu():
    """GELU activation compiled via TileLang."""

    x = torch.randn(512, 1024, device="cuda", dtype=torch.float16)

    torch._dynamo.reset()
    fn = lambda x: torch.nn.functional.gelu(x)
    out = torch.compile(fn, backend="tilelang")(x)
    torch.testing.assert_close(out, fn(x), atol=1e-2, rtol=1e-2)


def test_per_op_fallback():
    """Matmul compiled with TileLang, unsupported op falls back to torch."""

    @torch.library.custom_op("test_graph::scale", mutates_args=())
    def custom_scale(x: torch.Tensor, factor: float) -> torch.Tensor:
        return x * factor

    @custom_scale.register_fake
    def _(x, factor):
        return torch.empty_like(x)

    def fn(x, w):
        y = torch.matmul(x, w)
        return custom_scale(y, 2.0)

    x = torch.randn(512, 1024, device="cuda", dtype=torch.float16)
    w = torch.randn(1024, 2048, device="cuda", dtype=torch.float16)

    torch._dynamo.reset()
    out = torch.compile(fn, backend="tilelang")(x, w)
    torch.testing.assert_close(out, fn(x, w), atol=1e-2, rtol=1e-2)


def test_dynamic_shapes():
    """Single compilation, multiple batch sizes."""

    def fn(x, y):
        return x + y

    torch._dynamo.reset()
    compiled = torch.compile(fn, backend="tilelang", dynamic=True)

    for bs in [128, 256, 512]:
        x = torch.randn(bs, 1024, device="cuda", dtype=torch.float16)
        y = torch.randn(bs, 1024, device="cuda", dtype=torch.float16)
        out = compiled(x, y)
        torch.testing.assert_close(out, fn(x, y), atol=1e-3, rtol=1e-3)


def test_dynamic_matmul():
    """Dynamic batch dimension for matmul."""

    def fn(x, w):
        return torch.matmul(x, w)

    w = torch.randn(1024, 2048, device="cuda", dtype=torch.float16)
    torch._dynamo.reset()
    compiled = torch.compile(fn, backend="tilelang", dynamic=True)

    for bs in [128, 256, 512]:
        x = torch.randn(bs, 1024, device="cuda", dtype=torch.float16)
        out = compiled(x, w)
        torch.testing.assert_close(out, fn(x, w), atol=1e-2, rtol=1e-2)


if __name__ == "__main__":
    test_elementwise_add()
    print("PASS: test_elementwise_add")
    test_matmul()
    print("PASS: test_matmul")
    test_gelu()
    print("PASS: test_gelu")
    test_per_op_fallback()
    print("PASS: test_per_op_fallback")
    test_dynamic_shapes()
    print("PASS: test_dynamic_shapes")
    test_dynamic_matmul()
    print("PASS: test_dynamic_matmul")
    print("ALL TESTS PASSED")
