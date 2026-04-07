"""Test Relax VM build path for TileLang torch.compile backend."""

import torch
import torch._dynamo
import tilelang  # noqa: F401 — triggers backend registration
from tilelang.graph import backend_config


def _run_vm(fn, *inputs, atol=1e-3, rtol=1e-3):
    """Compile fn with VM backend and check against eager."""
    torch._dynamo.reset()
    backend_config.use_vm = True
    try:
        compiled = torch.compile(fn, backend="tilelang")
        out = compiled(*inputs)
        expected = fn(*inputs)
        torch.testing.assert_close(out, expected, atol=atol, rtol=rtol)
        return out
    finally:
        backend_config.use_vm = False


def test_vm_elementwise_add():
    """Element-wise add through VM."""
    x = torch.randn(512, 1024, device="cuda", dtype=torch.float16)
    y = torch.randn(512, 1024, device="cuda", dtype=torch.float16)
    _run_vm(lambda x, y: x + y, x, y)


def test_vm_matmul():
    """Matmul through VM."""
    x = torch.randn(128, 256, device="cuda", dtype=torch.float16)
    w = torch.randn(256, 512, device="cuda", dtype=torch.float16)
    _run_vm(lambda x, w: x @ w, x, w, atol=1e-2, rtol=1e-2)


def test_vm_gelu():
    """GELU through VM."""
    x = torch.randn(512, 1024, device="cuda", dtype=torch.float16)
    _run_vm(lambda x: torch.nn.functional.gelu(x), x, atol=1e-2, rtol=1e-2)


def test_vm_multi_call():
    """Verify output cloning prevents storage corruption across calls."""
    x1 = torch.randn(64, 128, device="cuda", dtype=torch.float16)
    y1 = torch.randn(64, 128, device="cuda", dtype=torch.float16)
    x2 = torch.randn(64, 128, device="cuda", dtype=torch.float16)
    y2 = torch.randn(64, 128, device="cuda", dtype=torch.float16)

    fn = lambda x, y: x + y

    torch._dynamo.reset()
    backend_config.use_vm = True
    try:
        compiled = torch.compile(fn, backend="tilelang")
        out1 = compiled(x1, y1)
        out2 = compiled(x2, y2)
        # out1 should NOT be corrupted by second call
        torch.testing.assert_close(out1, x1 + y1, atol=1e-3, rtol=1e-3)
        torch.testing.assert_close(out2, x2 + y2, atol=1e-3, rtol=1e-3)
    finally:
        backend_config.use_vm = False


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)

    print("test_vm_elementwise_add...")
    test_vm_elementwise_add()
    print("  PASS")

    print("test_vm_matmul...")
    test_vm_matmul()
    print("  PASS")

    print("test_vm_gelu...")
    test_vm_gelu()
    print("  PASS")

    print("test_vm_multi_call...")
    test_vm_multi_call()
    print("  PASS")

    print("\nAll VM tests passed!")
