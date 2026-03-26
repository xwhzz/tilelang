"""End-to-end coverage for torch.compile(backend="tilelang")."""

from __future__ import annotations

import torch
import torch.nn.functional as F
import torch._dynamo

import tilelang  # noqa: F401  (triggers backend registration)


def test_graph_compile_mlp():
    if not torch.cuda.is_available():
        print("CUDA not available, skipping.")
        return

    torch._dynamo.reset()

    m = 256
    n = 256
    k = 256

    torch.manual_seed(0)
    a = torch.randn((m, k), device="cuda", dtype=torch.float32)
    b = torch.randn((k, n), device="cuda", dtype=torch.float32)
    bias = torch.randn((n,), device="cuda", dtype=torch.float32)

    def mlp(a_in: torch.Tensor, b_in: torch.Tensor, bias_in: torch.Tensor) -> torch.Tensor:
        return F.relu(a_in @ b_in + bias_in)

    compiled = torch.compile(mlp, backend="tilelang")

    out = compiled(a, b, bias)
    ref = mlp(a, b, bias)
    torch.testing.assert_close(out, ref, rtol=2e-3, atol=2e-3)
    print("\033[92mtest_graph_compile_mlp: passed.\033[0m")


if __name__ == "__main__":
    test_graph_compile_mlp()
