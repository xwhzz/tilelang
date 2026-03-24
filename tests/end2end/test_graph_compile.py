"""End-to-end coverage for tilelang.compile(..., mode="graph")."""

from __future__ import annotations

import torch
import torch.nn.functional as F

import tilelang


def test_graph_compile_mlp():
    if not torch.cuda.is_available():
        print("CUDA not available, skipping.")
        return

    m = 256
    n = 256
    k = 256

    torch.manual_seed(0)
    a = torch.randn((m, k), device="cuda", dtype=torch.float32)
    b = torch.randn((k, n), device="cuda", dtype=torch.float32)
    bias = torch.randn((n,), device="cuda", dtype=torch.float32)

    def mlp(a_in: torch.Tensor, b_in: torch.Tensor, bias_in: torch.Tensor) -> torch.Tensor:
        return F.relu(a_in @ b_in + bias_in)

    runner = tilelang.compile(
        mlp,
        mode="graph",
        example_args=(a, b, bias),
    )

    assert hasattr(runner, "kernels"), "GraphRunner should expose .kernels"
    assert hasattr(runner, "calls"), "GraphRunner should expose .calls"
    assert len(runner.calls) > 0, "Expected at least one scheduled kernel call"

    out = runner(a, b, bias)
    ref = mlp(a, b, bias)
    torch.testing.assert_close(out, ref, rtol=2e-3, atol=2e-3)
    print("\033[92mtest_graph_compile_mlp: passed.\033[0m")


if __name__ == "__main__":
    test_graph_compile_mlp()
