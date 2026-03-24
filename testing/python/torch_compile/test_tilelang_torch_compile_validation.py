import pytest
import torch

import tilelang
import tilelang.testing


@tilelang.jit(mode="graph")
def add_one(x: torch.Tensor) -> torch.Tensor:
    return x + 1


@tilelang.testing.requires_cuda
def test_graph_jit_rejects_noncontiguous_inputs_before_compile():
    x = torch.randn((8, 16), device="cuda", dtype=torch.float32).transpose(0, 1)

    with pytest.raises(ValueError, match="contiguous CUDA tensors"):
        add_one(x)


@tilelang.testing.requires_cuda
def test_graph_runner_rejects_noncontiguous_inputs():
    compiled = add_one.compile(torch.randn((16, 8), device="cuda", dtype=torch.float32))
    x = torch.randn((8, 16), device="cuda", dtype=torch.float32).transpose(0, 1)

    with pytest.raises(ValueError, match="contiguous CUDA tensors"):
        compiled(x)
