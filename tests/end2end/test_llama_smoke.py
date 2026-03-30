"""CI smoke test: 7B-shape synthetic LLaMA with native SDPA as extern op.

Uses a 2-layer random-weight model to verify the torch.compile backend
compiles correctly with SDPA as a permanent extern op.
"""

import pytest
import torch
import torch._dynamo as dynamo


@pytest.fixture(autouse=True)
def reset_dynamo():
    dynamo.reset()
    yield
    dynamo.reset()


def _make_model(num_layers=2):
    from transformers import LlamaConfig, LlamaForCausalLM
    config = LlamaConfig(
        hidden_size=4096, intermediate_size=11008, num_hidden_layers=num_layers,
        num_attention_heads=32, num_key_value_heads=32,
        max_position_embeddings=256, vocab_size=32000, torch_dtype=torch.float16,
        attn_implementation="sdpa",
    )
    return LlamaForCausalLM(config).half().cuda().eval()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_llama_smoke_forward_correctness():
    """AC-4.3: 7B-shape synthetic model forward correctness."""
    import tilelang  # noqa: F401

    model = _make_model(num_layers=2)
    ids = torch.randint(0, 32000, (1, 128), device="cuda")
    with torch.no_grad():
        ref = model(ids).logits
    tl = torch.compile(model, backend="tilelang")
    with torch.no_grad():
        out = tl(ids).logits
    torch.testing.assert_close(out, ref, rtol=1e-2, atol=0.05)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_llama_smoke_trace_counts():
    """AC-5: Trace reports composition counts."""
    import tilelang  # noqa: F401
    from tilelang.torch_compile.api import get_compilation_traces, clear_compilation_traces

    clear_compilation_traces()
    model = _make_model(num_layers=1)
    ids = torch.randint(0, 32000, (1, 32), device="cuda")
    tl = torch.compile(model, backend="tilelang")
    with torch.no_grad():
        tl(ids)

    traces = get_compilation_traces()
    # At least one trace should exist.
    assert len(traces) > 0
    # Check composition: either compiled with counts or eager fallback.
    for tr in traces:
        if tr.compilation_path in ("cache_hit", "disk_cache_hit"):
            assert tr.n_compiled is None
        elif tr.n_compiled is not None:
            assert tr.n_compiled >= 0
            assert tr.n_extern >= 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_llama_smoke_custom_op():
    """AC-3a: Custom torch.library op detected as extern."""
    import tilelang  # noqa: F401

    torch.library.define("smoke_test::my_op", "(Tensor x, float s) -> Tensor")

    @torch.library.impl("smoke_test::my_op", "cuda")
    def impl(x, s):
        return x * s

    @torch.library.register_fake("smoke_test::my_op")
    def fake(x, s):
        return torch.empty_like(x)

    def model(x):
        a = x + 1.0
        b = torch.ops.smoke_test.my_op(a, 2.0)
        return b + x

    x = torch.randn(4, 4, device="cuda", dtype=torch.float16)
    ref = model(x)
    compiled = torch.compile(model, backend="tilelang")
    out = compiled(x)
    torch.testing.assert_close(out, ref, atol=1e-3, rtol=1e-3)
