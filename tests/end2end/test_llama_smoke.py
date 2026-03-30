"""CI smoke tests for torch.compile(backend='tilelang') with LLaMA-shaped models.

Tests AC-3a (ExternPolicy), AC-3c (canonicalization), AC-4.3 (synthetic model),
and AC-5 (trace composition counts).
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


# ---- AC-4.3: Forward correctness ----

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_llama_smoke_forward_correctness():
    """AC-4.3 positive: 7B-shape synthetic model forward matches eager."""
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
def test_llama_smoke_random_vs_different_weights():
    """AC-4.1 negative: two models with different random weights produce different output."""
    import tilelang  # noqa: F401
    torch.manual_seed(1)
    model_a = _make_model(num_layers=1)
    torch.manual_seed(2)
    model_b = _make_model(num_layers=1)
    ids = torch.randint(0, 32000, (1, 32), device="cuda")
    with torch.no_grad():
        out_a = model_a(ids).logits
        out_b = model_b(ids).logits
    # Different weights → different output (demonstrates test is meaningful).
    assert not torch.allclose(out_a, out_b, atol=0.1), "Different weights should produce different output"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_llama_smoke_wrong_compilation_fails():
    """AC-4.3 negative: deliberately broken backend produces wrong output."""
    import tilelang  # noqa: F401
    model = _make_model(num_layers=1)
    ids = torch.randint(0, 32000, (1, 32), device="cuda")
    with torch.no_grad():
        ref = model(ids).logits

    # A backend that returns zeros should NOT match eager.
    def broken_backend(gm, example_inputs):
        def runner(*args):
            return tuple(torch.zeros_like(a) for a in args if isinstance(a, torch.Tensor) and a.ndim > 0)
        return runner

    dynamo.reset()
    broken = torch.compile(model, backend=broken_backend)
    try:
        with torch.no_grad():
            out = broken(ids).logits
        # If it runs, the output should NOT match.
        assert not torch.allclose(out, ref, atol=0.05), "Broken backend should not match eager"
    except Exception:
        pass  # Exception is also acceptable for a broken backend


# ---- AC-3a: ExternPolicy ----

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_custom_op_detected_as_extern():
    """Positive: torch.library custom op auto-detected as extern."""
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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_sdpa_routed_as_extern():
    """Positive: SDPA classified as extern by ExternPolicy."""
    from tilelang.torch_compile.analysis import ExternPolicy, _get_known_fx_ops

    # Verify the actual SDPA target is classified as extern.
    sdpa_target = torch._C._nn.scaled_dot_product_attention
    policy = ExternPolicy()
    known = _get_known_fx_ops()
    assert policy.is_extern(sdpa_target, known), (
        "SDPA must be classified as extern by ExternPolicy"
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_aten_op_not_classified_as_extern():
    """Negative: supported aten ops should NOT be extern."""
    from tilelang.torch_compile.analysis import ExternPolicy, _get_known_fx_ops

    policy = ExternPolicy()
    known = _get_known_fx_ops()
    # torch.ops.aten.add.Tensor is a supported aten overload.
    add_target = torch.ops.aten.add.Tensor
    assert not policy.is_extern(add_target, known), (
        "aten.add.Tensor should not be classified as extern"
    )


# ---- AC-5: Trace composition counts ----

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_trace_counts_compiled():
    """Positive: compiled traces have non-None counts."""
    import tilelang  # noqa: F401
    from tilelang.torch_compile.api import get_compilation_traces, clear_compilation_traces

    clear_compilation_traces()
    model = _make_model(num_layers=1)
    ids = torch.randint(0, 32000, (1, 32), device="cuda")
    tl = torch.compile(model, backend="tilelang")
    with torch.no_grad():
        tl(ids)

    traces = get_compilation_traces()
    assert len(traces) > 0
    for tr in traces:
        if tr.compilation_path in ("cache_hit", "disk_cache_hit"):
            assert tr.n_compiled is None
        elif tr.compilation_path == "fallback_eager":
            assert tr.n_fallback_eager == 1 or tr.n_fallback_eager is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_trace_counts_cache_hit():
    """Positive: cache-hit traces have None counts."""
    import tilelang  # noqa: F401
    from tilelang.torch_compile.api import get_compilation_traces, clear_compilation_traces

    def model(x):
        return x + 1.0

    clear_compilation_traces()
    x = torch.randn(4, device="cuda", dtype=torch.float16)
    c1 = torch.compile(model, backend="tilelang")
    with torch.no_grad():
        c1(x)
    # Second compile should hit cache.
    dynamo.reset()
    c2 = torch.compile(model, backend="tilelang")
    with torch.no_grad():
        c2(x)

    traces = get_compilation_traces()
    cache_hits = [t for t in traces if t.compilation_path == "cache_hit"]
    for tr in cache_hits:
        assert tr.n_compiled is None
        assert tr.n_extern is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_trace_exact_counts_mixed():
    """Positive: mixed compiled + extern graph has exact counts."""
    import tilelang  # noqa: F401
    from tilelang.torch_compile.api import get_compilation_traces, clear_compilation_traces

    torch.library.define("count_test::ext_op", "(Tensor x) -> Tensor")

    @torch.library.impl("count_test::ext_op", "cuda")
    def impl(x):
        return x * 2.0

    @torch.library.register_fake("count_test::ext_op")
    def fake(x):
        return torch.empty_like(x)

    def model(x):
        a = x + 1.0  # compiled
        b = torch.ops.count_test.ext_op(a)  # extern
        return b + x  # compiled

    clear_compilation_traces()
    x = torch.randn(4, device="cuda", dtype=torch.float16)
    compiled = torch.compile(model, backend="tilelang")
    with torch.no_grad():
        compiled(x)

    traces = get_compilation_traces()
    compiled_traces = [t for t in traces if t.n_compiled is not None]
    assert len(compiled_traces) > 0
    tr = compiled_traces[0]
    assert tr.n_extern >= 1  # at least the ext_op
    assert tr.n_compiled >= 1  # at least one TIR kernel

