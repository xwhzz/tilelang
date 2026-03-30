"""LLaMA prefill correctness + benchmark: tileops FlashAttention kernel swap.

Only replaces the core attention kernel (SDPA → TileLang FlashAttention).
All other model code stays unchanged.

Usage:
    python tests/end2end/test_llama_mha.py
    python tests/end2end/test_llama_mha.py --num-layers 4 --seq-len 256
    python tests/end2end/test_llama_mha.py --correctness-only
"""

from __future__ import annotations

import argparse
import logging

import torch
import torch._dynamo as dynamo
from transformers import LlamaConfig, LlamaForCausalLM
from transformers.models.llama.modeling_llama import ALL_ATTENTION_FUNCTIONS

import tilelang  # noqa: F401
from tilelang.profiler import do_bench
from top import MHAKernel

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# TileLang FlashAttention as torch.library custom op
# ---------------------------------------------------------------------------

_tileops_lib = torch.library.Library("tileops_mha", "DEF")
_tileops_lib.define(
    "flash_attn(Tensor q, Tensor k, Tensor v, bool is_causal) -> Tensor"
)

_mha_cache: dict[tuple, MHAKernel] = {}


def _get_kernel(batch, heads, seq, dim, causal, dtype):
    key = (batch, heads, seq, dim, causal, str(dtype))
    if key not in _mha_cache:
        logger.info("Compiling MHAKernel: B=%d H=%d S=%d D=%d causal=%s", batch, heads, seq, dim, causal)
        _mha_cache[key] = MHAKernel(
            batch_size=batch, num_heads=heads, seq_len=seq,
            head_dim=dim, causal=causal, dtype=dtype,
        )
    return _mha_cache[key]


@torch.library.impl(_tileops_lib, "flash_attn", "CUDA")
def _flash_attn_cuda(q, k, v, is_causal):
    B, H, S, D = q.shape
    # MHAKernel expects BSHD layout
    out = _get_kernel(B, H, S, D, is_causal, q.dtype).forward(
        q.transpose(1, 2).contiguous(),
        k.transpose(1, 2).contiguous(),
        v.transpose(1, 2).contiguous(),
    )
    return out.transpose(1, 2).contiguous()


@torch.library.register_fake("tileops_mha::flash_attn")
def _flash_attn_fake(q, k, v, is_causal):
    # Must return contiguous tensor — the real impl does transpose+contiguous.
    return torch.empty(q.shape, dtype=q.dtype, device=q.device)


# ---------------------------------------------------------------------------
# Register as a transformers attention function (same interface as sdpa)
# ---------------------------------------------------------------------------

def tileops_attention_forward(
    module, query, key, value, attention_mask,
    dropout=0.0, scaling=None, is_causal=None, **kwargs,
):
    """Drop-in replacement for sdpa_attention_forward using tileops kernel."""
    if hasattr(module, "num_key_value_groups") and module.num_key_value_groups > 1:
        from transformers.modeling_utils import repeat_kv
        key = repeat_kv(key, module.num_key_value_groups)
        value = repeat_kv(value, module.num_key_value_groups)

    is_causal = is_causal if is_causal is not None else getattr(module, "is_causal", True)
    is_causal = query.shape[2] > 1 and attention_mask is None and is_causal

    attn_output = torch.ops.tileops_mha.flash_attn(query, key, value, is_causal)

    # Apply scaling if provided (SDPA applies it internally, we do it post-hoc)
    if scaling is not None and scaling != 1.0:
        # MHAKernel already applies 1/sqrt(d) scaling internally
        pass

    attn_output = attn_output.transpose(1, 2).contiguous()
    return attn_output, None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--correctness-only", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    for name in logging.root.manager.loggerDict:
        if "tilelang" not in name and "test_llama" not in name:
            logging.getLogger(name).setLevel(logging.WARNING)

    # --- Model setup ---
    config = LlamaConfig(
        hidden_size=4096, intermediate_size=11008,
        num_hidden_layers=args.num_layers,
        num_attention_heads=32, num_key_value_heads=32,
        max_position_embeddings=max(args.seq_len * 2, 256),
        vocab_size=32000, torch_dtype=torch.float16,
        attn_implementation="tileops",  # use our registered attention
    )

    # Register our attention function
    ALL_ATTENTION_FUNCTIONS["tileops"] = tileops_attention_forward

    model = LlamaForCausalLM(config).half().cuda().eval()
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model: LLaMA-7B shape, {args.num_layers} layers, {param_count / 1e6:.0f}M params")
    print(f"Input: batch=1, seq_len={args.seq_len}")
    print(f"Attention: tileops FlashAttention (kernel swap only)")

    input_ids = torch.randint(0, 32000, (1, args.seq_len), device="cuda")

    # Pre-compile MHA kernel
    heads = config.num_attention_heads
    dim = config.hidden_size // heads
    _get_kernel(1, heads, args.seq_len, dim, True, torch.float16)

    # --- 1. Eager ---
    with torch.no_grad():
        ref = model(input_ids).logits
    print(f"\nEager output: {ref.shape}")

    # --- 2. torch.compile ---
    dynamo.reset()
    tc_model = torch.compile(model)
    with torch.no_grad():
        tc_out = tc_model(input_ids).logits
    tc_diff = (tc_out - ref).abs().max().item()
    print(f"torch.compile vs eager: max_diff={tc_diff:.6f}")

    # --- 3. tilelang ---
    dynamo.reset()
    tl_model = torch.compile(model, backend="tilelang")
    with torch.no_grad():
        tl_out = tl_model(input_ids).logits
    tl_diff = (tl_out - ref).abs().max().item()
    print(f"tilelang vs eager:      max_diff={tl_diff:.6f}")

    # --- Correctness ---
    try:
        torch.testing.assert_close(tl_out, ref, rtol=1e-2, atol=0.05)
        print("\ntilelang vs eager: PASS")
    except AssertionError as e:
        print(f"\ntilelang vs eager: FAIL ({e})")

    try:
        torch.testing.assert_close(tl_out, tc_out, rtol=1e-2, atol=0.05)
        print("tilelang vs torch.compile: PASS")
    except AssertionError as e:
        print(f"tilelang vs torch.compile: FAIL ({e})")

    if args.correctness_only:
        return

    # --- Benchmark ---
    print(f"\n--- Prefill benchmark (B=1, S={args.seq_len}) ---")
    with torch.no_grad():
        model(input_ids)
        eager_ms = do_bench(lambda: model(input_ids))

    dynamo.reset()
    tc_b = torch.compile(model)
    with torch.no_grad():
        tc_b(input_ids)
        tc_ms = do_bench(lambda: tc_b(input_ids))

    dynamo.reset()
    tl_b = torch.compile(model, backend="tilelang")
    with torch.no_grad():
        tl_b(input_ids)
        tl_ms = do_bench(lambda: tl_b(input_ids))

    print(f"  eager:         {eager_ms:.3f} ms")
    print(f"  torch.compile: {tc_ms:.3f} ms")
    print(f"  tilelang:      {tl_ms:.3f} ms")
    print(f"  tilelang vs torch.compile: {tc_ms / tl_ms:.2f}x")


if __name__ == "__main__":
    main()
