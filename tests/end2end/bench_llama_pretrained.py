"""LLaMA-2-7B pretrained benchmark: text generation with tilelang vs inductor.

Loads actual pretrained weights, generates text from a real prompt, and
verifies the output is coherent. Tests dynamic shape support via
autoregressive token-by-token generation.

Only the core attention kernel (SDPA) is swapped with TileLang FlashAttention
via torch.library. All other model code stays unchanged.

Usage:
    python tests/end2end/bench_llama_pretrained.py
    python tests/end2end/bench_llama_pretrained.py --model meta-llama/Llama-2-7b-hf
    python tests/end2end/bench_llama_pretrained.py --num-layers 4 --max-new-tokens 64
    python tests/end2end/bench_llama_pretrained.py --correctness-only
"""

from __future__ import annotations

import argparse
import logging
import time

import torch
import torch._dynamo as dynamo
from transformers import AutoTokenizer, LlamaConfig, LlamaForCausalLM
from transformers.models.llama.modeling_llama import ALL_ATTENTION_FUNCTIONS

import tilelang  # noqa: F401
from tilelang.profiler import do_bench
from top import MHAKernel

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# TileLang FlashAttention — torch.library custom op
# ---------------------------------------------------------------------------

_lib = torch.library.Library("tileops_bench", "DEF")
_lib.define("mha(Tensor q, Tensor k, Tensor v, bool is_causal) -> Tensor")

_kernel_cache: dict[tuple, MHAKernel] = {}


def _get_kernel(B, H, S, D, causal, dtype):
    key = (B, H, S, D, causal, str(dtype))
    if key not in _kernel_cache:
        logger.info("Compiling MHAKernel: B=%d H=%d S=%d D=%d causal=%s", B, H, S, D, causal)
        _kernel_cache[key] = MHAKernel(
            batch_size=B, num_heads=H, seq_len=S, head_dim=D, causal=causal, dtype=dtype,
        )
    return _kernel_cache[key]


@torch.library.impl(_lib, "mha", "CUDA")
def _mha_cuda(q, k, v, is_causal):
    B, H, S, D = q.shape
    out = _get_kernel(B, H, S, D, is_causal, q.dtype).forward(
        q.transpose(1, 2).contiguous(),
        k.transpose(1, 2).contiguous(),
        v.transpose(1, 2).contiguous(),
    )
    return out.transpose(1, 2).contiguous()


@torch.library.register_fake("tileops_bench::mha")
def _mha_fake(q, k, v, is_causal):
    return torch.empty(q.shape, dtype=q.dtype, device=q.device)


# ---------------------------------------------------------------------------
# Register as transformers attention function
# ---------------------------------------------------------------------------

def _tileops_attn_forward(
    module, query, key, value, attention_mask,
    dropout=0.0, scaling=None, is_causal=None, **kwargs,
):
    if hasattr(module, "num_key_value_groups") and module.num_key_value_groups > 1:
        from transformers.modeling_attn_mask_utils import repeat_kv
        key = repeat_kv(key, module.num_key_value_groups)
        value = repeat_kv(value, module.num_key_value_groups)

    is_causal = is_causal if is_causal is not None else getattr(module, "is_causal", True)
    is_causal = query.shape[2] > 1 and attention_mask is None and is_causal

    attn_output = torch.ops.tileops_bench.mha(query, key, value, is_causal)
    attn_output = attn_output.transpose(1, 2).contiguous()
    return attn_output, None


ALL_ATTENTION_FUNCTIONS["tileops"] = _tileops_attn_forward


# ---------------------------------------------------------------------------
# Greedy generation (no KV cache — full recompute each step)
# ---------------------------------------------------------------------------

@torch.no_grad()
def greedy_generate(model, input_ids, max_new_tokens=32):
    generated = input_ids.clone()
    for _ in range(max_new_tokens):
        logits = model(generated).logits
        next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        generated = torch.cat([generated, next_token], dim=1)
    return generated


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", type=str, default="meta-llama/Llama-2-7b-hf",
                        help="HuggingFace model name or local path")
    parser.add_argument("--num-layers", type=int, default=0,
                        help="Limit number of layers (0 = all)")
    parser.add_argument("--prompt", type=str,
                        default="The future of artificial intelligence is",
                        help="Input prompt for generation")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--attn", type=str, default="sdpa",
                        choices=["sdpa", "tileops"],
                        help="Attention backend (sdpa=native, tileops=TileLang FA)")
    parser.add_argument("--correctness-only", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    for name in logging.root.manager.loggerDict:
        if "tilelang" not in name:
            logging.getLogger(name).setLevel(logging.WARNING)

    # ---- Load model ----
    attn_impl = args.attn
    print(f"Loading {args.model} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = LlamaForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        attn_implementation=attn_impl,
    )
    if args.num_layers > 0:
        model.model.layers = model.model.layers[:args.num_layers]
    model = model.cuda().eval()

    num_layers = len(model.model.layers)
    params_m = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Model: {args.model} ({num_layers} layers, {params_m:.0f}M params)")
    print(f"Attention: {attn_impl}")

    # ---- Tokenize ----
    encoded = tokenizer(args.prompt, return_tensors="pt")
    input_ids = encoded["input_ids"].cuda()
    seq_len = input_ids.shape[1]
    print(f'\nPrompt: "{args.prompt}"')
    print(f"Prompt tokens: {seq_len}")
    print(f"Max new tokens: {args.max_new_tokens}")

    # Pre-compile MHA kernel for prefill shape (only for tileops)
    if attn_impl == "tileops":
        H = model.config.num_attention_heads
        D = model.config.hidden_size // H
        _get_kernel(1, H, seq_len, D, True, torch.float16)

    # ---- 1. Eager generation ----
    print("\n--- Eager ---")
    t0 = time.time()
    eager_tokens = greedy_generate(model, input_ids, args.max_new_tokens)
    eager_time = time.time() - t0
    eager_text = tokenizer.decode(eager_tokens[0, seq_len:], skip_special_tokens=True)
    print(f'  Generated: "{eager_text}"')
    print(f"  Time: {eager_time:.2f}s")

    # ---- 2. Inductor generation ----
    print("\n--- torch.compile (inductor) ---")
    dynamo.reset()
    inductor_model = torch.compile(model, backend="inductor")
    t0 = time.time()
    inductor_tokens = greedy_generate(inductor_model, input_ids, args.max_new_tokens)
    inductor_time = time.time() - t0
    inductor_text = tokenizer.decode(inductor_tokens[0, seq_len:], skip_special_tokens=True)
    print(f'  Generated: "{inductor_text}"')
    print(f"  Time: {inductor_time:.2f}s (includes compilation)")

    ind_match = (inductor_tokens == eager_tokens).all().item()
    ind_diff = (inductor_tokens != eager_tokens).sum().item()
    print(f"  Token match vs eager: {'PASS' if ind_match else f'FAIL ({ind_diff}/{args.max_new_tokens} differ)'}")

    # ---- 3. TileLang generation ----
    print("\n--- torch.compile (tilelang) ---")
    dynamo.reset()
    tilelang_model = torch.compile(model, backend="tilelang")
    t0 = time.time()
    tilelang_tokens = greedy_generate(tilelang_model, input_ids, args.max_new_tokens)
    tilelang_time = time.time() - t0
    tilelang_text = tokenizer.decode(tilelang_tokens[0, seq_len:], skip_special_tokens=True)
    print(f'  Generated: "{tilelang_text}"')
    print(f"  Time: {tilelang_time:.2f}s (includes compilation)")

    tl_match = (tilelang_tokens == eager_tokens).all().item()
    tl_diff = (tilelang_tokens != eager_tokens).sum().item()
    print(f"  Token match vs eager: {'PASS' if tl_match else f'FAIL ({tl_diff}/{args.max_new_tokens} differ)'}")

    tl_ind_match = (tilelang_tokens == inductor_tokens).all().item()
    tl_ind_diff = (tilelang_tokens != inductor_tokens).sum().item()
    print(f"  Token match vs inductor: {'PASS' if tl_ind_match else f'FAIL ({tl_ind_diff}/{args.max_new_tokens} differ)'}")

    # ---- 4. Prefill-only forward correctness ----
    print("\n--- Prefill forward correctness ---")
    with torch.no_grad():
        eager_logits = model(input_ids).logits

    dynamo.reset()
    tl_fwd = torch.compile(model, backend="tilelang")
    with torch.no_grad():
        tl_logits = tl_fwd(input_ids).logits
    fwd_diff = (tl_logits - eager_logits).abs().max().item()
    print(f"  tilelang vs eager forward max_diff: {fwd_diff:.6f}")
    try:
        torch.testing.assert_close(tl_logits, eager_logits, rtol=1e-2, atol=0.05)
        print("  Forward correctness: PASS")
    except AssertionError as e:
        print(f"  Forward correctness: FAIL ({e})")

    if args.correctness_only:
        return

    # ---- 5. Prefill-only benchmark ----
    print(f"\n--- Prefill benchmark (B=1, S={seq_len}) ---")

    with torch.no_grad():
        model(input_ids)
        eager_ms = do_bench(lambda: model(input_ids))

    dynamo.reset()
    ind_b = torch.compile(model, backend="inductor")
    with torch.no_grad():
        ind_b(input_ids)
        ind_ms = do_bench(lambda: ind_b(input_ids))

    dynamo.reset()
    tl_b = torch.compile(model, backend="tilelang")
    with torch.no_grad():
        tl_b(input_ids)
        tl_ms = do_bench(lambda: tl_b(input_ids))

    print(f"  eager:    {eager_ms:8.3f} ms")
    print(f"  inductor: {ind_ms:8.3f} ms")
    print(f"  tilelang: {tl_ms:8.3f} ms")
    print(f"  tilelang / inductor: {ind_ms / tl_ms:.2f}x")
    print(f"  tilelang / eager:    {eager_ms / tl_ms:.2f}x")


if __name__ == "__main__":
    main()
