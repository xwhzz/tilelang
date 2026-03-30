"""LLaMA-2 prefill benchmark: tilelang vs inductor, with tileops FlashAttention.

Only the core attention kernel (SDPA) is replaced with TileLang FlashAttention
via torch.library registration. All other model code (QKV projections, RoPE,
RMSNorm, MLP, output projection) stays unchanged.

For the tilelang backend, the custom op flows through the extern-op mechanism
(no graph break). For inductor, it goes through the standard custom-op path.

Usage:
    python tests/end2end/bench_llama_tileops_attn.py
    python tests/end2end/bench_llama_tileops_attn.py --num-layers 4 --seq-len 256
    python tests/end2end/bench_llama_tileops_attn.py --correctness-only
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
# TileLang FlashAttention — torch.library custom op
# ---------------------------------------------------------------------------

_lib = torch.library.Library("tileops_attn", "DEF")
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


@torch.library.register_fake("tileops_attn::mha")
def _mha_fake(q, k, v, is_causal):
    return torch.empty(q.shape, dtype=q.dtype, device=q.device)


# ---------------------------------------------------------------------------
# Register as transformers attention function
# ---------------------------------------------------------------------------

def _tileops_attention_forward(
    module, query, key, value, attention_mask,
    dropout=0.0, scaling=None, is_causal=None, **kwargs,
):
    if hasattr(module, "num_key_value_groups") and module.num_key_value_groups > 1:
        from transformers.modeling_attn_mask_utils import repeat_kv
        key = repeat_kv(key, module.num_key_value_groups)
        value = repeat_kv(value, module.num_key_value_groups)

    is_causal = is_causal if is_causal is not None else getattr(module, "is_causal", True)
    is_causal = query.shape[2] > 1 and attention_mask is None and is_causal

    attn_output = torch.ops.tileops_attn.mha(query, key, value, is_causal)
    attn_output = attn_output.transpose(1, 2).contiguous()
    return attn_output, None


ALL_ATTENTION_FUNCTIONS["tileops"] = _tileops_attention_forward


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--correctness-only", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    for name in logging.root.manager.loggerDict:
        if "tilelang" not in name:
            logging.getLogger(name).setLevel(logging.WARNING)

    config = LlamaConfig(
        hidden_size=4096,
        intermediate_size=11008,
        num_hidden_layers=args.num_layers,
        num_attention_heads=32,
        num_key_value_heads=32,
        max_position_embeddings=max(args.seq_len * 2, 256),
        vocab_size=32000,
        torch_dtype=torch.float16,
        attn_implementation="tileops",
    )
    model = LlamaForCausalLM(config).half().cuda().eval()
    params_m = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Model: LLaMA-2-7B shape, {args.num_layers} layers, {params_m:.0f}M params")
    print(f"Input: B={args.batch}, S={args.seq_len}")
    print(f"Attention: tileops FlashAttention (kernel-only swap via torch.library)")

    input_ids = torch.randint(0, config.vocab_size, (args.batch, args.seq_len), device="cuda")

    # Pre-compile the MHA kernel
    H = config.num_attention_heads
    D = config.hidden_size // H
    _get_kernel(args.batch, H, args.seq_len, D, True, torch.float16)

    # ---- Eager ----
    with torch.no_grad():
        eager_logits = model(input_ids).logits
    print(f"\nEager: {eager_logits.shape}")

    # ---- Inductor ----
    dynamo.reset()
    inductor_model = torch.compile(model, backend="inductor")
    with torch.no_grad():
        inductor_logits = inductor_model(input_ids).logits
    ind_diff = (inductor_logits - eager_logits).abs().max().item()
    print(f"inductor vs eager:  max_diff={ind_diff:.6f}")

    # ---- TileLang ----
    dynamo.reset()
    tilelang_model = torch.compile(model, backend="tilelang")
    with torch.no_grad():
        tilelang_logits = tilelang_model(input_ids).logits
    tl_diff = (tilelang_logits - eager_logits).abs().max().item()
    print(f"tilelang vs eager:  max_diff={tl_diff:.6f}")

    # ---- Correctness ----
    all_pass = True
    for name, logits in [("inductor", inductor_logits), ("tilelang", tilelang_logits)]:
        try:
            torch.testing.assert_close(logits, eager_logits, rtol=1e-2, atol=0.05)
            print(f"  {name} vs eager: PASS")
        except AssertionError as e:
            print(f"  {name} vs eager: FAIL ({e})")
            all_pass = False

    try:
        torch.testing.assert_close(tilelang_logits, inductor_logits, rtol=1e-2, atol=0.05)
        print(f"  tilelang vs inductor: PASS")
    except AssertionError as e:
        print(f"  tilelang vs inductor: FAIL ({e})")
        all_pass = False

    if args.correctness_only:
        return

    # ---- Benchmark ----
    print(f"\n--- Prefill latency (B={args.batch}, S={args.seq_len}) ---")

    with torch.no_grad():
        model(input_ids)
        eager_ms = do_bench(lambda: model(input_ids))

    dynamo.reset()
    ind_bench = torch.compile(model, backend="inductor")
    with torch.no_grad():
        ind_bench(input_ids)
        ind_ms = do_bench(lambda: ind_bench(input_ids))

    dynamo.reset()
    tl_bench = torch.compile(model, backend="tilelang")
    with torch.no_grad():
        tl_bench(input_ids)
        tl_ms = do_bench(lambda: tl_bench(input_ids))

    print(f"  eager:    {eager_ms:8.3f} ms")
    print(f"  inductor: {ind_ms:8.3f} ms")
    print(f"  tilelang: {tl_ms:8.3f} ms")
    print(f"  tilelang / inductor: {ind_ms / tl_ms:.2f}x")
    print(f"  tilelang / eager:    {eager_ms / tl_ms:.2f}x")


if __name__ == "__main__":
    main()
