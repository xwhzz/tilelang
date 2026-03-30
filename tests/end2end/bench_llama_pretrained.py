"""LLaMA-2-7B pretrained benchmark: tilelang vs inductor with native SDPA.

SDPA is a permanent extern op in the TileLang backend. All other ops
(matmul, RMSNorm, MLP) go through TIR compilation.

Usage:
    python tests/end2end/bench_llama_pretrained.py
    python tests/end2end/bench_llama_pretrained.py --num-layers 4
    python tests/end2end/bench_llama_pretrained.py --correctness-only
"""

from __future__ import annotations

import argparse
import logging

import torch
import torch._dynamo as dynamo
from transformers import AutoTokenizer, LlamaConfig, LlamaForCausalLM

import tilelang  # noqa: F401
from tilelang.profiler import do_bench

logger = logging.getLogger(__name__)

PROMPT = "The future of artificial intelligence is"
SEED = 42


@torch.no_grad()
def greedy_generate(model, input_ids, max_new_tokens=32):
    generated = input_ids.clone()
    for _ in range(max_new_tokens):
        logits = model(generated).logits
        next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        generated = torch.cat([generated, next_token], dim=1)
    return generated


@torch.no_grad()
def sampled_generate(model, input_ids, max_new_tokens=32, temperature=1.0):
    """Token-by-token generation with multinomial sampling (seed-sensitive)."""
    generated = input_ids.clone()
    for _ in range(max_new_tokens):
        logits = model(generated).logits[:, -1, :] / temperature
        probs = torch.nn.functional.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        generated = torch.cat([generated, next_token], dim=1)
    return generated


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", type=str, default="meta-llama/Llama-2-7b-hf")
    parser.add_argument("--num-layers", type=int, default=0, help="0 = all layers")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--correctness-only", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    for name in logging.root.manager.loggerDict:
        if "tilelang" not in name:
            logging.getLogger(name).setLevel(logging.WARNING)

    print(f"Loading {args.model} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    # Graph break on RoPE to prevent int64 position-encoding from poisoning
    # the whole subgraph. This lets matmul/RMSNorm/MLP compile as TIR while
    # SDPA and RoPE run as extern/eager.
    from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding

    if not hasattr(LlamaRotaryEmbedding, "_orig_forward"):
        LlamaRotaryEmbedding._orig_forward = LlamaRotaryEmbedding.forward

        @torch.compiler.disable
        def rope_forward(self, x, position_ids):
            return LlamaRotaryEmbedding._orig_forward(self, x, position_ids)

        LlamaRotaryEmbedding.forward = rope_forward

    model = LlamaForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float16, attn_implementation="sdpa",
    )
    if args.num_layers > 0:
        model.model.layers = model.model.layers[:args.num_layers]
    model = model.cuda().eval()
    num_layers = len(model.model.layers)
    params_m = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Model: {args.model} ({num_layers} layers, {params_m:.0f}M params)")
    print(f"Attention: native SDPA (permanent extern)")

    encoded = tokenizer(PROMPT, return_tensors="pt")
    input_ids = encoded["input_ids"].cuda()
    seq_len = input_ids.shape[1]
    print(f'\nPrompt: "{PROMPT}" ({seq_len} tokens)')

    # Eager reference
    torch.manual_seed(SEED)
    with torch.no_grad():
        eager_logits = model(input_ids).logits
    eager_tokens = greedy_generate(model, input_ids, args.max_new_tokens)
    eager_text = tokenizer.decode(eager_tokens[0, seq_len:], skip_special_tokens=True)
    print(f'\nEager: "{eager_text}"')

    # TileLang
    dynamo.reset()
    torch.manual_seed(SEED)
    tl_model = torch.compile(model, backend="tilelang")
    with torch.no_grad():
        tl_logits = tl_model(input_ids).logits
    tl_tokens = greedy_generate(tl_model, input_ids, args.max_new_tokens)
    tl_text = tokenizer.decode(tl_tokens[0, seq_len:], skip_special_tokens=True)
    print(f'TileLang: "{tl_text}"')

    # Forward correctness
    fwd_diff = (tl_logits - eager_logits).abs().max().item()
    print(f"\nForward max diff: {fwd_diff:.6f}")
    try:
        torch.testing.assert_close(tl_logits, eager_logits, rtol=1e-2, atol=0.05)
        print("Forward correctness: PASS")
    except AssertionError as e:
        print(f"Forward correctness: FAIL ({e})")

    # Token match
    token_match = (tl_tokens == eager_tokens).all().item()
    n_diff = (tl_tokens != eager_tokens).sum().item()
    print(f"Token match: {'PASS' if token_match else f'FAIL ({n_diff}/{args.max_new_tokens} differ)'}")

    # Negative control 1: random-weight model → different from pretrained
    from transformers import LlamaConfig as _LC
    rand_config = _LC(
        hidden_size=model.config.hidden_size,
        intermediate_size=model.config.intermediate_size,
        num_hidden_layers=1,
        num_attention_heads=model.config.num_attention_heads,
        num_key_value_heads=model.config.num_key_value_heads,
        max_position_embeddings=model.config.max_position_embeddings,
        vocab_size=model.config.vocab_size,
        torch_dtype=torch.float16,
        attn_implementation="sdpa",
    )
    torch.manual_seed(99)
    rand_model = LlamaForCausalLM(rand_config).half().cuda().eval()
    with torch.no_grad():
        rand_logits = rand_model(input_ids).logits
    rand_diff = (rand_logits - eager_logits).abs().max().item()
    print(f"Negative control (random weights): diff={rand_diff:.2f} {'PASS (differs)' if rand_diff > 1.0 else 'FAIL (too similar)'}")
    del rand_model
    torch.cuda.empty_cache()

    # Negative control 2: different seeds → different sampled tokens
    torch.manual_seed(SEED)
    tokens_seed_a = sampled_generate(model, input_ids, max_new_tokens=min(8, args.max_new_tokens))
    torch.manual_seed(SEED + 1)
    tokens_seed_b = sampled_generate(model, input_ids, max_new_tokens=min(8, args.max_new_tokens))
    seed_match = (tokens_seed_a == tokens_seed_b).all().item()
    print(f"Negative control (diff seeds): {'PASS (tokens differ)' if not seed_match else 'FAIL (tokens identical)'}")

    # Trace composition
    from tilelang.torch_compile.api import get_compilation_traces
    traces = get_compilation_traces()
    for tr in traces:
        if tr.n_compiled is not None:
            print(f"  Trace: compiled={tr.n_compiled}, extern={tr.n_extern}, fallback_eager={tr.n_fallback_eager}")

    if args.correctness_only:
        return

    # Benchmark
    print(f"\n--- Prefill (B=1, S={seq_len}) ---")
    with torch.no_grad():
        model(input_ids)
        eager_ms = do_bench(lambda: model(input_ids))

    dynamo.reset()
    ind_model = torch.compile(model, backend="inductor")
    with torch.no_grad():
        ind_model(input_ids)
        ind_ms = do_bench(lambda: ind_model(input_ids))

    dynamo.reset()
    tl_bench = torch.compile(model, backend="tilelang")
    with torch.no_grad():
        tl_bench(input_ids)
        tl_ms = do_bench(lambda: tl_bench(input_ids))

    print(f"  eager:    {eager_ms:8.3f} ms")
    print(f"  inductor: {ind_ms:8.3f} ms")
    print(f"  tilelang: {tl_ms:8.3f} ms")
    print(f"  tilelang / inductor: {ind_ms / tl_ms:.2f}x")


if __name__ == "__main__":
    main()
