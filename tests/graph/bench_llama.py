"""Benchmark: LLaMA-2-7B TileLang vs Inductor backend."""

import time
import torch
import torch._dynamo
import tilelang  # noqa: F401
from tilelang.graph.cache import clear_cache


def bench_generate(model, input_ids, tokenizer, backend, warmup=2, repeat=5, max_new_tokens=128):
    torch._dynamo.reset()
    if backend == "eager":
        compiled = model
    else:
        compiled = torch.compile(model, backend=backend)

    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            compiled.generate(input_ids, max_new_tokens=max_new_tokens, do_sample=False)
    torch.cuda.synchronize()

    # Benchmark
    times = []
    tokens_out = None
    with torch.no_grad():
        for _ in range(repeat):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            out = compiled.generate(input_ids, max_new_tokens=max_new_tokens, do_sample=False)
            torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)
            tokens_out = out

    avg = sum(times) / len(times)
    num_new_tokens = tokens_out.shape[1] - input_ids.shape[1]
    tok_per_sec = num_new_tokens / avg
    text = tokenizer.decode(tokens_out[0], skip_special_tokens=True)
    return avg, tok_per_sec, text, times


def main():
    from transformers import LlamaForCausalLM, AutoTokenizer

    model_name = "meta-llama/Llama-2-7b-hf"
    max_new_tokens = 128

    print(f"Model: {model_name}")
    print(f"Max new tokens: {max_new_tokens}")
    print(f"GPU: {torch.cuda.get_device_name()}")
    print()

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = LlamaForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16).cuda().eval()

    prompt = "The capital of France is"
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(
        "cuda", dtype=torch.int32)
    print(f"Prompt: \"{prompt}\" ({input_ids.shape[1]} tokens)\n")

    results = {}
    for backend in ["eager", "inductor", "tilelang"]:
        if backend == "tilelang":
            clear_cache()
        print(f"--- {backend} ---")
        avg, tok_s, text, times = bench_generate(
            model, input_ids, tokenizer, backend,
            warmup=2, repeat=5, max_new_tokens=max_new_tokens)
        results[backend] = (avg, tok_s, text)
        print(f"  Avg time:   {avg:.3f}s")
        print(f"  Tokens/sec: {tok_s:.1f}")
        print(f"  All times:  {[f'{t:.3f}' for t in times]}")
        print(f"  Output:     {text[:100]}...")
        print()

    # Compare
    print("=== Summary ===")
    eager_tok = results["eager"][1]
    for backend, (avg, tok_s, text) in results.items():
        speedup = tok_s / eager_tok if eager_tok > 0 else 0
        print(f"{backend:12s}  {tok_s:7.1f} tok/s  {speedup:.2f}x vs eager")

    # Verify outputs match
    if results["tilelang"][2] == results["eager"][2]:
        print("\nTileLang output matches eager: YES")
    else:
        print("\nTileLang output matches eager: NO")
        print(f"  Eager:    {results['eager'][2][:150]}")
        print(f"  TileLang: {results['tilelang'][2][:150]}")

    if results["inductor"][2] == results["eager"][2]:
        print("Inductor output matches eager: YES")
    else:
        print("Inductor output matches eager: NO")


if __name__ == "__main__":
    main()
