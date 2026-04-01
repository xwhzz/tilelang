"""Latency breakdown: LLaMA-2-7B TileLang vs Inductor.

Measures prefill and decode latency separately.
Uses dynamic=True so the model compiles once for all sequence lengths.
"""

import time
import torch
import torch._dynamo
import tilelang  # noqa: F401
from tilelang.graph.cache import clear_cache


def bench_prefill(model, input_ids, warmup=3, repeat=10):
    torch.cuda.synchronize()
    with torch.no_grad():
        for _ in range(warmup):
            model(input_ids)
    torch.cuda.synchronize()

    times = []
    with torch.no_grad():
        for _ in range(repeat):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            model(input_ids)
            torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)
    return times


def bench_decode(model, input_ids, num_steps=128, warmup=1, repeat=3):
    torch.cuda.synchronize()
    with torch.no_grad():
        for _ in range(warmup):
            model.generate(input_ids, max_new_tokens=num_steps, do_sample=False)
    torch.cuda.synchronize()

    times = []
    with torch.no_grad():
        for _ in range(repeat):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            out = model.generate(input_ids, max_new_tokens=num_steps, do_sample=False)
            torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)
    num_new = out.shape[1] - input_ids.shape[1]
    return times, num_new


def fmt(times):
    avg = sum(times) / len(times)
    mn = min(times)
    return f"avg={avg*1000:.1f}ms  min={mn*1000:.1f}ms"


def main():
    from transformers import LlamaForCausalLM, AutoTokenizer

    model_name = "meta-llama/Llama-2-7b-hf"
    print(f"Model: {model_name}")
    print(f"GPU:   {torch.cuda.get_device_name()}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = LlamaForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16).cuda().eval()

    # Compile once with dynamic shapes
    backends = {}
    backends["eager"] = model

    torch._dynamo.reset()
    backends["inductor"] = torch.compile(model, backend="inductor", dynamic=True)

    torch._dynamo.reset()
    clear_cache()
    backends["tilelang"] = torch.compile(model, backend="tilelang", dynamic=True)

    # Warm up all backends with a medium sequence
    warmup_ids = tokenizer(
        " ".join(["hello"] * 128), return_tensors="pt",
        max_length=128, truncation=True).input_ids.to("cuda", dtype=torch.int32)
    with torch.no_grad():
        for name, m in backends.items():
            print(f"Warming up {name}...")
            m(warmup_ids)

    # --- Prefill benchmark ---
    print("\n" + "=" * 70)
    print("PREFILL LATENCY (single forward, compute-bound)")
    print("=" * 70)

    for seq_len in [128, 256, 512, 1024]:
        text = " ".join(["word"] * seq_len * 2)
        input_ids = tokenizer(
            text, return_tensors="pt", max_length=seq_len,
            truncation=True).input_ids.to("cuda", dtype=torch.int32)
        actual_len = input_ids.shape[1]
        print(f"\n  seq_len={actual_len}:")

        for name, m in backends.items():
            try:
                times = bench_prefill(m, input_ids, warmup=3, repeat=10)
                avg = sum(times) / len(times)
                print(f"    {name:12s}  {fmt(times)}  ({avg*1000/actual_len:.2f} ms/tok)")
            except Exception as e:
                print(f"    {name:12s}  FAILED: {str(e)[:80]}")

    # --- Decode benchmark ---
    print("\n" + "=" * 70)
    print("DECODE LATENCY (generate 128 tokens, memory-bound)")
    print("=" * 70)

    prompt = "The capital of France is"
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(
        "cuda", dtype=torch.int32)
    print(f"  prompt_len={input_ids.shape[1]}, decode_steps=128\n")

    for name, m in backends.items():
        try:
            times, num_new = bench_decode(m, input_ids, num_steps=128, warmup=1, repeat=3)
            avg = sum(times) / len(times)
            tok_s = num_new / avg
            print(f"    {name:12s}  {fmt(times)}  {tok_s:.1f} tok/s")
        except Exception as e:
            print(f"    {name:12s}  FAILED: {str(e)[:80]}")

    print()


if __name__ == "__main__":
    main()
