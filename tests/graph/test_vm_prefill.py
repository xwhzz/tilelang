"""Test: LLaMA-2-7B prefill via Relax VM backend."""

import time
import torch
import torch._dynamo
import tilelang  # noqa: F401
from tilelang.graph import backend_config
from tilelang.graph.cache import clear_cache


def test_vm_prefill():
    from transformers import LlamaForCausalLM, AutoTokenizer

    model_name = "meta-llama/Llama-2-7b-hf"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = LlamaForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16).cuda().eval()

    ids = tokenizer(
        " ".join(["word"] * 1024), return_tensors="pt",
        max_length=512, truncation=True,
    ).input_ids.to("cuda", dtype=torch.int32)

    # Reference
    with torch.no_grad():
        ref = model(ids)

    # VM compile
    clear_cache()
    torch._dynamo.reset()
    backend_config.use_vm = True
    try:
        with torch.no_grad():
            compiled = torch.compile(model, backend="tilelang")
            # Warm-up (includes compilation)
            out = compiled(ids)

        max_err = (out.logits - ref.logits).abs().max().item()
        print(f"Correctness: max_err={max_err:.4f}")
        assert max_err < 0.2, f"Prefill error too large: {max_err}"

        # Benchmark
        torch.cuda.synchronize()
        N = 10
        t0 = time.perf_counter()
        with torch.no_grad():
            for _ in range(N):
                out = compiled(ids)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        vm_ms = (t1 - t0) / N * 1000
        print(f"VM latency: {vm_ms:.2f} ms (seq={ids.shape[1]})")

    finally:
        backend_config.use_vm = False

    # Compare with standard TileLang backend
    clear_cache()
    torch._dynamo.reset()
    with torch.no_grad():
        compiled_std = torch.compile(model, backend="tilelang")
        compiled_std(ids)  # warm-up

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(N):
            compiled_std(ids)
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    std_ms = (t1 - t0) / N * 1000
    print(f"Standard latency: {std_ms:.2f} ms")
    print(f"Speedup: {std_ms / vm_ms:.2f}x")


if __name__ == "__main__":
    test_vm_prefill()
