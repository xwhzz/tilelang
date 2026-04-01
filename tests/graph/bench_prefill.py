"""Prefill latency breakdown: LLaMA-2-7B TileLang vs Inductor vs Eager."""

import os
os.environ.setdefault("TQDM_DISABLE", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

import time
import warnings
warnings.filterwarnings("ignore")

import torch
import torch._dynamo
import tilelang  # noqa: F401
from tilelang.graph.cache import clear_cache


def bench(model, ids, warmup=3, repeat=10):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    with torch.no_grad():
        for _ in range(warmup):
            model(ids)
    torch.cuda.synchronize()
    times = []
    with torch.no_grad():
        for _ in range(repeat):
            torch.cuda.synchronize()
            start.record()
            model(ids)
            end.record()
            torch.cuda.synchronize()
            times.append(start.elapsed_time(end))
    return times


def main():
    from transformers import LlamaForCausalLM, AutoTokenizer

    model_name = "meta-llama/Llama-2-7b-hf"
    print(f"Model: {model_name}")
    print(f"GPU:   {torch.cuda.get_device_name()}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = LlamaForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16).cuda().eval()

    for seq_len in [128, 256, 512]:
        ids = tokenizer(
            " ".join(["word"] * seq_len * 2), return_tensors="pt",
            max_length=seq_len, truncation=True,
        ).input_ids.to("cuda", dtype=torch.int32)
        al = ids.shape[1]

        # Correctness
        with torch.no_grad():
            ref = model(ids)
        torch._dynamo.reset()
        clear_cache()
        tl = torch.compile(model, backend="tilelang")
        with torch.no_grad():
            out = tl(ids)
        max_err = (out.logits - ref.logits).abs().max().item()

        # Benchmark
        e = bench(model, ids)
        torch._dynamo.reset()
        ind = torch.compile(model, backend="inductor")
        with torch.no_grad():
            ind(ids)
        i = bench(ind, ids)
        t = bench(tl, ids)

        ea = sum(e) / len(e)
        ia = sum(i) / len(i)
        ta = sum(t) / len(t)
        print(
            f"seq={al:<4d} err={max_err:.4f}  "
            f"eager={ea:.1f}ms  "
            f"inductor={ia:.1f}ms({ea/ia:.2f}x)  "
            f"tilelang={ta:.1f}ms({ea/ta:.2f}x)"
        )


if __name__ == "__main__":
    main()
