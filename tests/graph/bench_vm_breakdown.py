"""Latency breakdown: TileLang VM vs Inductor for LLaMA-2-7B prefill (seq=512).

Uses torch.profiler for accurate per-kernel GPU timing breakdown.
"""

import os
os.environ.setdefault("TQDM_DISABLE", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

import warnings
warnings.filterwarnings("ignore")

import torch
import torch._dynamo
import tilelang  # noqa: F401
from tilelang.graph import backend_config
from tilelang.graph.cache import clear_cache
from tilelang.profiler import do_bench


def profile_run(fn, label="profile"):
    """Run fn under torch.profiler and return the key_averages table."""
    # Warmup
    with torch.no_grad():
        for _ in range(3):
            fn()
    torch.cuda.synchronize()

    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=True,
    ) as prof:
        with torch.no_grad():
            fn()
    torch.cuda.synchronize()
    return prof


def print_top_kernels(prof, label, top_n=30):
    """Print top CUDA kernels by GPU time."""
    print(f"\n{'=' * 80}")
    print(f"  {label} — Top {top_n} CUDA kernels by GPU time")
    print(f"{'=' * 80}")
    table = prof.key_averages().table(
        sort_by="cuda_time_total", row_limit=top_n)
    print(table)


def main():
    from transformers import LlamaForCausalLM, AutoTokenizer

    model_name = "meta-llama/Llama-2-7b-hf"
    print(f"Model: {model_name}")
    print(f"GPU:   {torch.cuda.get_device_name()}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = LlamaForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16).cuda().eval()

    ids = tokenizer(
        " ".join(["word"] * 1024), return_tensors="pt",
        max_length=512, truncation=True,
    ).input_ids.to("cuda", dtype=torch.int32)
    seq = ids.shape[1]
    print(f"seq_len={seq}\n")

    # ── 1. End-to-end baselines ──
    print("=" * 80)
    print("  END-TO-END LATENCY (seq=512)")
    print("=" * 80)

    with torch.no_grad():
        eager_ms = do_bench(lambda: model(ids))
    print(f"  Eager:         {eager_ms:.2f} ms")

    torch._dynamo.reset()
    ind = torch.compile(model, backend="inductor")
    with torch.no_grad():
        ind(ids)
        ind_ms = do_bench(lambda: ind(ids))
    print(f"  Inductor:      {ind_ms:.2f} ms  ({eager_ms/ind_ms:.2f}x vs eager)")

    clear_cache()
    torch._dynamo.reset()
    backend_config.use_vm = True
    vm_compiled = torch.compile(model, backend="tilelang")
    with torch.no_grad():
        vm_compiled(ids)
        vm_ms = do_bench(lambda: vm_compiled(ids))
    backend_config.use_vm = False
    print(f"  TileLang VM:   {vm_ms:.2f} ms  ({eager_ms/vm_ms:.2f}x vs eager)")

    gap = vm_ms - ind_ms
    print(f"\n  Gap (VM - Inductor): {gap:+.2f} ms")

    # ── 2. Torch profiler breakdown ──
    # Profile Inductor
    prof_ind = profile_run(lambda: ind(ids), "Inductor")
    print_top_kernels(prof_ind, "Inductor")

    # Profile TileLang VM
    backend_config.use_vm = True
    prof_vm = profile_run(lambda: vm_compiled(ids), "TileLang VM")
    backend_config.use_vm = False
    print_top_kernels(prof_vm, "TileLang VM")

    # ── 3. Summary comparison ──
    def extract_cuda_breakdown(prof):
        """Extract key categories from profiler."""
        cats = {"gemm": 0, "attention": 0, "memcpy": 0, "tilelang": 0,
                "triton": 0, "other_cuda": 0, "fill_where": 0}
        for evt in prof.key_averages():
            t = evt.self_device_time_total / 1000.0  # us -> ms
            name = evt.key
            if t <= 0:
                continue
            if "nvjet" in name or "gemm" in name.lower() or "cutlass_80" in name:
                cats["gemm"] += t
            elif "fmha" in name or "flash" in name.lower():
                cats["attention"] += t
            elif "Memcpy" in name or "copy_" in name:
                cats["memcpy"] += t
            elif "_kernel" in name and ("fused_" in name or "add3" in name or "power" in name
                                         or "expand_dims" in name or "reshape7" in name):
                cats["tilelang"] += t
            elif "triton_" in name:
                cats["triton"] += t
            elif "fill_" in name or "where" in name or "scalar_tensor" in name:
                cats["fill_where"] += t
            else:
                cats["other_cuda"] += t
        return cats

    ind_cats = extract_cuda_breakdown(prof_ind)
    vm_cats = extract_cuda_breakdown(prof_vm)

    print(f"\n{'=' * 80}")
    print(f"  CUDA TIME BREAKDOWN (ms)")
    print(f"{'=' * 80}")
    print(f"  {'Category':<25s} {'Inductor':>10s} {'TileLang VM':>12s} {'Delta':>10s}")
    print(f"  {'-' * 60}")
    all_keys = sorted(set(list(ind_cats.keys()) + list(vm_cats.keys())))
    ind_total = 0
    vm_total = 0
    for k in all_keys:
        iv = ind_cats.get(k, 0)
        vv = vm_cats.get(k, 0)
        ind_total += iv
        vm_total += vv
        delta = vv - iv
        print(f"  {k:<25s} {iv:>8.2f}ms {vv:>10.2f}ms {delta:>+8.2f}ms")
    print(f"  {'-' * 60}")
    print(f"  {'TOTAL':<25s} {ind_total:>8.2f}ms {vm_total:>10.2f}ms {vm_total-ind_total:>+8.2f}ms")
    print(f"\n  do_bench:  Inductor={ind_ms:.2f}ms  VM={vm_ms:.2f}ms  gap={gap:+.2f}ms")


if __name__ == "__main__":
    main()
