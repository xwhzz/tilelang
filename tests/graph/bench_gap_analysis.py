"""Deep analysis of TileLang vs Inductor performance gap.

Compares:
1. Number of kernel launches
2. Inductor fusion patterns (what gets fused with linear?)
3. Per-category GPU time breakdown
4. Host dispatch overhead measurement
"""

import os
os.environ["TQDM_DISABLE"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

import warnings
warnings.filterwarnings("ignore")
import logging
logging.disable(logging.WARNING)

import torch
import torch._dynamo
import time
import tilelang  # noqa: F401
from tilelang.graph.cache import clear_cache


def cuda_bench(fn, warmup=10, repeat=50):
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(repeat):
        torch.cuda.synchronize()
        s.record()
        fn()
        e.record()
        torch.cuda.synchronize()
        ts.append(s.elapsed_time(e))
    return min(ts), sum(ts) / len(ts)


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

    # ================================================================
    # 1. Profile Inductor with torch.profiler to count kernels
    # ================================================================
    print("=" * 80)
    print("1. INDUCTOR KERNEL ANALYSIS")
    print("=" * 80)

    torch._dynamo.reset()
    ind_model = torch.compile(model, backend="inductor")
    with torch.no_grad():
        ind_model(ids)  # warmup + compile

    # Use torch profiler to capture kernel launches
    with torch.no_grad():
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            record_shapes=True,
        ) as prof:
            ind_model(ids)

    # Count and categorize CUDA kernels
    cuda_events = [e for e in prof.key_averages()
                   if e.device_type == torch.autograd.DeviceType.CUDA
                   and e.count > 0]

    # Categorize
    ind_categories = {}
    ind_total_cuda_time = 0
    for e in cuda_events:
        name = e.key
        t_us = e.self_device_time_total  # microseconds
        count = e.count
        ind_total_cuda_time += t_us

        if "gemm" in name.lower() or "cutlass" in name.lower() or "cublas" in name.lower():
            cat = "GEMM"
        elif "flash" in name.lower() or "attention" in name.lower() or "fmha" in name.lower():
            cat = "SDPA"
        elif "triton" in name.lower() or "kernel" in name.lower():
            cat = "Triton"
        elif "copy" in name.lower() or "memcpy" in name.lower() or "memset" in name.lower():
            cat = "Memory"
        else:
            cat = "Other"

        if cat not in ind_categories:
            ind_categories[cat] = {"count": 0, "time_us": 0, "kernels": []}
        ind_categories[cat]["count"] += count
        ind_categories[cat]["time_us"] += t_us
        ind_categories[cat]["kernels"].append((name, count, t_us))

    print(f"\n  {'Category':<15s} {'Kernels':>8s} {'GPU time':>12s} {'%':>6s}")
    print(f"  {'-'*45}")
    total_ind_launches = 0
    for cat in sorted(ind_categories.keys(), key=lambda c: -ind_categories[c]["time_us"]):
        info = ind_categories[cat]
        total_ind_launches += info["count"]
        pct = info["time_us"] / max(ind_total_cuda_time, 1) * 100
        print(f"  {cat:<15s} {info['count']:>8d} {info['time_us']/1000:>10.2f}ms {pct:>5.1f}%")
    print(f"  {'TOTAL':<15s} {total_ind_launches:>8d} {ind_total_cuda_time/1000:>10.2f}ms")

    # Show top inductor triton kernels
    print(f"\n  Top Triton kernels:")
    triton_kernels = ind_categories.get("Triton", {}).get("kernels", [])
    triton_kernels.sort(key=lambda x: -x[2])
    for name, count, t_us in triton_kernels[:10]:
        print(f"    {name[:70]:<70s} {count:>4d}× {t_us/1000:>7.3f}ms")

    # ================================================================
    # 2. Profile TileLang with torch.profiler
    # ================================================================
    print(f"\n{'='*80}")
    print("2. TILELANG KERNEL ANALYSIS")
    print("=" * 80)

    torch._dynamo.reset()
    clear_cache()
    tl_model = torch.compile(model, backend="tilelang")
    with torch.no_grad():
        tl_model(ids)  # warmup + compile

    with torch.no_grad():
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            record_shapes=True,
        ) as prof2:
            tl_model(ids)

    cuda_events2 = [e for e in prof2.key_averages()
                    if e.device_type == torch.autograd.DeviceType.CUDA
                    and e.count > 0]

    tl_categories = {}
    tl_total_cuda_time = 0
    for e in cuda_events2:
        name = e.key
        t_us = e.self_device_time_total
        count = e.count
        tl_total_cuda_time += t_us

        if "gemm" in name.lower() or "cutlass" in name.lower() or "cublas" in name.lower():
            cat = "GEMM (cuBLAS)"
        elif "flash" in name.lower() or "attention" in name.lower() or "fmha" in name.lower():
            cat = "SDPA"
        elif "copy" in name.lower() or "memcpy" in name.lower() or "memset" in name.lower():
            cat = "Memory"
        elif "tilelang" in name.lower() or "tl_" in name.lower():
            cat = "TileLang"
        else:
            cat = "Other (TileLang)"

        if cat not in tl_categories:
            tl_categories[cat] = {"count": 0, "time_us": 0, "kernels": []}
        tl_categories[cat]["count"] += count
        tl_categories[cat]["time_us"] += t_us
        tl_categories[cat]["kernels"].append((name, count, t_us))

    print(f"\n  {'Category':<20s} {'Kernels':>8s} {'GPU time':>12s} {'%':>6s}")
    print(f"  {'-'*50}")
    total_tl_launches = 0
    for cat in sorted(tl_categories.keys(), key=lambda c: -tl_categories[c]["time_us"]):
        info = tl_categories[cat]
        total_tl_launches += info["count"]
        pct = info["time_us"] / max(tl_total_cuda_time, 1) * 100
        print(f"  {cat:<20s} {info['count']:>8d} {info['time_us']/1000:>10.2f}ms {pct:>5.1f}%")
    print(f"  {'TOTAL':<20s} {total_tl_launches:>8d} {tl_total_cuda_time/1000:>10.2f}ms")

    # ================================================================
    # 3. Compare wall time and GPU time
    # ================================================================
    print(f"\n{'='*80}")
    print("3. WALL TIME vs GPU TIME COMPARISON")
    print("=" * 80)

    # Measure wall time accurately
    with torch.no_grad():
        ind_min, ind_avg = cuda_bench(lambda: ind_model(ids))
        tl_min, tl_avg = cuda_bench(lambda: tl_model(ids))

    print(f"\n  {'':30s} {'Inductor':>12s} {'TileLang':>12s} {'Diff':>10s}")
    print(f"  {'-'*70}")
    print(f"  {'Wall time (min)':30s} {ind_min:>10.2f}ms {tl_min:>10.2f}ms {tl_min-ind_min:>+9.2f}ms")
    print(f"  {'Wall time (avg)':30s} {ind_avg:>10.2f}ms {tl_avg:>10.2f}ms {tl_avg-ind_avg:>+9.2f}ms")
    print(f"  {'Profiler GPU time':30s} {ind_total_cuda_time/1000:>10.2f}ms {tl_total_cuda_time/1000:>10.2f}ms {(tl_total_cuda_time-ind_total_cuda_time)/1000:>+9.2f}ms")
    print(f"  {'Kernel launches':30s} {total_ind_launches:>10d} {total_tl_launches:>10d} {total_tl_launches-total_ind_launches:>+10d}")

    # ================================================================
    # 4. Breakdown the gap
    # ================================================================
    print(f"\n{'='*80}")
    print("4. GAP BREAKDOWN")
    print("=" * 80)

    gap = tl_min - ind_min
    print(f"\n  Total gap: {gap:+.2f}ms (TileLang - Inductor)")
    print()

    # Compare GEMM time
    ind_gemm = ind_categories.get("GEMM", {}).get("time_us", 0)
    tl_gemm = tl_categories.get("GEMM (cuBLAS)", {}).get("time_us", 0)
    ind_gemm_count = ind_categories.get("GEMM", {}).get("count", 0)
    tl_gemm_count = tl_categories.get("GEMM (cuBLAS)", {}).get("count", 0)
    print(f"  GEMM:")
    print(f"    Inductor: {ind_gemm_count} launches, {ind_gemm/1000:.2f}ms GPU")
    print(f"    TileLang: {tl_gemm_count} launches, {tl_gemm/1000:.2f}ms GPU")
    print(f"    Diff:     {(tl_gemm-ind_gemm)/1000:+.2f}ms")

    # Compare SDPA
    ind_sdpa = ind_categories.get("SDPA", {}).get("time_us", 0)
    tl_sdpa = tl_categories.get("SDPA", {}).get("time_us", 0)
    ind_sdpa_count = ind_categories.get("SDPA", {}).get("count", 0)
    tl_sdpa_count = tl_categories.get("SDPA", {}).get("count", 0)
    print(f"\n  SDPA:")
    print(f"    Inductor: {ind_sdpa_count} launches, {ind_sdpa/1000:.2f}ms GPU")
    print(f"    TileLang: {tl_sdpa_count} launches, {tl_sdpa/1000:.2f}ms GPU")
    print(f"    Diff:     {(tl_sdpa-ind_sdpa)/1000:+.2f}ms")

    # Compare elementwise/fused
    ind_triton = ind_categories.get("Triton", {}).get("time_us", 0)
    ind_triton_count = ind_categories.get("Triton", {}).get("count", 0)
    tl_other = tl_categories.get("Other (TileLang)", {}).get("time_us", 0)
    tl_tilelang = tl_categories.get("TileLang", {}).get("time_us", 0)
    tl_elem_total = tl_other + tl_tilelang
    tl_elem_count = (tl_categories.get("Other (TileLang)", {}).get("count", 0) +
                     tl_categories.get("TileLang", {}).get("count", 0))
    print(f"\n  Elementwise/Fused kernels:")
    print(f"    Inductor (Triton):  {ind_triton_count} launches, {ind_triton/1000:.2f}ms GPU")
    print(f"    TileLang:           {tl_elem_count} launches, {tl_elem_total/1000:.2f}ms GPU")
    print(f"    Diff:               {(tl_elem_total-ind_triton)/1000:+.2f}ms")

    # Host overhead estimation
    print(f"\n  Host overhead estimate:")
    ind_host = ind_min - ind_total_cuda_time / 1000
    tl_host = tl_min - tl_total_cuda_time / 1000
    print(f"    Inductor: wall={ind_min:.2f} - gpu={ind_total_cuda_time/1000:.2f} = {ind_host:.2f}ms")
    print(f"    TileLang: wall={tl_min:.2f} - gpu={tl_total_cuda_time/1000:.2f} = {tl_host:.2f}ms")
    print(f"    Note: negative means GPU kernels overlap (pipelined)")

    # ================================================================
    # 5. What does inductor fuse that we don't?
    # ================================================================
    print(f"\n{'='*80}")
    print("5. INDUCTOR FUSION ANALYSIS")
    print("=" * 80)

    # Show all inductor triton kernel names (these are fused kernels)
    print(f"\n  Inductor Triton kernels (fused ops):")
    all_triton = ind_categories.get("Triton", {}).get("kernels", [])
    all_triton.sort(key=lambda x: -x[2])
    for name, count, t_us in all_triton:
        print(f"    {count:>4d}× {t_us/1000:>7.3f}ms  {name[:80]}")

    # Show all inductor GEMM kernels
    print(f"\n  Inductor GEMM kernels:")
    all_gemm = ind_categories.get("GEMM", {}).get("kernels", [])
    all_gemm.sort(key=lambda x: -x[2])
    for name, count, t_us in all_gemm:
        print(f"    {count:>4d}× {t_us/1000:>7.3f}ms  {name[:80]}")

    print(f"\n  Total Inductor kernel launches: {total_ind_launches}")
    print(f"  Total TileLang kernel launches: {total_tl_launches}")
    print(f"  Launch count ratio: {total_tl_launches/max(total_ind_launches,1):.1f}x more launches in TileLang")


if __name__ == "__main__":
    main()
