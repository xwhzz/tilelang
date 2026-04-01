"""Per-kernel latency breakdown: TileLang vs Inductor for LLaMA-2-7B prefill.

Measures:
1. Total prefill time (eager, inductor, tilelang)
2. Per-kernel GPU time via CUDA events (tilelang only)
3. Host overhead (total - sum of kernel GPU times)
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


def cuda_bench(fn, warmup=5, repeat=20):
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
    return sum(ts) / len(ts)


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
    print(f"seq_len={ids.shape[1]}")

    # ── Total latency comparison ──
    print("\n=== TOTAL PREFILL LATENCY ===")
    with torch.no_grad():
        eager_ms = cuda_bench(lambda: model(ids))
    print(f"  Eager:    {eager_ms:.2f} ms")

    torch._dynamo.reset()
    ind = torch.compile(model, backend="inductor")
    with torch.no_grad():
        ind(ids)
        ind_ms = cuda_bench(lambda: ind(ids))
    print(f"  Inductor: {ind_ms:.2f} ms  ({eager_ms/ind_ms:.2f}x)")

    clear_cache()
    torch._dynamo.reset()
    tl = torch.compile(model, backend="tilelang")
    with torch.no_grad():
        tl(ids)
        tl_ms = cuda_bench(lambda: tl(ids))
    print(f"  TileLang: {tl_ms:.2f} ms  ({eager_ms/tl_ms:.2f}x)")

    # ── Per-kernel profiling (TileLang) ──
    print("\n=== PER-KERNEL GPU TIME (TileLang) ===")

    # Capture the compiled wrapper's internals
    captured = {}
    def cap(gm, inputs):
        captured["gm"] = gm
        captured["inputs"] = inputs
        return gm.forward
    torch._dynamo.reset()
    clear_cache()
    with torch.no_grad():
        torch.compile(model, backend=cap)(ids)

    from tilelang.graph.converter import fx_to_relax
    from tilelang.graph.pipeline import run_pipeline
    from tilelang.graph.compiler import compile_tir_functions
    from tilelang.graph.codegen import _compile_instructions, KernelCallInstr, TorchFallbackInstr
    from tilelang.utils.target import determine_target

    ti = [i for i in captured["inputs"] if isinstance(i, torch.Tensor)]
    mod, fb = fx_to_relax(captured["gm"], ti)
    target = determine_target("auto")
    with target:
        opt = run_pipeline(mod, target)
    kernels = compile_tir_functions(opt, target)
    instrs, params, outs, consts = _compile_instructions(opt)

    # Count kernel calls and fallback calls per unique name
    from collections import Counter
    kernel_counts = Counter()
    fallback_counts = Counter()
    for instr in instrs:
        if isinstance(instr, KernelCallInstr):
            kernel_counts[instr.kernel_name] += 1
        elif isinstance(instr, TorchFallbackInstr):
            fallback_counts[instr.op_name] += 1

    # Profile each unique kernel in isolation
    from tvm import tir
    kernel_shapes = {}
    for gvar, func in opt.functions.items():
        if not isinstance(func, tir.PrimFunc):
            continue
        name = gvar.name_hint
        if name in kernels:
            shapes = []
            for p in func.params:
                if p in func.buffer_map:
                    b = func.buffer_map[p]
                    shapes.append(([int(s) for s in b.shape], str(b.dtype)))
            kernel_shapes[name] = shapes

    print(f"\n{'Kernel':<65s} {'Calls':>5s} {'Per-call':>10s} {'Total':>10s} {'Buffers'}")
    print("-" * 130)

    kernel_times = {}
    total_kernel_gpu = 0.0

    for name in sorted(kernel_counts.keys()):
        count = kernel_counts[name]
        exe = kernels[name]
        shapes = kernel_shapes.get(name, [])

        # Create test tensors
        test_tensors = []
        for shape, dtype in shapes:
            dtype_map = {
                "float16": torch.float16, "float32": torch.float32,
                "int32": torch.int32, "int64": torch.int64, "bool": torch.bool,
            }
            dt = dtype_map.get(dtype, torch.float32)
            if dt in (torch.float16, torch.float32):
                test_tensors.append(torch.randn(*shape, device="cuda", dtype=dt))
            elif dt == torch.bool:
                test_tensors.append(torch.ones(*shape, device="cuda", dtype=torch.bool))
            else:
                test_tensors.append(torch.randint(0, 100, shape, device="cuda", dtype=dt))

        # Benchmark single call
        try:
            with torch.no_grad():
                per_call = cuda_bench(lambda: exe(*test_tensors), warmup=10, repeat=50)
        except Exception as ex:
            per_call = float("nan")

        import math
        total = per_call * count
        if not math.isnan(total):
            total_kernel_gpu += total
        kernel_times[name] = (per_call, count, total)

        buf_str = " × ".join(f"{s}" for s, _ in shapes)
        print(f"  {name:<63s} {count:>5d} {per_call:>8.3f}ms {total:>8.2f}ms  {buf_str}")

    print(f"\n{'TOTAL kernel GPU time:':<70s} {total_kernel_gpu:>18.2f}ms")

    # ── Fallback ops: profile representative calls ──
    print(f"\n=== FALLBACK OPS (GPU time) ===")
    total_fallback_count = sum(fallback_counts.values())
    total_fallback_gpu = 0.0

    # Profile linear with actual LLaMA-2-7B shapes
    # 7 linears per layer × 32 layers + 1 lm_head = 225
    seq = ids.shape[1]
    linear_benchmarks = [
        ("qkv_proj", (1, seq, 4096), (4096, 4096), 3 * 32),
        ("o_proj",   (1, seq, 4096), (4096, 4096), 32),
        ("gate_up",  (1, seq, 4096), (11008, 4096), 2 * 32),
        ("down",     (1, seq, 11008), (4096, 11008), 32),
        ("lm_head",  (1, seq, 4096), (32000, 4096), 1),
    ]

    print(f"\n  {'Op':<50s} {'Calls':>5s} {'Per-call':>10s} {'Total':>10s}")
    print(f"  {'-'*80}")

    for op_name in sorted(fallback_counts.keys()):
        count = fallback_counts[op_name]
        if op_name == "linear":
            total_linear_gpu = 0.0
            for label, xshape, wshape, n_calls in linear_benchmarks:
                x = torch.randn(*xshape, device="cuda", dtype=torch.float16)
                w = torch.randn(*wshape, device="cuda", dtype=torch.float16)
                t = cuda_bench(lambda: torch.nn.functional.linear(x, w), warmup=10, repeat=50)
                total_linear_gpu += t * n_calls
                print(f"    linear/{label:<44s} {n_calls:>5d} {t:>8.3f}ms {t*n_calls:>8.2f}ms")
            total_fallback_gpu += total_linear_gpu
            per_call = total_linear_gpu / max(count, 1)
            print(f"  {'linear (subtotal)':<50s} {count:>5d} {per_call:>8.3f}ms {total_linear_gpu:>8.2f}ms")
        elif op_name == "scaled_dot_product_attention":
            q = torch.randn(1, 32, seq, 128, device="cuda", dtype=torch.float16)
            k = torch.randn(1, 32, seq, 128, device="cuda", dtype=torch.float16)
            v = torch.randn(1, 32, seq, 128, device="cuda", dtype=torch.float16)
            per_call = cuda_bench(
                lambda: torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True),
                warmup=10, repeat=50)
            total = per_call * count
            total_fallback_gpu += total
            print(f"  {op_name:<50s} {count:>5d} {per_call:>8.3f}ms {total:>8.2f}ms")
        elif op_name == "cat":
            a_t = torch.randn(1, 32, seq, 128, device="cuda", dtype=torch.float16)
            b_t = torch.randn(1, 32, seq, 128, device="cuda", dtype=torch.float16)
            per_call = cuda_bench(lambda: torch.cat([a_t, b_t], dim=-2), warmup=10, repeat=50)
            total = per_call * count
            total_fallback_gpu += total
            print(f"  {op_name:<50s} {count:>5d} {per_call:>8.3f}ms {total:>8.2f}ms")
        elif op_name == "embedding":
            emb = torch.nn.Embedding(32000, 4096, dtype=torch.float16, device="cuda")
            inp_emb = torch.randint(0, 32000, (1, seq), device="cuda")
            per_call = cuda_bench(lambda: emb(inp_emb), warmup=10, repeat=50)
            total = per_call * count
            total_fallback_gpu += total
            print(f"  {op_name:<50s} {count:>5d} {per_call:>8.3f}ms {total:>8.2f}ms")
        else:
            print(f"  {op_name:<50s} {count:>5d}       ~0ms")

    print(f"  {'TOTAL fallback GPU:':<40s} {'':>5s} {'':>10s} {total_fallback_gpu:>8.2f}ms")

    # ── Host overhead ──
    all_gpu = total_kernel_gpu + total_fallback_gpu
    host_overhead = tl_ms - all_gpu
    print(f"\n=== OVERHEAD ANALYSIS ===")
    print(f"  Total TileLang wall time:     {tl_ms:.2f} ms")
    print(f"  TileLang kernel GPU time:     {total_kernel_gpu:.2f} ms")
    print(f"  Fallback ops GPU time:        {total_fallback_gpu:.2f} ms")
    print(f"  All GPU time:                 {all_gpu:.2f} ms")
    print(f"  Host overhead (diff):         {host_overhead:.2f} ms ({host_overhead/tl_ms*100:.1f}%)")
    print(f"  Kernel calls:                 {sum(kernel_counts.values())}")
    print(f"  Fallback calls:               {total_fallback_count}")
    print(f"  Total instructions:           {len(instrs)}")
    if host_overhead > 0:
        print(f"  Per-instruction overhead:     {host_overhead/len(instrs)*1000:.1f} µs")

    # ── Compare with inductor kernel time ──
    print(f"\n=== SUMMARY ===")
    print(f"  Inductor total:               {ind_ms:.2f} ms")
    print(f"  TileLang kernel GPU:          {total_kernel_gpu:.2f} ms")
    print(f"  Fallback GPU:                 {total_fallback_gpu:.2f} ms")
    print(f"  Host overhead:                {host_overhead:.2f} ms")
    print(f"  TileLang total:               {tl_ms:.2f} ms")

    gap = tl_ms - ind_ms
    print(f"\n  Gap vs inductor:              {gap:+.2f} ms")
    if host_overhead > gap:
        print(f"  → Host overhead ({host_overhead:.1f}ms) > gap ({gap:.1f}ms) — reducing overhead would close the gap")
    elif total_fallback_gpu > total_kernel_gpu:
        print(f"  → Fallback GPU ({total_fallback_gpu:.1f}ms) dominates — compiling more ops with TileLang would help")
    else:
        print(f"  → Kernel performance is the bottleneck")


if __name__ == "__main__":
    main()
