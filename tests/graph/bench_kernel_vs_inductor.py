"""Per-kernel comparison: TileLang scheduled kernels vs Inductor triton kernels.

For each unique TileLang kernel in LLaMA-2-7B prefill, creates an equivalent
PyTorch function, compiles it with inductor, and compares GPU time.
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
    return min(ts)


def bench_kernel(name, tl_kernel, torch_fn, input_tensors, out_idx=-1):
    """Benchmark a TileLang kernel vs inductor-compiled torch function."""

    # TileLang kernel time
    try:
        tl_time = cuda_bench(lambda: tl_kernel(*input_tensors))
    except Exception as ex:
        tl_time = float("nan")
        print(f"    TileLang FAILED: {ex}")

    # Inductor time
    torch._dynamo.reset()
    try:
        compiled_fn = torch.compile(torch_fn, backend="inductor")
        # Warmup inductor
        for _ in range(3):
            compiled_fn(*input_tensors[:out_idx] if out_idx < 0 else input_tensors)
        ind_time = cuda_bench(
            lambda: compiled_fn(*input_tensors[:out_idx] if out_idx < 0 else input_tensors))
    except Exception as ex:
        ind_time = float("nan")
        print(f"    Inductor FAILED: {ex}")

    return tl_time, ind_time


def main():
    from transformers import LlamaForCausalLM, AutoTokenizer
    from tvm import tir
    from collections import Counter

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

    # Compile with TileLang backend
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
    from tilelang.graph.codegen import _compile_instructions, KernelCallInstr
    from tilelang.utils.target import determine_target

    ti = [i for i in captured["inputs"] if isinstance(i, torch.Tensor)]
    mod, fb = fx_to_relax(captured["gm"], ti)
    target = determine_target("auto")
    with target:
        opt = run_pipeline(mod, target)
    kernels = compile_tir_functions(opt, target)
    instrs, _, _, _ = _compile_instructions(opt)

    kernel_counts = Counter()
    for instr in instrs:
        if isinstance(instr, KernelCallInstr):
            kernel_counts[instr.kernel_name] += 1

    # Collect kernel shapes
    kernel_info = {}
    for gvar, func in opt.functions.items():
        if not isinstance(func, tir.PrimFunc):
            continue
        name = gvar.name_hint
        if name not in kernel_counts:
            continue
        bufs = []
        for p in func.params:
            if p in func.buffer_map:
                b = func.buffer_map[p]
                bufs.append(([int(s) for s in b.shape], str(b.dtype)))
        kernel_info[name] = bufs

    # ====================================================================
    # Define torch equivalents for each kernel type
    # ====================================================================

    results = []

    def dtype_map(s):
        return {"float16": torch.float16, "float32": torch.float32,
                "int32": torch.int32, "int64": torch.int64, "bool": torch.bool}[s]

    def make_tensors(bufs):
        ts = []
        for shape, dtype in bufs:
            dt = dtype_map(dtype)
            if dt in (torch.float16, torch.float32):
                ts.append(torch.randn(*shape, device="cuda", dtype=dt))
            elif dt == torch.bool:
                ts.append(torch.ones(*shape, device="cuda", dtype=dt))
            else:
                ts.append(torch.randint(0, 100, shape, device="cuda", dtype=dt))
        return ts

    print(f"{'Kernel':<45s} {'Calls':>5s} {'TileLang':>10s} {'Inductor':>10s} {'Ratio':>7s} {'TL Total':>10s}")
    print("=" * 95)

    # --- 1. add3: C = A + B (fp16) ---
    name = "add3"
    if name in kernels:
        count = kernel_counts[name]
        ts = make_tensors(kernel_info[name])
        def torch_add(a, b): return a + b
        tl_t, ind_t = bench_kernel(name, kernels[name], torch_add, ts)
        ratio = tl_t / ind_t if ind_t > 0 else float("nan")
        results.append((name, count, tl_t, ind_t, ratio))
        print(f"  {name:<43s} {count:>5d} {tl_t:>8.3f}ms {ind_t:>8.3f}ms {ratio:>6.2f}x {tl_t*count:>8.2f}ms")

    # --- 2. cast4: fp16 → fp32 ---
    name = "cast4"
    if name in kernels:
        count = kernel_counts[name]
        ts = make_tensors(kernel_info[name])
        def torch_cast(a): return a.float()
        tl_t, ind_t = bench_kernel(name, kernels[name], torch_cast, ts)
        ratio = tl_t / ind_t if ind_t > 0 else float("nan")
        results.append((name, count, tl_t, ind_t, ratio))
        print(f"  {name:<43s} {count:>5d} {tl_t:>8.3f}ms {ind_t:>8.3f}ms {ratio:>6.2f}x {tl_t*count:>8.2f}ms")

    # --- 3. power: x^2 (fp32) ---
    name = "power"
    if name in kernels:
        count = kernel_counts[name]
        ts = make_tensors(kernel_info[name])
        def torch_power(a): return a * a
        tl_t, ind_t = bench_kernel(name, kernels[name], torch_power, ts)
        ratio = tl_t / ind_t if ind_t > 0 else float("nan")
        results.append((name, count, tl_t, ind_t, ratio))
        print(f"  {name:<43s} {count:>5d} {tl_t:>8.3f}ms {ind_t:>8.3f}ms {ratio:>6.2f}x {tl_t*count:>8.2f}ms")

    # --- 4. fused_mean_add1_tir_rsqrt_multiply1_cast5_multiply2 (RMSNorm) ---
    name = "fused_mean_add1_tir_rsqrt_multiply1_cast5_multiply2"
    if name in kernels:
        count = kernel_counts[name]
        ts = make_tensors(kernel_info[name])
        # RMSNorm: out = (x / sqrt(mean(x^2) + eps)) * w
        # Inputs: power_out(fp32), cast_out(fp32), weight(fp16) → out(fp16)
        def torch_rmsnorm(power_out, cast_out, weight):
            variance = power_out.mean(dim=-1, keepdim=True)
            rsqrt = torch.rsqrt(variance + 1e-6)
            normed = cast_out * rsqrt
            return (normed * weight.float()).half()
        tl_t, ind_t = bench_kernel(name, kernels[name], torch_rmsnorm, ts)
        ratio = tl_t / ind_t if ind_t > 0 else float("nan")
        results.append((name, count, tl_t, ind_t, ratio))
        print(f"  {'rmsnorm':<43s} {count:>5d} {tl_t:>8.3f}ms {ind_t:>8.3f}ms {ratio:>6.2f}x {tl_t*count:>8.2f}ms")

    # --- 5. fused_silu_multiply4: SiLU(a) * b ---
    name = "fused_silu_multiply4"
    if name in kernels:
        count = kernel_counts[name]
        ts = make_tensors(kernel_info[name])
        def torch_silu_mul(a, b): return torch.nn.functional.silu(a) * b
        tl_t, ind_t = bench_kernel(name, kernels[name], torch_silu_mul, ts)
        ratio = tl_t / ind_t if ind_t > 0 else float("nan")
        results.append((name, count, tl_t, ind_t, ratio))
        print(f"  {'silu_mul':<43s} {count:>5d} {tl_t:>8.3f}ms {ind_t:>8.3f}ms {ratio:>6.2f}x {tl_t*count:>8.2f}ms")

    # --- 6. fused_reshape4_transpose1: reshape + transpose for Q/K ---
    name = "fused_reshape4_transpose1"
    if name in kernels:
        count = kernel_counts[name]
        ts = make_tensors(kernel_info[name])
        # [1,512,4096] → [1,32,512,128] (reshape + transpose dims 1,2)
        def torch_reshape_transpose(x):
            return x.reshape(1, seq, 32, 128).transpose(1, 2).contiguous()
        tl_t, ind_t = bench_kernel(name, kernels[name], torch_reshape_transpose, ts)
        ratio = tl_t / ind_t if ind_t > 0 else float("nan")
        results.append((name, count, tl_t, ind_t, ratio))
        print(f"  {'reshape_transpose':<43s} {count:>5d} {tl_t:>8.3f}ms {ind_t:>8.3f}ms {ratio:>6.2f}x {tl_t*count:>8.2f}ms")

    # --- 7. fused_transpose2_reshape6: transpose + reshape (inverse of above) ---
    name = "fused_transpose2_reshape6"
    if name in kernels:
        count = kernel_counts[name]
        ts = make_tensors(kernel_info[name])
        # [1,32,512,128] → [1,512,4096]
        def torch_transpose_reshape(x):
            return x.transpose(1, 2).contiguous().reshape(1, seq, 4096)
        tl_t, ind_t = bench_kernel(name, kernels[name], torch_transpose_reshape, ts)
        ratio = tl_t / ind_t if ind_t > 0 else float("nan")
        results.append((name, count, tl_t, ind_t, ratio))
        print(f"  {'transpose_reshape':<43s} {count:>5d} {tl_t:>8.3f}ms {ind_t:>8.3f}ms {ratio:>6.2f}x {tl_t*count:>8.2f}ms")

    # --- 8. RoPE: reshape+transpose+cos/sin multiply+concat ---
    name = "fused_reshape4_transpose1_multiply3_strided_slice2_reshape5_strided_slice3_reshape5_tir_negative_concatenate1_multiply3_add2"
    if name in kernels:
        count = kernel_counts[name]
        ts = make_tensors(kernel_info[name])
        # x:[1,512,4096], cos:[1,1,512,128], sin:[1,1,512,128] → [1,32,512,128]
        # RoPE: reshape to [1,32,512,128], then apply rotary
        def torch_rope(x, cos, sin):
            x = x.reshape(1, seq, 32, 128).transpose(1, 2)
            x1, x2 = x[..., :64], x[..., 64:]
            rotated = torch.cat([-x2, x1], dim=-1)
            return x * cos + rotated * sin
        tl_t, ind_t = bench_kernel(name, kernels[name], torch_rope, ts)
        ratio = tl_t / ind_t if ind_t > 0 else float("nan")
        results.append((name, count, tl_t, ind_t, ratio))
        print(f"  {'rope':<43s} {count:>5d} {tl_t:>8.3f}ms {ind_t:>8.3f}ms {ratio:>6.2f}x {tl_t*count:>8.2f}ms")

    # --- Summary ---
    print("\n" + "=" * 95)
    total_tl = sum(t * c for _, c, t, _, _ in results if t == t)
    total_ind = sum(i * c for _, c, _, i, _ in results if i == i)
    print(f"  {'TOTAL (TileLang kernels only)':<43s} {'':>5s} {'':>10s} {'':>10s} {'':>7s} {total_tl:>8.2f}ms")
    print(f"  {'TOTAL (Inductor equivalent)':<43s} {'':>5s} {'':>10s} {'':>10s} {'':>7s} {total_ind:>8.2f}ms")
    print(f"  {'Ratio':<43s} {'':>5s} {'':>10s} {'':>10s} {'':>7s} {total_tl/total_ind:>7.2f}x")

    # Find worst offenders
    print("\n--- Optimization priorities (sorted by potential savings) ---")
    savings = []
    for name, count, tl_t, ind_t, ratio in results:
        if tl_t == tl_t and ind_t == ind_t:
            save = (tl_t - ind_t) * count
            savings.append((save, name, count, tl_t, ind_t, ratio))
    savings.sort(reverse=True)
    for save, name, count, tl_t, ind_t, ratio in savings:
        print(f"  {name:<43s}  saving={save:>6.2f}ms  ({ratio:.2f}x slower, {count} calls)")


if __name__ == "__main__":
    main()
