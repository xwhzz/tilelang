"""Benchmark elementwise kernels through torch.compile backends.

Compares TileLang, Inductor, and Eager across real LLM-sized elementwise
operations: GELU, SiLU, RMSNorm-like, residual add, mul+add fused, etc.
"""
import os
os.environ["TQDM_DISABLE"] = "1"
import torch
import torch._dynamo
import tilelang  # noqa: F401 — registers backend
from tilelang.profiler import do_bench
from tilelang.graph.cache import clear_cache

# ---------------------------------------------------------------------------
# Real LLM shapes: (batch*seq, hidden) for elementwise ops
# Derived from LLaMA-2-7B / Qwen3-0.6B / Qwen2.5-7B configs
# ---------------------------------------------------------------------------
SHAPES = [
    # (name, M, N)  — M = batch*seq, N = hidden
    # Prefill shapes
    ("prefill_s128_h4096",   128, 4096),
    ("prefill_s512_h4096",   512, 4096),
    ("prefill_s128_h1024",   128, 1024),   # Qwen3-0.6B
    ("prefill_s512_h1024",   512, 1024),
    # FFN intermediate
    ("ffn_s128_h11008",      128, 11008),  # LLaMA-2-7B SwiGLU
    ("ffn_s512_h11008",      512, 11008),
    ("ffn_s128_h3072",       128, 3072),   # Qwen3-0.6B
    ("ffn_s512_h3072",       512, 3072),
    # Decode (single token)
    ("decode_s1_h4096",        1, 4096),
    ("decode_s1_h1024",        1, 1024),
    ("decode_s1_h11008",       1, 11008),
    # Large batch
    # ("large_s2048_h4096",   2048, 4096),
    # ("large_s4096_h4096",   4096, 4096),
]

# ---------------------------------------------------------------------------
# Elementwise operations to benchmark
# ---------------------------------------------------------------------------
OPERATIONS = {
    "gelu": lambda x: torch.nn.functional.gelu(x),
    "silu": lambda x: torch.nn.functional.silu(x),
    # relu omitted: TileLang codegen emits max(__half,__half) which nvcc rejects
    "add": lambda x, y: x + y,
    "mul_add": lambda x, y, z: x * y + z,
    "rsqrt_mul": lambda x, w: x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6) * w,
}

UNARY_OPS = {"gelu", "silu", "relu"}
BINARY_OPS = {"add"}
TERNARY_OPS = {"mul_add"}
NORM_OPS = {"rsqrt_mul"}


def make_inputs(op_name, M, N, dtype=torch.float32, device="cuda"):
    x = torch.randn(M, N, dtype=dtype, device=device)
    if op_name in UNARY_OPS:
        return (x,)
    elif op_name in BINARY_OPS:
        return (x, torch.randn(M, N, dtype=dtype, device=device))
    elif op_name in TERNARY_OPS:
        return (x, torch.randn(M, N, dtype=dtype, device=device),
                torch.randn(M, N, dtype=dtype, device=device))
    elif op_name in NORM_OPS:
        return (x, torch.randn(N, dtype=dtype, device=device))
    return (x,)


def bench_one(op_name, op_fn, shape_name, M, N, dtype=torch.float32):
    inputs = make_inputs(op_name, M, N, dtype)

    # 1) Eager
    eager_ms = do_bench(lambda: op_fn(*inputs))

    # 2) Inductor
    torch._dynamo.reset()
    torch.cuda.empty_cache()
    try:
        compiled_ind = torch.compile(op_fn, backend="inductor")
        compiled_ind(*inputs)  # trigger compilation outside timing
        inductor_ms = do_bench(lambda: compiled_ind(*inputs))
    except Exception:
        inductor_ms = float("nan")

    # 3) TileLang
    torch._dynamo.reset()
    torch.cuda.empty_cache()
    clear_cache()
    try:
        compiled_tl = torch.compile(op_fn, backend="tilelang")
        compiled_tl(*inputs)  # trigger compilation outside timing
        tilelang_ms = do_bench(lambda: compiled_tl(*inputs))
    except Exception:
        tilelang_ms = float("nan")

    del inputs
    torch._dynamo.reset()
    torch.cuda.empty_cache()
    return eager_ms, inductor_ms, tilelang_ms


def main():
    print(f"{'Op':<12} {'Shape':<25} {'M':>6} {'N':>6} "
          f"{'Eager':>8} {'Inductor':>9} {'TileLang':>9} "
          f"{'TL/Eager':>9} {'TL/Ind':>7}")
    print("-" * 110)

    results = []
    for op_name, op_fn in OPERATIONS.items():
        for shape_name, M, N in SHAPES:
            eager_ms, inductor_ms, tilelang_ms = bench_one(
                op_name, op_fn, shape_name, M, N)

            tl_vs_eager = tilelang_ms / eager_ms if eager_ms > 0 else float("nan")
            tl_vs_ind = tilelang_ms / inductor_ms if inductor_ms > 0 else float("nan")

            print(f"{op_name:<12} {shape_name:<25} {M:>6} {N:>6} "
                  f"{eager_ms:>7.3f}ms {inductor_ms:>8.3f}ms {tilelang_ms:>8.3f}ms "
                  f"{tl_vs_eager:>8.2f}x {tl_vs_ind:>6.2f}x")

            results.append({
                "op": op_name, "shape": shape_name, "M": M, "N": N,
                "eager_ms": eager_ms, "inductor_ms": inductor_ms,
                "tilelang_ms": tilelang_ms,
            })

    # Summary
    print("\n" + "=" * 110)
    print("SUMMARY (geometric mean of TileLang/Inductor ratio per op)")
    print("=" * 110)
    import math
    for op_name in OPERATIONS:
        op_results = [r for r in results if r["op"] == op_name
                      and not math.isnan(r["tilelang_ms"])
                      and not math.isnan(r["inductor_ms"])
                      and r["inductor_ms"] > 0]
        if op_results:
            ratios = [r["tilelang_ms"] / r["inductor_ms"] for r in op_results]
            geo_mean = math.exp(sum(math.log(r) for r in ratios) / len(ratios))
            print(f"  {op_name:<12}: TileLang/Inductor = {geo_mean:.3f}x "
                  f"({'faster' if geo_mean < 1 else 'slower'})")


if __name__ == "__main__":
    main()
