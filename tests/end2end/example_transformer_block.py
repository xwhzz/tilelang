"""Transformer block demo: tilelang.jit(mode='graph') with FlashAttention vs torch.compile.

Demonstrates:
- Hand-written FlashAttention kernel (BHSD layout, bfloat16, WGMMA pipelined)
- Auto-scheduled matmul projections and FFN via graph-mode JIT
- JITKernel auto-detection: graph mode finds the pre-compiled FlashAttention
  in the closure and routes it through torch.export as a custom op
- Comparison against torch.compile with F.scaled_dot_product_attention
"""

from __future__ import annotations

import argparse

import torch
import torch.nn.functional as F

import tilelang
import tilelang.language as T
from tilelang.profiler import do_bench

# ---------------------------------------------------------------------------
# Default configuration
# ---------------------------------------------------------------------------
BATCH = 2
SEQ_LEN = 512
NUM_HEADS = 16
HEAD_DIM = 128
HIDDEN_DIM = NUM_HEADS * HEAD_DIM  # 2048
FFN_DIM = HIDDEN_DIM * 4  # 8192
IS_CAUSAL = True


# ---------------------------------------------------------------------------
# FlashAttention kernel (BHSD layout, bfloat16, WGMMA pipelined for sm_90)
# ---------------------------------------------------------------------------

def _build_flash_attention_func(
    batch: int,
    heads: int,
    seq_q: int,
    seq_kv: int,
    dim: int,
    is_causal: bool,
    block_M: int = 128,
    block_N: int = 128,
    num_stages: int = 2,
    threads: int = 256,
):
    """Build a FlashAttention PrimFunc for the given shapes."""
    scale = (1.0 / dim) ** 0.5 * 1.44269504  # log2(e)
    dtype = T.bfloat16
    accum_dtype = T.float32
    past_len = seq_kv - seq_q

    @T.prim_func
    def flash_attn(
        Q: T.Tensor([batch, heads, seq_q, dim], dtype),
        K: T.Tensor([batch, heads, seq_kv, dim], dtype),
        V: T.Tensor([batch, heads, seq_kv, dim], dtype),
        Output: T.Tensor([batch, heads, seq_q, dim], dtype),
    ):
        with T.Kernel(T.ceildiv(seq_q, block_M), heads, batch, threads=threads) as (bx, by, bz):
            Q_shared = T.alloc_shared([block_M, dim], dtype)
            K_shared = T.alloc_shared([block_N, dim], dtype)
            V_shared = T.alloc_shared([block_N, dim], dtype)
            O_shared = T.alloc_shared([block_M, dim], dtype)
            acc_s = T.alloc_fragment([block_M, block_N], accum_dtype)
            acc_s_cast = T.alloc_fragment([block_M, block_N], dtype)
            acc_o = T.alloc_fragment([block_M, dim], accum_dtype)
            scores_max = T.alloc_fragment([block_M], accum_dtype)
            scores_max_prev = T.alloc_fragment([block_M], accum_dtype)
            scores_scale = T.alloc_fragment([block_M], accum_dtype)
            scores_sum = T.alloc_fragment([block_M], accum_dtype)
            logsum = T.alloc_fragment([block_M], accum_dtype)

            T.copy(Q[bz, by, bx * block_M:(bx + 1) * block_M, :], Q_shared)
            T.fill(acc_o, 0)
            T.fill(logsum, 0)
            T.fill(scores_max, -T.infinity(accum_dtype))

            loop_range = (
                T.min(
                    T.ceildiv(seq_kv, block_N),
                    T.ceildiv((bx + 1) * block_M + past_len, block_N),
                )
                if is_causal
                else T.ceildiv(seq_kv, block_N)
            )

            for k in T.Pipelined(
                loop_range,
                num_stages=num_stages,
                order=[-1, 0, 3, 1, -1, 2],
                stage=[-1, 0, 0, 1, -1, 1],
                group=[[0], [1, 2], [3, 4, 5, 6, 7, 8, 9, 10, 11], [12], [13], [14]],
            ):
                T.copy(K[bz, by, k * block_N:(k + 1) * block_N, :], K_shared)
                if is_causal:
                    for i, j in T.Parallel(block_M, block_N):
                        q_idx = bx * block_M + i + past_len
                        k_idx = k * block_N + j
                        acc_s[i, j] = T.if_then_else(
                            q_idx >= k_idx, 0, -T.infinity(acc_s.dtype))
                else:
                    for i, j in T.Parallel(block_M, block_N):
                        acc_s[i, j] = T.if_then_else(
                            k * block_N + j >= seq_kv, -T.infinity(acc_s.dtype), 0)
                T.gemm(Q_shared, K_shared, acc_s, transpose_B=True,
                        policy=T.GemmWarpPolicy.FullRow)

                T.copy(scores_max, scores_max_prev)
                T.fill(scores_max, -T.infinity(accum_dtype))
                T.reduce_max(acc_s, scores_max, dim=1, clear=False)
                for i in T.Parallel(block_M):
                    scores_max[i] = T.max(scores_max[i], scores_max_prev[i])
                for i in T.Parallel(block_M):
                    scores_scale[i] = T.exp2(
                        scores_max_prev[i] * scale - scores_max[i] * scale)
                for i, j in T.Parallel(block_M, block_N):
                    acc_s[i, j] = T.exp2(
                        acc_s[i, j] * scale - scores_max[i] * scale)
                T.reduce_sum(acc_s, scores_sum, dim=1)
                for i in T.Parallel(block_M):
                    logsum[i] = logsum[i] * scores_scale[i] + scores_sum[i]
                T.copy(acc_s, acc_s_cast)

                for i, j in T.Parallel(block_M, dim):
                    acc_o[i, j] *= scores_scale[i]

                T.copy(V[bz, by, k * block_N:(k + 1) * block_N, :], V_shared)
                T.gemm(acc_s_cast, V_shared, acc_o,
                        policy=T.GemmWarpPolicy.FullRow)

            for i, j in T.Parallel(block_M, dim):
                acc_o[i, j] /= logsum[i]
            T.copy(acc_o, O_shared)
            T.copy(O_shared, Output[bz, by, bx * block_M:(bx + 1) * block_M, :])

    return flash_attn


# ---------------------------------------------------------------------------
# Lazy-compiled FlashAttention kernel (compiled once on first access)
# ---------------------------------------------------------------------------

_flash_attn_cache: dict[tuple, tilelang.JITKernel] = {}


def _get_flash_attn_kernel(
    batch: int = BATCH,
    heads: int = NUM_HEADS,
    seq_q: int = SEQ_LEN,
    seq_kv: int = SEQ_LEN,
    dim: int = HEAD_DIM,
    is_causal: bool = IS_CAUSAL,
) -> tilelang.JITKernel:
    key = (batch, heads, seq_q, seq_kv, dim, is_causal)
    if key not in _flash_attn_cache:
        func = _build_flash_attention_func(*key)
        _flash_attn_cache[key] = tilelang.compile(
            func,
            out_idx=[3],
            pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True},
        )
    return _flash_attn_cache[key]


# ---------------------------------------------------------------------------
# RMSNorm (traced inline by torch.export)
# ---------------------------------------------------------------------------

def _rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    # Compute variance in float32 to avoid bf16 scalar constant issues in Relax frontend.
    x_f32 = x.float()
    variance = (x_f32 * x_f32).mean(-1, keepdim=True)
    scale = torch.rsqrt(variance + eps).to(x.dtype)
    return weight * (x * scale)


# ---------------------------------------------------------------------------
# Transformer block (graph-mode JIT with auto-detected FlashAttention)
# ---------------------------------------------------------------------------

def _build_transformer_block(
    batch: int = BATCH,
    seq_len: int = SEQ_LEN,
    num_heads: int = NUM_HEADS,
    head_dim: int = HEAD_DIM,
    hidden_dim: int = HIDDEN_DIM,
    ffn_dim: int = FFN_DIM,
    is_causal: bool = IS_CAUSAL,
    arch: str | None = None,
    cuda_graph: bool = False,
    native: bool = False,
):
    flash_attn = _get_flash_attn_kernel(batch, num_heads, seq_len, seq_len, head_dim, is_causal)

    @tilelang.jit(
        mode="graph",
        target=arch,
        cuda_graph=cuda_graph,
        native=native,
        pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True},
    )
    def transformer_block(x, w_q, w_k, w_v, w_o, w_gate, w_up, w_down):
        # QKV projections: [B, S, H] @ [H, H] -> [B, S, H]
        # Explicit bf16 casts: Relax matmul legalization upcasts to f32;
        # FlashAttention requires bf16 inputs.
        q = torch.matmul(x, w_q).to(torch.bfloat16)
        k = torch.matmul(x, w_k).to(torch.bfloat16)
        v = torch.matmul(x, w_v).to(torch.bfloat16)

        # Reshape to BHSD: [B, S, H] -> [B, S, Nh, D] -> [B, Nh, S, D]
        q = q.view(batch, seq_len, num_heads, head_dim).permute(0, 2, 1, 3).contiguous()
        k = k.view(batch, seq_len, num_heads, head_dim).permute(0, 2, 1, 3).contiguous()
        v = v.view(batch, seq_len, num_heads, head_dim).permute(0, 2, 1, 3).contiguous()

        # FlashAttention (auto-detected JITKernel from closure)
        attn_out = flash_attn(q, k, v)

        # Reshape back: [B, Nh, S, D] -> [B, S, Nh, D] -> [B, S, H]
        attn_out = attn_out.permute(0, 2, 1, 3).contiguous().view(batch, seq_len, hidden_dim)

        # Output projection + residual
        x = x + torch.matmul(attn_out, w_o)

        # SwiGLU FFN (no explicit bf16 casts — Relax matmul computes in f32,
        # and matmul+cast fusion would break the Matmul schedule rule)
        gate = F.silu(torch.matmul(x, w_gate))
        up = torch.matmul(x, w_up)
        ffn_out = torch.matmul(gate * up, w_down)
        x = x + ffn_out

        return x.to(torch.bfloat16)

    return transformer_block


# ---------------------------------------------------------------------------
# Reference: torch.compile with SDPA
# ---------------------------------------------------------------------------

def _build_reference(
    batch: int = BATCH,
    seq_len: int = SEQ_LEN,
    num_heads: int = NUM_HEADS,
    head_dim: int = HEAD_DIM,
    hidden_dim: int = HIDDEN_DIM,
    is_causal: bool = IS_CAUSAL,
):
    @torch.compile()
    def transformer_block_ref(x, w_q, w_k, w_v, w_o, w_gate, w_up, w_down):
        q = torch.matmul(x, w_q)
        k = torch.matmul(x, w_k)
        v = torch.matmul(x, w_v)

        q = q.view(batch, seq_len, num_heads, head_dim).permute(0, 2, 1, 3)
        k = k.view(batch, seq_len, num_heads, head_dim).permute(0, 2, 1, 3)
        v = v.view(batch, seq_len, num_heads, head_dim).permute(0, 2, 1, 3)

        attn_out = F.scaled_dot_product_attention(q, k, v, is_causal=is_causal)

        attn_out = attn_out.permute(0, 2, 1, 3).contiguous().view(batch, seq_len, hidden_dim)
        x = x + torch.matmul(attn_out, w_o)

        gate = F.silu(torch.matmul(x, w_gate))
        up = torch.matmul(x, w_up)
        ffn_out = torch.matmul(gate * up, w_down)
        x = x + ffn_out

        return x

    return transformer_block_ref


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------

def _make_inputs(
    batch: int = BATCH,
    seq_len: int = SEQ_LEN,
    hidden_dim: int = HIDDEN_DIM,
    ffn_dim: int = FFN_DIM,
    dtype: torch.dtype = torch.bfloat16,
):
    torch.manual_seed(42)
    x = torch.randn(batch, seq_len, hidden_dim, device="cuda", dtype=dtype)
    w_q = torch.randn(hidden_dim, hidden_dim, device="cuda", dtype=dtype) * 0.02
    w_k = torch.randn(hidden_dim, hidden_dim, device="cuda", dtype=dtype) * 0.02
    w_v = torch.randn(hidden_dim, hidden_dim, device="cuda", dtype=dtype) * 0.02
    w_o = torch.randn(hidden_dim, hidden_dim, device="cuda", dtype=dtype) * 0.02
    w_gate = torch.randn(hidden_dim, ffn_dim, device="cuda", dtype=dtype) * 0.02
    w_up = torch.randn(hidden_dim, ffn_dim, device="cuda", dtype=dtype) * 0.02
    w_down = torch.randn(ffn_dim, hidden_dim, device="cuda", dtype=dtype) * 0.02
    return (x, w_q, w_k, w_v, w_o, w_gate, w_up, w_down)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def check_only(arch: str | None) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required.")

    args = _make_inputs()
    block = _build_transformer_block(arch=arch)
    runner = block.compile(*args)

    print("=== Scheduled Relax Module ===")
    print(runner.scheduled_mod.script())


def build_and_run(
    arch: str | None,
    bench_backend: str,
    cuda_graph: bool,
    native: bool,
) -> tuple[float, float]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required.")

    args = _make_inputs()

    block = _build_transformer_block(arch=arch, cuda_graph=cuda_graph, native=native)
    runner = block.compile(*args)
    tl_out = runner(*args)

    ref_fn = _build_reference()
    ref_out = ref_fn(*args)

    torch.testing.assert_close(tl_out, ref_out, rtol=0.1, atol=0.1)
    print("\033[92mCorrectness check passed.\033[0m")

    tl_time = do_bench(lambda: runner(*args), backend=bench_backend)
    ref_time = do_bench(lambda: ref_fn(*args), backend=bench_backend)

    print(f"tilelang graph:  {tl_time:.3f} ms")
    print(f"torch.compile:   {ref_time:.3f} ms")
    print(f"speedup:         {ref_time / tl_time:.2f}x")

    if cuda_graph:
        cg_time = do_bench(lambda: runner(*args), backend=bench_backend)
        print(f"tilelang CUDA graph: {cg_time:.3f} ms")

    return tl_time, ref_time


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Transformer block: tilelang graph-mode vs torch.compile")
    parser.add_argument("--arch", type=str, default=None,
                        help='CUDA arch, e.g. "sm_90a".')
    parser.add_argument("--bench-backend", type=str, default="event",
                        choices=["event", "cuda"])
    parser.add_argument("--cuda-graph", action="store_true",
                        help="Enable CUDA graph capture.")
    parser.add_argument("--native", action="store_true",
                        help="Enable C++ native dispatch.")
    parser.add_argument("--check-only", action="store_true",
                        help="Only compile and print scheduled IR.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.check_only:
        check_only(args.arch)
        print("\033[92mCompilation check passed.\033[0m")
    else:
        build_and_run(args.arch, args.bench_backend, args.cuda_graph, args.native)


"""
Usage:
python tests/end2end/example_transformer_block.py --arch sm_90a
python tests/end2end/example_transformer_block.py --arch sm_90a --cuda-graph
python tests/end2end/example_transformer_block.py --arch sm_90a --native
python tests/end2end/example_transformer_block.py --arch sm_90a --check-only
"""
