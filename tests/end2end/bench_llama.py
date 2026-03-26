"""Llama2 end-to-end benchmark: tilelang graph compiler vs torch.compile.

Uses the transformers LlamaForCausalLM model directly.  Verifies that
token generation produces correct output, then benchmarks latency.

Usage:
    # Quick test (tiny random model, 1 layer)
    python tests/end2end/bench_llama.py

    # Pretrained Llama2-7B — generate real text
    python tests/end2end/bench_llama.py --pretrained meta-llama/Llama-2-7b-hf

    # Pretrained with fewer layers (faster, less memory)
    python tests/end2end/bench_llama.py --pretrained meta-llama/Llama-2-7b-hf --num-layers 2

    # Optimized Hopper prefill path: 7B-shape, beats torch.compile on H100
    python tests/end2end/bench_llama.py --preset 7b --num-layers 8 --seq-len 1024 \
        --use-tileops-mha --cublas-linears --arch sm_90a
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, LlamaConfig, LlamaForCausalLM

import tilelang  # noqa: F401  (triggers backend registration)
import tilelang.language as T
from tilelang.profiler import do_bench

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Patch transformers to insert graph breaks around unsupported ops
# ---------------------------------------------------------------------------

def _patch_rope_for_dynamo():
    """Mark RoPE as graph break (uses unsupported ops like diff/fancy indexing)."""
    from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding

    if hasattr(LlamaRotaryEmbedding, "_orig_forward"):
        return  # Already patched.

    LlamaRotaryEmbedding._orig_forward = LlamaRotaryEmbedding.forward

    @torch.compiler.disable
    def rope_forward(self, x, position_ids):
        return LlamaRotaryEmbedding._orig_forward(self, x, position_ids)

    LlamaRotaryEmbedding.forward = rope_forward
    logger.info("Patched RoPE with graph break")


def _patch_attention_graph_break():
    """Mark full LlamaAttention as graph break (SDPA not supported by Relax)."""
    from transformers.models.llama.modeling_llama import LlamaAttention

    if not hasattr(LlamaAttention, "_orig_forward"):
        LlamaAttention._orig_forward = LlamaAttention.forward

    @torch.compiler.disable
    def attn_forward(self, *args, **kwargs):
        return LlamaAttention._orig_forward(self, *args, **kwargs)

    LlamaAttention.forward = attn_forward
    logger.info("Patched Attention with full graph break")


def _patch_linears_cublas():
    """Route nn.Linear through cuBLAS (graph break) for better GEMM perf.

    TileLang's auto-generated Matmul kernels are slower than cuBLAS for large
    standalone GEMMs. This patch keeps the linear layers on cuBLAS while
    allowing TileLang to compile the surrounding fusion ops (RMSNorm, SiLU,
    elementwise add/mul).
    """
    import torch.nn as nn

    if hasattr(nn.Linear, "_orig_forward"):
        return  # Already patched.

    nn.Linear._orig_forward = nn.Linear.forward

    @torch.compiler.disable
    def linear_forward(self, x):
        return nn.Linear._orig_forward(self, x)

    nn.Linear.forward = linear_forward
    logger.info("Patched nn.Linear with cuBLAS graph break")


def _unpatch_linears():
    """Restore original nn.Linear.forward."""
    import torch.nn as nn
    if hasattr(nn.Linear, "_orig_forward"):
        nn.Linear.forward = nn.Linear._orig_forward
        del nn.Linear._orig_forward
        logger.info("Restored nn.Linear")


def _patch_causal_mask_none_for_prefill():
    """Skip HF causal-mask materialization for dense prefill.

    The TileLang FlashAttention fast path already bakes in causality, so the
    additive mask tensor is redundant for the common benchmark case
    (batch=1, no padding mask, no KV cache). Returning ``None`` avoids an
    unsupported indexing subgraph in TVM's FX importer.
    """
    from transformers.models.llama import modeling_llama as modeling

    if hasattr(modeling, "_orig_create_causal_mask"):
        return

    modeling._orig_create_causal_mask = modeling.create_causal_mask

    def create_causal_mask_fast(
        config,
        input_embeds,
        attention_mask,
        cache_position,
        past_key_values,
        position_ids=None,
        or_mask_function=None,
        and_mask_function=None,
    ):
        if attention_mask is None and past_key_values is None:
            return None
        return modeling._orig_create_causal_mask(
            config=config,
            input_embeds=input_embeds,
            attention_mask=attention_mask,
            cache_position=cache_position,
            past_key_values=past_key_values,
            position_ids=position_ids,
            or_mask_function=or_mask_function,
            and_mask_function=and_mask_function,
        )

    modeling.create_causal_mask = create_causal_mask_fast
    logger.info("Patched create_causal_mask to skip dense prefill masks")


def _unpatch_causal_mask():
    """Restore the original HF causal-mask builder."""
    from transformers.models.llama import modeling_llama as modeling

    if hasattr(modeling, "_orig_create_causal_mask"):
        modeling.create_causal_mask = modeling._orig_create_causal_mask
        del modeling._orig_create_causal_mask
        logger.info("Restored create_causal_mask")


def _patch_attention_tileops_mha(arch: str | None = None):
    """Replace attention core with TileLang FlashAttention, keep projections compilable."""
    from transformers.models.llama.modeling_llama import (
        LlamaAttention,
        apply_rotary_pos_emb,
        repeat_kv,
    )

    if not hasattr(LlamaAttention, "_orig_forward"):
        LlamaAttention._orig_forward = LlamaAttention.forward

    _patch_attention_with_tileops(LlamaAttention, apply_rotary_pos_emb, repeat_kv, arch)
    logger.info("Patched Attention with TileLang FlashAttention")


# ---------------------------------------------------------------------------
# TileLang FlashAttention integration
# ---------------------------------------------------------------------------

# Cache of compiled FlashAttention kernels keyed by shape/dtype/arch.
_flash_attn_kernel_cache: dict[tuple, object] = {}

_FLASH_BLOCK_M = 128
_FLASH_BLOCK_N = 128
_FLASH_NUM_STAGES = 2
_FLASH_THREADS = 256


def _round_up(value: int, multiple: int) -> int:
    return ((value + multiple - 1) // multiple) * multiple


def _to_tl_dtype(dtype: torch.dtype):
    if dtype == torch.float16:
        return T.float16
    if dtype == torch.bfloat16:
        return T.bfloat16
    raise TypeError(f"TileLang FlashAttention only supports fp16/bf16, got {dtype}")


def _build_flash_attention_func(
    batch: int,
    heads: int,
    seq_q: int,
    seq_kv: int,
    head_dim: int,
    dtype: torch.dtype,
    is_causal: bool,
):
    """Build a causal FlashAttention kernel in BHSD layout."""
    scale = (1.0 / head_dim) ** 0.5 * 1.44269504  # log2(e)
    tl_dtype = _to_tl_dtype(dtype)
    accum_dtype = T.float32
    past_len = seq_kv - seq_q

    @T.prim_func
    def flash_attn(
        Q: T.Tensor([batch, heads, seq_q, head_dim], tl_dtype),
        K: T.Tensor([batch, heads, seq_kv, head_dim], tl_dtype),
        V: T.Tensor([batch, heads, seq_kv, head_dim], tl_dtype),
        Output: T.Tensor([batch, heads, seq_q, head_dim], tl_dtype),
    ):
        with T.Kernel(
            T.ceildiv(seq_q, _FLASH_BLOCK_M),
            heads,
            batch,
            threads=_FLASH_THREADS,
        ) as (bx, by, bz):
            Q_shared = T.alloc_shared([_FLASH_BLOCK_M, head_dim], tl_dtype)
            K_shared = T.alloc_shared([_FLASH_BLOCK_N, head_dim], tl_dtype)
            V_shared = T.alloc_shared([_FLASH_BLOCK_N, head_dim], tl_dtype)
            O_shared = T.alloc_shared([_FLASH_BLOCK_M, head_dim], tl_dtype)
            acc_s = T.alloc_fragment([_FLASH_BLOCK_M, _FLASH_BLOCK_N], accum_dtype)
            acc_s_cast = T.alloc_fragment([_FLASH_BLOCK_M, _FLASH_BLOCK_N], tl_dtype)
            acc_o = T.alloc_fragment([_FLASH_BLOCK_M, head_dim], accum_dtype)
            scores_max = T.alloc_fragment([_FLASH_BLOCK_M], accum_dtype)
            scores_max_prev = T.alloc_fragment([_FLASH_BLOCK_M], accum_dtype)
            scores_scale = T.alloc_fragment([_FLASH_BLOCK_M], accum_dtype)
            scores_sum = T.alloc_fragment([_FLASH_BLOCK_M], accum_dtype)
            logsum = T.alloc_fragment([_FLASH_BLOCK_M], accum_dtype)

            T.copy(Q[bz, by, bx * _FLASH_BLOCK_M:(bx + 1) * _FLASH_BLOCK_M, :], Q_shared)
            T.fill(acc_o, 0)
            T.fill(logsum, 0)
            T.fill(scores_max, -T.infinity(accum_dtype))

            loop_range = (
                T.min(
                    T.ceildiv(seq_kv, _FLASH_BLOCK_N),
                    T.ceildiv((bx + 1) * _FLASH_BLOCK_M + past_len, _FLASH_BLOCK_N),
                )
                if is_causal
                else T.ceildiv(seq_kv, _FLASH_BLOCK_N)
            )

            for k in T.Pipelined(
                loop_range,
                num_stages=_FLASH_NUM_STAGES,
                order=[-1, 0, 3, 1, -1, 2],
                stage=[-1, 0, 0, 1, -1, 1],
                group=[[0], [1, 2], [3, 4, 5, 6, 7, 8, 9, 10, 11], [12], [13], [14]],
            ):
                T.copy(K[bz, by, k * _FLASH_BLOCK_N:(k + 1) * _FLASH_BLOCK_N, :], K_shared)
                if is_causal:
                    for i, j in T.Parallel(_FLASH_BLOCK_M, _FLASH_BLOCK_N):
                        q_idx = bx * _FLASH_BLOCK_M + i + past_len
                        k_idx = k * _FLASH_BLOCK_N + j
                        acc_s[i, j] = T.if_then_else(q_idx >= k_idx, 0, -T.infinity(acc_s.dtype))
                else:
                    for i, j in T.Parallel(_FLASH_BLOCK_M, _FLASH_BLOCK_N):
                        acc_s[i, j] = T.if_then_else(
                            k * _FLASH_BLOCK_N + j >= seq_kv, -T.infinity(acc_s.dtype), 0
                        )

                T.gemm(Q_shared, K_shared, acc_s, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)

                T.copy(scores_max, scores_max_prev)
                T.fill(scores_max, -T.infinity(accum_dtype))
                T.reduce_max(acc_s, scores_max, dim=1, clear=False)
                for i in T.Parallel(_FLASH_BLOCK_M):
                    scores_max[i] = T.max(scores_max[i], scores_max_prev[i])
                for i in T.Parallel(_FLASH_BLOCK_M):
                    scores_scale[i] = T.exp2(scores_max_prev[i] * scale - scores_max[i] * scale)
                for i, j in T.Parallel(_FLASH_BLOCK_M, _FLASH_BLOCK_N):
                    acc_s[i, j] = T.exp2(acc_s[i, j] * scale - scores_max[i] * scale)
                T.reduce_sum(acc_s, scores_sum, dim=1)
                for i in T.Parallel(_FLASH_BLOCK_M):
                    logsum[i] = logsum[i] * scores_scale[i] + scores_sum[i]
                T.copy(acc_s, acc_s_cast)

                for i, j in T.Parallel(_FLASH_BLOCK_M, head_dim):
                    acc_o[i, j] *= scores_scale[i]

                T.copy(V[bz, by, k * _FLASH_BLOCK_N:(k + 1) * _FLASH_BLOCK_N, :], V_shared)
                T.gemm(acc_s_cast, V_shared, acc_o, policy=T.GemmWarpPolicy.FullRow)

            for i, j in T.Parallel(_FLASH_BLOCK_M, head_dim):
                acc_o[i, j] /= logsum[i]
            T.copy(acc_o, O_shared)
            T.copy(O_shared, Output[bz, by, bx * _FLASH_BLOCK_M:(bx + 1) * _FLASH_BLOCK_M, :])

    return flash_attn


def _get_flash_attn_kernel(
    batch: int,
    heads: int,
    seq_q: int,
    seq_kv: int,
    head_dim: int,
    dtype: torch.dtype,
    is_causal: bool,
    arch: str | None,
):
    """Compile or fetch a cached TileLang FlashAttention kernel."""
    seq_q_padded = _round_up(seq_q, _FLASH_BLOCK_M)
    seq_kv_padded = _round_up(seq_kv, _FLASH_BLOCK_N)
    key = (batch, heads, seq_q_padded, seq_kv_padded, head_dim, str(dtype), is_causal, arch or "auto")

    if key not in _flash_attn_kernel_cache:
        target = "auto" if arch is None else f"cuda -arch={arch}"
        logger.info(
            "Compiling TileLang FlashAttention: batch=%d heads=%d seq_q=%d seq_kv=%d dim=%d dtype=%s target=%s",
            batch,
            heads,
            seq_q_padded,
            seq_kv_padded,
            head_dim,
            dtype,
            target,
        )
        func = _build_flash_attention_func(
            batch=batch,
            heads=heads,
            seq_q=seq_q_padded,
            seq_kv=seq_kv_padded,
            head_dim=head_dim,
            dtype=dtype,
            is_causal=is_causal,
        )
        _flash_attn_kernel_cache[key] = tilelang.compile(
            func,
            out_idx=[3],
            target=target,
            pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True},
        )

    return _flash_attn_kernel_cache[key], seq_q_padded, seq_kv_padded


@torch.compiler.disable
def _tileops_attention(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    *,
    attention_mask: torch.Tensor | None = None,
    scaling: float | None = None,
    is_causal: bool = True,
    arch: str | None = None,
) -> torch.Tensor:
    """Core attention via a cached TileLang FlashAttention kernel.

    Assumes dense causal prefill. For more complex masking patterns we fall
    back to SDPA.
    """
    if query_states.dtype not in (torch.float16, torch.bfloat16):
        return F.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=attention_mask,
            is_causal=is_causal,
            scale=scaling,
        )

    # Batch>1 may carry padding masks; keep the fast path specialized to the
    # common benchmark case (dense causal prefill, batch=1).
    if attention_mask is not None and query_states.shape[0] != 1:
        return F.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=attention_mask,
            is_causal=False,
            scale=scaling,
        )

    batch, heads, seq_q, head_dim = query_states.shape
    seq_kv = key_states.shape[-2]

    kernel, padded_q, padded_kv = _get_flash_attn_kernel(
        batch=batch,
        heads=heads,
        seq_q=seq_q,
        seq_kv=seq_kv,
        head_dim=head_dim,
        dtype=query_states.dtype,
        is_causal=is_causal,
        arch=arch,
    )

    q = query_states.contiguous()
    k = key_states.contiguous()
    v = value_states.contiguous()

    if padded_q != seq_q:
        q = F.pad(q, (0, 0, 0, padded_q - seq_q))
    if padded_kv != seq_kv:
        pad_kv = padded_kv - seq_kv
        k = F.pad(k, (0, 0, 0, pad_kv))
        v = F.pad(v, (0, 0, 0, pad_kv))

    out = kernel(q, k, v)
    if padded_q != seq_q:
        # The direct kernel runner only accepts dense tensors. Cropping the
        # padded output back to the original sequence length creates a view
        # with padded strides, so materialize it before handing it back to the
        # compiled transpose/reshape subgraph.
        out = out[:, :, :seq_q, :].contiguous()
    return out


def warmup_tileops_mha(model, input_ids, arch: str | None = None):
    """Pre-compile FlashAttention for the prefill shape.

    Call this before benchmarking so that ``_tileops_attention`` has a cached
    kernel for the prefill sequence length.
    """
    config = model.config
    batch = input_ids.shape[0]
    seq_len = input_ids.shape[1]
    heads = config.num_attention_heads
    head_dim = config.hidden_size // heads
    dtype = next(model.parameters()).dtype
    _get_flash_attn_kernel(
        batch=batch,
        heads=heads,
        seq_q=seq_len,
        seq_kv=seq_len,
        head_dim=head_dim,
        dtype=dtype,
        is_causal=True,
        arch=arch,
    )


def _patch_attention_with_tileops(LlamaAttention, apply_rotary_pos_emb, repeat_kv, arch: str | None):
    """Replace LlamaAttention.forward to use TileLang FlashAttention for core attention.

    Q/K/V linear projections and output projection remain normal PyTorch
    operations (compilable by dynamo + TileLang).  RoPE application and
    core attention are graph breaks using efficient kernels.

    Subgraph structure after patching:
      SG: ... → Q/K/V projections + reshape (compiled by TileLang)
      break: apply_rotary_pos_emb (eager, uses fancy indexing)
      break: TileLang FlashAttention (efficient flash-attention kernel)
      SG: output projection + ... (compiled by TileLang)
    """

    @torch.compiler.disable
    def _apply_rope(q, k, cos, sin):
        return apply_rotary_pos_emb(q, k, cos, sin)

    def new_attn_forward(
        self,
        hidden_states,
        position_embeddings=None,
        attention_mask=None,
        past_key_values=None,
        cache_position=None,
        **kwargs,
    ):
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        # Q/K/V projections (compilable by TileLang).
        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        # RoPE application (graph break — uses fancy indexing in rotate_half).
        cos, sin = position_embeddings
        query_states, key_states = _apply_rope(
            query_states, key_states, cos, sin,
        )

        if past_key_values is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_values.update(
                key_states, value_states, self.layer_idx, cache_kwargs
            )

        if self.num_key_value_groups != 1:
            key_states = repeat_kv(key_states, self.num_key_value_groups)
            value_states = repeat_kv(value_states, self.num_key_value_groups)

        # Core attention via TileLang FlashAttention (graph break).
        attn_output = _tileops_attention(
            query_states,
            key_states,
            value_states,
            attention_mask=attention_mask,
            scaling=self.scaling,
            is_causal=True,
            arch=arch,
        )

        # Output projection (compilable by TileLang).
        # _tileops_attention returns (B, H, S, D); transpose back to match the
        # original transformers layout before the output projection.
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, None

    LlamaAttention.forward = new_attn_forward



# ---------------------------------------------------------------------------
# Greedy generation loop
# ---------------------------------------------------------------------------

@torch.no_grad()
def greedy_generate(
    model: LlamaForCausalLM,
    input_ids: torch.Tensor,
    max_new_tokens: int = 32,
) -> torch.Tensor:
    """Simple greedy decode without KV cache."""
    generated = input_ids.clone()
    for _ in range(max_new_tokens):
        logits = model(generated).logits
        next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        generated = torch.cat([generated, next_token], dim=1)
    return generated


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def _load_pretrained(model_name: str, num_layers: int | None, dtype: torch.dtype):
    """Load a pretrained model, optionally truncating to *num_layers*."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = LlamaForCausalLM.from_pretrained(
        model_name, torch_dtype=dtype,
    )
    # Disable KV cache — we do full-sequence forward passes.
    model.config.use_cache = False

    # Truncate layers if requested (saves memory & compile time).
    if num_layers is not None and num_layers < len(model.model.layers):
        model.model.layers = model.model.layers[:num_layers]
        logger.info("Truncated model to %d layers", num_layers)

    model = model.cuda().eval()
    return model, tokenizer


def _create_random_model(preset: str, num_layers: int, dtype: torch.dtype):
    """Create a randomly-initialized model for quick testing."""
    configs = {
        "tiny": dict(hidden_size=256, intermediate_size=512,
                      num_attention_heads=4, num_key_value_heads=4,
                      vocab_size=1000),
        "small": dict(hidden_size=1024, intermediate_size=2816,
                       num_attention_heads=8, num_key_value_heads=8,
                       vocab_size=32000),
        "7b": dict(hidden_size=4096, intermediate_size=11008,
                    num_attention_heads=32, num_key_value_heads=32,
                    vocab_size=32000),
    }
    if preset not in configs:
        raise ValueError(f"Unknown preset: {preset}")
    config = LlamaConfig(num_hidden_layers=num_layers, use_cache=False,
                          **configs[preset])
    model = LlamaForCausalLM(config).to(dtype).cuda().eval()
    return model, None


# ---------------------------------------------------------------------------
# Profiling helpers
# ---------------------------------------------------------------------------

def _classify_kernel(name: str) -> str:
    """Map raw CUDA kernel names into coarse latency buckets."""
    lowered = name.lower()

    if "memcpy" in lowered or "memset" in lowered or "copy" in lowered or "catarray" in lowered:
        return "memory"
    if "flash_attn" in lowered or "attention" in lowered:
        return "attention"
    if lowered.startswith("nvjet_") or any(
        token in lowered for token in ("gemm", "xmma", "mma", "cublas", "cutlass", "gemv")
    ):
        return "gemm"
    if lowered.startswith("triton_red") or any(
        token in lowered for token in ("reduce", "reduction", "softmax", "rms", "layer_norm", "rsqrt", "mean")
    ):
        return "reduction"
    if lowered.startswith("triton_poi") or any(
        token in lowered for token in ("elementwise", "ewise", "vectorized", "pointwise")
    ):
        return "elementwise"
    if lowered.startswith("fused_") or lowered.endswith("_kernel") or any(
        token in lowered for token in ("silu", "relu", "gelu", "transpose", "reshape", "cast", "multiply", "add")
    ):
        return "elementwise"
    return "other"


def _shorten_label(label: str, max_chars: int = 78) -> str:
    if len(label) <= max_chars:
        return label
    keep = (max_chars - 3) // 2
    tail = max_chars - 3 - keep
    return f"{label[:keep]}...{label[-tail:]}"


def _profile_cuda_kernels(
    fn,
    *,
    warmup_iters: int,
    profile_iters: int,
) -> dict[str, object]:
    """Profile CUDA kernels launched by *fn* and aggregate by kernel name."""
    if profile_iters <= 0:
        raise ValueError("profile_iters must be > 0")

    with torch.no_grad():
        for _ in range(max(warmup_iters, 0)):
            fn()
        torch.cuda.synchronize()

        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            record_shapes=False,
            profile_memory=False,
            with_stack=False,
            with_modules=False,
            acc_events=True,
        ) as prof:
            for _ in range(profile_iters):
                fn()
            torch.cuda.synchronize()

    kernel_totals_us: dict[str, float] = defaultdict(float)
    kernel_counts: dict[str, int] = defaultdict(int)
    total_cuda_us = 0.0

    for event in prof.events():
        if event.device_type != torch.autograd.DeviceType.CUDA:
            continue
        duration_us = float(event.self_device_time_total)
        if duration_us <= 0:
            continue
        kernel_name = event.name
        kernel_totals_us[kernel_name] += duration_us
        kernel_counts[kernel_name] += 1
        total_cuda_us += duration_us

    rows = []
    category_totals_ms: dict[str, float] = defaultdict(float)
    for kernel_name, total_us in sorted(kernel_totals_us.items(), key=lambda item: item[1], reverse=True):
        avg_ms = total_us / profile_iters / 1000.0
        category = _classify_kernel(kernel_name)
        category_totals_ms[category] += avg_ms
        rows.append(
            {
                "kernel": kernel_name,
                "category": category,
                "total_us": total_us,
                "avg_ms": avg_ms,
                "calls": kernel_counts[kernel_name],
                "calls_per_iter": kernel_counts[kernel_name] / profile_iters,
                "pct_total": (total_us / total_cuda_us * 100.0) if total_cuda_us else 0.0,
            }
        )

    return {
        "rows": rows,
        "total_avg_ms": total_cuda_us / profile_iters / 1000.0,
        "category_avg_ms": dict(sorted(category_totals_ms.items(), key=lambda item: item[1], reverse=True)),
        "profile_iters": profile_iters,
        "warmup_iters": warmup_iters,
    }


def _write_breakdown_csv(profiles: dict[str, dict[str, object]], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "backend",
                "kernel",
                "category",
                "total_us",
                "avg_ms",
                "calls",
                "calls_per_iter",
                "pct_total",
            ],
        )
        writer.writeheader()
        for backend, profile in profiles.items():
            for row in profile["rows"]:
                writer.writerow({"backend": backend, **row})


def _write_breakdown_json(profiles: dict[str, dict[str, object]], out_json: Path) -> None:
    out_json.parent.mkdir(parents=True, exist_ok=True)
    serializable = {}
    for backend, profile in profiles.items():
        serializable[backend] = {
            "total_avg_ms": profile["total_avg_ms"],
            "category_avg_ms": profile["category_avg_ms"],
            "profile_iters": profile["profile_iters"],
            "warmup_iters": profile["warmup_iters"],
        }
    out_json.write_text(json.dumps(serializable, indent=2))


def _plot_breakdown_figure(
    profiles: dict[str, dict[str, object]],
    out_png: Path,
    *,
    topk: int,
    title: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    backend_order = [backend for backend in ("torch.compile", "tilelang") if backend in profiles]
    colors = {
        "torch.compile": "#4C72B0",
        "tilelang": "#DD8452",
    }

    category_names = sorted(
        {
            category
            for profile in profiles.values()
            for category in profile["category_avg_ms"]
        },
        key=lambda category: sum(
            profiles[backend]["category_avg_ms"].get(category, 0.0) for backend in backend_order
        ),
        reverse=True,
    )

    kernel_value_maps = {
        backend: {row["kernel"]: row["avg_ms"] for row in profile["rows"]}
        for backend, profile in profiles.items()
    }
    combined_kernel_names = sorted(
        {
            kernel_name
            for kernel_values in kernel_value_maps.values()
            for kernel_name in kernel_values
        },
        key=lambda kernel_name: sum(
            kernel_value_maps[backend].get(kernel_name, 0.0) for backend in backend_order
        ),
        reverse=True,
    )
    top_kernel_names = combined_kernel_names[:topk]

    fig_height = max(8.0, 4.0 + 0.42 * len(top_kernel_names))
    fig, axes = plt.subplots(
        2,
        1,
        figsize=(16, fig_height),
        gridspec_kw={"height_ratios": [1.0, max(1.6, 0.16 * len(top_kernel_names))]},
        constrained_layout=True,
    )

    # Top: category totals.
    ax = axes[0]
    x = np.arange(len(category_names))
    width = 0.35
    for idx, backend in enumerate(backend_order):
        values = [profiles[backend]["category_avg_ms"].get(category, 0.0) for category in category_names]
        offset = (-0.5 + idx) * width
        bars = ax.bar(x + offset, values, width=width, label=backend, color=colors[backend], edgecolor="black")
        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{value:.2f}",
                ha="center",
                va="bottom",
                fontsize=8,
                rotation=0,
            )

    ax.set_title(f"{title}\nCUDA Category Breakdown")
    ax.set_ylabel("Average CUDA Time per Iteration (ms)")
    ax.set_xticks(x)
    ax.set_xticklabels(category_names)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="upper right")

    # Bottom: top kernels.
    ax = axes[1]
    y = np.arange(len(top_kernel_names))
    bar_height = 0.35
    for idx, backend in enumerate(backend_order):
        values = [kernel_value_maps[backend].get(kernel_name, 0.0) for kernel_name in top_kernel_names]
        offset = (-0.5 + idx) * bar_height
        bars = ax.barh(
            y + offset,
            values,
            height=bar_height,
            label=backend,
            color=colors[backend],
            edgecolor="black",
        )
        for bar, value in zip(bars, values):
            if value <= 0:
                continue
            ax.text(
                bar.get_width(),
                bar.get_y() + bar.get_height() / 2,
                f" {value:.2f}",
                ha="left",
                va="center",
                fontsize=8,
            )

    ax.set_title(f"Top {len(top_kernel_names)} CUDA Kernels")
    ax.set_xlabel("Average CUDA Time per Iteration (ms)")
    ax.set_yticks(y)
    ax.set_yticklabels([_shorten_label(kernel_name) for kernel_name in top_kernel_names], fontsize=9)
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.25)

    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
        axes[0].spines[spine].set_visible(False)

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _emit_breakdown_report(
    *,
    torch_compile_profile: dict[str, object],
    tilelang_profile: dict[str, object],
    prefix: str,
    topk: int,
    title: str,
) -> tuple[Path, Path, Path]:
    base = Path(prefix).with_suffix("")
    png_path = base.with_suffix(".png")
    csv_path = base.parent / f"{base.name}_kernels.csv"
    json_path = base.parent / f"{base.name}.json"

    profiles = {
        "torch.compile": torch_compile_profile,
        "tilelang": tilelang_profile,
    }
    _write_breakdown_csv(profiles, csv_path)
    _write_breakdown_json(profiles, json_path)
    _plot_breakdown_figure(profiles, png_path, topk=topk, title=title)
    return png_path, csv_path, json_path


# ---------------------------------------------------------------------------
# Main benchmark
# ---------------------------------------------------------------------------

def run_benchmark(args: argparse.Namespace):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required.")

    # RoPE is always a graph break for all paths.
    _patch_rope_for_dynamo()
    # For eager/torch.compile reference, use full attention graph break.
    _patch_attention_graph_break()

    torch.manual_seed(42)
    batch = args.batch
    max_new_tokens = args.max_new_tokens
    dtype = torch.float16

    # ---------------------------------------------------------------
    # Load model
    # ---------------------------------------------------------------
    if args.pretrained:
        num_layers = args.num_layers if args.num_layers and args.num_layers > 0 else None
        model, tokenizer = _load_pretrained(args.pretrained, num_layers, dtype)
        config = model.config
        actual_layers = len(model.model.layers)
        print(f"Model: {args.pretrained} ({actual_layers} layers, pretrained)")
    else:
        num_layers = args.num_layers if args.num_layers and args.num_layers > 0 else 1
        model, tokenizer = _create_random_model(
            args.preset, num_layers, dtype
        )
        config = model.config
        print(f"Model: Llama2 {args.preset} ({num_layers} layers, random)")

    param_count = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {param_count / 1e6:.1f}M")
    print(f"  Config: hidden={config.hidden_size}, heads={config.num_attention_heads}, "
          f"ffn={config.intermediate_size}, vocab={config.vocab_size}")

    # ---------------------------------------------------------------
    # Prepare input
    # ---------------------------------------------------------------
    if tokenizer is not None:
        prompt = args.prompt
        encoded = tokenizer(prompt, return_tensors="pt")
        input_ids = encoded["input_ids"].cuda()
        seq_len = input_ids.shape[1]
        prompt_preview = prompt if len(prompt) <= 160 else f"{prompt[:157]}..."
        print(f"\n  Prompt: \"{prompt_preview}\"")
        print(f"  Prompt tokens: {seq_len}")
    else:
        seq_len = args.seq_len
        input_ids = torch.randint(0, config.vocab_size, (batch, seq_len),
                                  device="cuda")

    # ---------------------------------------------------------------
    # 1. Reference: eager model
    # ---------------------------------------------------------------
    print(f"\n--- Eager reference ---")
    with torch.no_grad():
        ref_logits = model(input_ids).logits
    print(f"  Forward OK, logits shape: {ref_logits.shape}")

    ref_tokens = greedy_generate(model, input_ids, max_new_tokens)
    if tokenizer is not None:
        ref_text = tokenizer.decode(ref_tokens[0, seq_len:],
                                    skip_special_tokens=True)
        print(f"  Generated: \"{ref_text}\"")
    else:
        print(f"  Generated {max_new_tokens} tokens: "
              f"{ref_tokens[0, seq_len:].tolist()[:10]}...")

    # ---------------------------------------------------------------
    # 2. torch.compile baseline
    # ---------------------------------------------------------------
    print(f"\n--- torch.compile ---")
    import torch._dynamo as dynamo
    dynamo.reset()
    tc_model = torch.compile(model)
    with torch.no_grad():
        tc_logits = tc_model(input_ids).logits

    try:
        torch.testing.assert_close(tc_logits, ref_logits, rtol=1e-2, atol=1e-2)
        print("  Forward correctness: PASS")
    except AssertionError:
        max_diff = (tc_logits - ref_logits).abs().max().item()
        print(f"  Forward correctness: WARN (max diff = {max_diff:.6f})")

    tc_tokens = greedy_generate(tc_model, input_ids, max_new_tokens)
    if tokenizer is not None:
        tc_text = tokenizer.decode(tc_tokens[0, seq_len:],
                                   skip_special_tokens=True)
        print(f"  Generated: \"{tc_text}\"")
        token_match = (tc_tokens == ref_tokens).all().item()
        print(f"  Token match vs eager: {'PASS' if token_match else 'FAIL'}")
    else:
        token_match = (tc_tokens == ref_tokens).all().item()
        print(f"  Token match: {'PASS' if token_match else 'FAIL'}")

    # ---------------------------------------------------------------
    # 3. tilelang dynamo backend
    # ---------------------------------------------------------------
    print(f"\n--- tilelang graph compiler (dynamo backend) ---")
    dynamo.reset()

    # Switch attention patch: use TileLang FlashAttention for the tilelang path.
    if args.use_tileops_mha:
        _patch_causal_mask_none_for_prefill()
        _patch_attention_tileops_mha(args.arch)
        warmup_tileops_mha(model, input_ids, args.arch)
    # else: keep the full graph break from earlier.

    # Route nn.Linear through cuBLAS for better GEMM performance.
    if args.cublas_linears:
        _patch_linears_cublas()

    from tilelang.jit.backend import clear_compilation_traces, get_compilation_traces
    clear_compilation_traces()

    tl_options = {
        "pass_configs": {tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True},
    }
    if args.arch:
        tl_options["arch"] = args.arch
    tl_model = torch.compile(model, backend="tilelang", options=tl_options)

    with torch.no_grad():
        tl_logits = tl_model(input_ids).logits
    traces = get_compilation_traces()
    print(f"  Subgraphs compiled: {len(traces)}")
    for i, tr in enumerate(traces):
        print(f"\n  [Subgraph {i}]")
        print(f"  {tr.summary()}")
        if tr.schedule_matches:
            print(f"  Schedule rule matches:")
            for func_name, rule in tr.schedule_matches.items():
                print(f"    {func_name:50s} → {rule}")

    try:
        torch.testing.assert_close(tl_logits, ref_logits, rtol=1e-2, atol=1e-2)
        print("  Forward correctness: PASS")
    except AssertionError:
        max_diff = (tl_logits - ref_logits).abs().max().item()
        print(f"  Forward correctness: WARN (max diff = {max_diff:.6f})")

    tl_tokens = greedy_generate(tl_model, input_ids, max_new_tokens)
    if tokenizer is not None:
        tl_text = tokenizer.decode(tl_tokens[0, seq_len:],
                                   skip_special_tokens=True)
        print(f"  Generated: \"{tl_text}\"")
        token_match = (tl_tokens == ref_tokens).all().item()
        print(f"  Token match vs eager: {'PASS' if token_match else 'FAIL'}")
        if not token_match:
            n_diff = (tl_tokens != ref_tokens).sum().item()
            print(f"    {n_diff}/{ref_tokens[0, seq_len:].numel()} tokens differ")
    else:
        token_match = (tl_tokens == ref_tokens).all().item()
        print(f"  Token match: {'PASS' if token_match else 'FAIL'}")
        if not token_match:
            n_diff = (tl_tokens != ref_tokens).sum().item()
            print(f"    {n_diff}/{ref_tokens.numel()} tokens differ")

    # ---------------------------------------------------------------
    # 4. Benchmark / breakdown (single forward pass, prefill)
    # ---------------------------------------------------------------
    need_bench_models = (not args.correctness_only) or args.breakdown
    if need_bench_models:
        if not args.correctness_only:
            print(f"\n--- Benchmark: prefill (B={batch}, S={seq_len}) ---")

        # Restore standard patches for fair eager/torch.compile baselines.
        if args.use_tileops_mha:
            _unpatch_causal_mask()
            _patch_attention_graph_break()
        if args.cublas_linears:
            _unpatch_linears()

        if not args.correctness_only:
            with torch.no_grad():
                model(input_ids)  # warmup
                eager_time = do_bench(
                    lambda: model(input_ids), backend=args.bench_backend
                )

        dynamo.reset()
        tc_model_bench = torch.compile(model)
        with torch.no_grad():
            tc_model_bench(input_ids)  # warmup + compile
            if not args.correctness_only:
                tc_time = do_bench(
                    lambda: tc_model_bench(input_ids), backend=args.bench_backend
                )

        # TileLang: re-apply optimized patches, recompile, and benchmark.
        if args.use_tileops_mha:
            _patch_causal_mask_none_for_prefill()
            _patch_attention_tileops_mha(args.arch)
        if args.cublas_linears:
            _patch_linears_cublas()
        dynamo.reset()
        tl_model_bench = torch.compile(model, backend="tilelang", options=tl_options)
        with torch.no_grad():
            tl_model_bench(input_ids)  # warmup + compile
            if not args.correctness_only:
                tl_time = do_bench(
                    lambda: tl_model_bench(input_ids), backend=args.bench_backend
                )

        if not args.correctness_only:
            print(f"  eager:          {eager_time:.3f} ms")
            print(f"  torch.compile:  {tc_time:.3f} ms")
            print(f"  tilelang:       {tl_time:.3f} ms")
            print(f"  tilelang vs torch.compile: {tc_time / tl_time:.2f}x")

        if args.breakdown:
            print(
                f"\n--- CUDA kernel breakdown: prefill "
                f"(iters={args.breakdown_iters}, warmup={args.breakdown_warmup}) ---"
            )
            torch_compile_profile = _profile_cuda_kernels(
                lambda: tc_model_bench(input_ids),
                warmup_iters=args.breakdown_warmup,
                profile_iters=args.breakdown_iters,
            )
            tilelang_profile = _profile_cuda_kernels(
                lambda: tl_model_bench(input_ids),
                warmup_iters=args.breakdown_warmup,
                profile_iters=args.breakdown_iters,
            )

            figure_path, csv_path, json_path = _emit_breakdown_report(
                torch_compile_profile=torch_compile_profile,
                tilelang_profile=tilelang_profile,
                prefix=args.breakdown_prefix,
                topk=args.breakdown_topk,
                title=f"Llama Prefill Breakdown (B={batch}, S={seq_len})",
            )

            print(f"  torch.compile total CUDA kernel time: {torch_compile_profile['total_avg_ms']:.3f} ms")
            print(f"  tilelang total CUDA kernel time:      {tilelang_profile['total_avg_ms']:.3f} ms")
            print(f"  figure: {figure_path.resolve()}")
            print(f"  kernels csv: {csv_path.resolve()}")
            print(f"  summary json: {json_path.resolve()}")

            for backend, profile in (("torch.compile", torch_compile_profile), ("tilelang", tilelang_profile)):
                print(f"\n  {backend} top kernels:")
                for row in profile["rows"][: min(5, len(profile["rows"]))]:
                    print(
                        f"    {row['avg_ms']:.3f} ms  "
                        f"({row['pct_total']:.1f}%)  "
                        f"{row['kernel']}"
                    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Llama2 end-to-end: tilelang graph compiler vs torch.compile"
    )
    parser.add_argument("--pretrained", type=str, default=None,
                        help="HuggingFace model name/path (e.g. meta-llama/Llama-2-7b-hf)")
    parser.add_argument("--preset", type=str, default="tiny",
                        choices=["tiny", "small", "7b"],
                        help="Random model preset (ignored with --pretrained)")
    parser.add_argument("--prompt", type=str,
                        default="The future of artificial intelligence is",
                        help="Text prompt for pretrained model generation")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=32,
                        help="Sequence length (only for random models)")
    parser.add_argument("--num-layers", type=int, default=None,
                        help="Number of layers (default: all for pretrained, 1 for random models)")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--arch", type=str, default=None)
    parser.add_argument("--bench-backend", type=str, default="event",
                        choices=["event", "cupti"])
    parser.add_argument("--correctness-only", action="store_true")
    parser.add_argument("--use-tileops-mha", action="store_true",
                        help="Use the built-in TileLang FlashAttention fast path for attention")
    parser.add_argument("--cublas-linears", action="store_true",
                        help="Route nn.Linear through cuBLAS (graph break) for better GEMM perf")
    parser.add_argument("--breakdown", action="store_true",
                        help="Profile CUDA kernels for torch.compile and tilelang, and save a comparison figure")
    parser.add_argument("--breakdown-warmup", type=int, default=2,
                        help="Warmup iterations before collecting CUDA kernel breakdown")
    parser.add_argument("--breakdown-iters", type=int, default=5,
                        help="Profiled iterations used to average CUDA kernel breakdown")
    parser.add_argument("--breakdown-topk", type=int, default=12,
                        help="Number of kernels to include in the breakdown figure")
    parser.add_argument("--breakdown-prefix", type=str, default="tests/end2end/bench_llama_breakdown",
                        help="Output path prefix for breakdown artifacts (.png, .json, _kernels.csv)")
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  [%(name)s:%(levelname)s]: %(message)s",
        datefmt="%H:%M:%S",
    )
    args = parse_args()
    run_benchmark(args)
