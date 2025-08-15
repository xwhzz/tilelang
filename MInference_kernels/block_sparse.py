import torch

import tilelang
import tilelang.language as T
from tilelang.profiler import do_bench

import triton
import triton.language as tl
from tilelang.engine import register_cuda_postproc

tilelang.disable_cache()

@register_cuda_postproc
def _postproc(code, _):
    code = """
#include <math_constants.h>
#include <tl_templates/cuda/gemm.h>
#include <tl_templates/cuda/copy.h>
#include <tl_templates/cuda/reduce.h>
#include <tl_templates/cuda/ldsm.h>
#include <tl_templates/cuda/threadblock_swizzle.h>
#include <tl_templates/cuda/debug.h>
#ifdef ENABLE_BF16
#include <tl_templates/cuda/cuda_bf16_fallbacks.cuh>
#endif

extern "C" __global__ void blocksparse_flashattn_kernel(int* __restrict__ BlockIndex, __grid_constant__ const CUtensorMap K_desc, half_t* __restrict__ Output, half_t* __restrict__ Q, __grid_constant__ const CUtensorMap V_desc);
extern "C" __global__ void __launch_bounds__(256, 1) blocksparse_flashattn_kernel(int* __restrict__ BlockIndex, __grid_constant__ const CUtensorMap K_desc, half_t* __restrict__ Output, half_t* __restrict__ Q, __grid_constant__ const CUtensorMap V_desc) {
  extern __shared__ __align__(1024) uchar buf_dyn_shmem[];
  half_t Q_local[32];
  float acc_o[32];
  float logsum[2];
  float scores_max[2];
  float acc_s[32];
  float scores_max_prev[2];
  float scores_scale[2];
  half_t acc_s_cast[32];
  float scores_sum[2];
  __shared__ uint64_t _mbarrier[10];
  if (tl::tl_shuffle_elect<0>()) {
    tl::prefetch_tma_descriptor(K_desc);
    tl::prefetch_tma_descriptor(V_desc);
    tl::mbarrier_init(_mbarrier[0], 128);
    tl::mbarrier_init(_mbarrier[1], 128);
    tl::mbarrier_init(_mbarrier[2], 128);
    tl::mbarrier_init(_mbarrier[3], 128);
    tl::mbarrier_init(_mbarrier[4], 1);
    tl::mbarrier_init(_mbarrier[5], 1);
    tl::mbarrier_init(_mbarrier[6], 1);
    tl::mbarrier_init(_mbarrier[7], 1);
    tl::mbarrier_init(_mbarrier[8], 128);
    tl::mbarrier_init(_mbarrier[9], 128);
  }
  __syncthreads();
  if (128 <= ((int)threadIdx.x)) {
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
      *(uint4*)(((half_t*)buf_dyn_shmem) + (((i * 1024) + (((int)threadIdx.x) * 8)) + 15360)) = *(uint4*)(Q + ((((((int)blockIdx.x) * 4096) + (i * 1024)) + (((int)threadIdx.x) * 8)) - 1024));
    }
    tl::fence_proxy_async();
    tl::mbarrier_cp_async_arrive(_mbarrier[8]);
    tl::mbarrier_arrive(_mbarrier[8]);
    if (((int)threadIdx.x) < 153) {
      *(int2*)(((int*)buf_dyn_shmem) + ((((int)threadIdx.x) * 2) + 9984)) = *(int2*)(BlockIndex + (((((int)blockIdx.x) * 50) + (((int)threadIdx.x) * 2)) - 256));
    }
    tl::fence_proxy_async();
    tl::mbarrier_cp_async_arrive(_mbarrier[9]);
    tl::mbarrier_arrive(_mbarrier[9]);
    for (int bi = 0; bi < min((((int)blockIdx.x) + 1), 50); ++bi) {
      tl::mbarrier_wait(_mbarrier[((bi & 1) + 4)], (((bi & 3) >> 1) ^ 1));
      cutlass::arch::NamedBarrier::sync(128, 0);
      if (tl::tl_shuffle_elect<128>()) {
        tl::mbarrier_expect_tx(_mbarrier[(bi & 1)], 8192);
        tl::tma_load<tl::CacheHintSm90::EVICT_NORMAL>(K_desc, _mbarrier[(bi & 1)], (&(((half_t*)buf_dyn_shmem)[((bi & 1) * 4096)])), 0, (((int*)buf_dyn_shmem)[(bi + 10240)] * 64), 0, 0);
      }
      tl::mbarrier_cp_async_arrive(_mbarrier[(bi & 1)]);
      tl::mbarrier_arrive(_mbarrier[(bi & 1)]);
      tl::mbarrier_wait(_mbarrier[((bi & 1) + 6)], (((bi & 3) >> 1) ^ 1));
      cutlass::arch::NamedBarrier::sync(128, 0);
      if (tl::tl_shuffle_elect<128>()) {
        tl::mbarrier_expect_tx(_mbarrier[((bi & 1) + 2)], 8192);
        tl::tma_load<tl::CacheHintSm90::EVICT_NORMAL>(V_desc, _mbarrier[((bi & 1) + 2)], (&(((half_t*)buf_dyn_shmem)[(((bi & 1) * 4096) + 8192)])), 0, (((int*)buf_dyn_shmem)[(bi + 10240)] * 64), 0, 0);
      }
      tl::mbarrier_cp_async_arrive(_mbarrier[((bi & 1) + 2)]);
      tl::mbarrier_arrive(_mbarrier[((bi & 1) + 2)]);
    }
  } else {
    tl::mbarrier_wait(_mbarrier[8], 0);
    #pragma unroll
    for (int i_1 = 0; i_1 < 16; ++i_1) {
      uint1 __1;
      float2 __2;
        float2 __3;
        uint1 v_ = *(uint1*)(((half_t*)buf_dyn_shmem) + (((((((((int)threadIdx.x) >> 5) * 1024) + ((i_1 & 1) * 512)) + (((((int)threadIdx.x) & 31) >> 2) * 64)) + ((i_1 >> 1) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + 16384));
        __3.x = (float)(((half2*)(&(v_.x)))->x);
        __3.y = (float)(((half2*)(&(v_.x)))->y);
        float2 v__1 = make_float2(1.803369e-01f, 1.803369e-01f);
        __2.x = (__3.x*v__1.x);
        __2.y = (__3.y*v__1.y);
      ((half2*)(&(__1.x)))->x = (half_t)(__2.x);
      ((half2*)(&(__1.x)))->y = (half_t)(__2.y);
      *(uint1*)(Q_local + (i_1 * 2)) = __1;
    }
    #pragma unroll
    for (int i_2 = 0; i_2 < 16; ++i_2) {
      *(float2*)(acc_o + (i_2 * 2)) = make_float2(0.000000e+00f, 0.000000e+00f);
    }
    #pragma unroll
    for (int i_3 = 0; i_3 < 2; ++i_3) {
      logsum[i_3] = 0.000000e+00f;
    }
    #pragma unroll
    for (int i_4 = 0; i_4 < 2; ++i_4) {
      scores_max[i_4] = -CUDART_INF_F;
    }
    tl::fence_proxy_async();
    tl::mbarrier_wait(_mbarrier[9], 0);
    for (int bi_1 = 0; bi_1 < min((((int)blockIdx.x) + 1), 50); ++bi_1) {
      #pragma unroll
      for (int i_5 = 0; i_5 < 16; ++i_5) {
        for (int vec_s = 0; vec_s < 2; ++vec_s) {
          float condval;
          if ((((((((int*)buf_dyn_shmem)[(bi_1 + 10240)] * 64) + ((i_5 >> 1) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + vec_s) <= ((((((int)blockIdx.x) * 64) + ((((int)threadIdx.x) >> 5) * 16)) + ((i_5 & 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2)))) {
            condval = 0.000000e+00f;
          } else {
            condval = -CUDART_INF_F;
          }
          acc_s[((i_5 * 2) + vec_s)] = condval;
        }
      }
      tl::fence_proxy_async();
      tl::mbarrier_wait(_mbarrier[(bi_1 & 1)], ((bi_1 & 3) >> 1));
      if (((int*)buf_dyn_shmem)[(bi_1 + 10240)] < 256) {
        tl::gemm_rs<64, 64, 64, 4, 1, 0, 1, 0, 64, 64, 0, 0, true>((&(Q_local[0])), (&(((half_t*)buf_dyn_shmem)[((bi_1 & 1) * 4096)])), (&(acc_s[0])));
      }
      tl::mbarrier_arrive(_mbarrier[((bi_1 & 1) + 4)], 0, (((int)threadIdx.x) == 0));
      #pragma unroll
      for (int i_6 = 0; i_6 < 2; ++i_6) {
        scores_max_prev[i_6] = scores_max[i_6];
      }
      #pragma unroll
      for (int i_7 = 0; i_7 < 2; ++i_7) {
        #pragma unroll
        for (int rv = 0; rv < 16; ++rv) {
          scores_max[i_7] = max(scores_max[i_7], acc_s[((((rv & 7) * 4) + (i_7 * 2)) + (rv >> 3))]);
        }
        scores_max[i_7] = tl::AllReduce<tl::MaxOp, 4, 1, 0, 128>::run_hopper(scores_max[i_7]);
      }
      #pragma unroll
      for (int i_8 = 0; i_8 < 2; ++i_8) {
        scores_scale[i_8] = exp2f((scores_max_prev[i_8] - scores_max[i_8]));
      }
      #pragma unroll
      for (int i_9 = 0; i_9 < 32; ++i_9) {
        acc_s[i_9] = exp2f((acc_s[i_9] - scores_max[((i_9 & 3) >> 1)]));
      }
      #pragma unroll
      for (int i_10 = 0; i_10 < 32; ++i_10) {
        acc_o[i_10] = (acc_o[i_10] * scores_scale[((i_10 & 3) >> 1)]);
      }
      #pragma unroll
      for (int i_11 = 0; i_11 < 16; ++i_11) {
        uint1 __4;
        float2 v__2 = *(float2*)(acc_s + (i_11 * 2));
        ((half2*)(&(__4.x)))->x = (half_t)(v__2.x);
        ((half2*)(&(__4.x)))->y = (half_t)(v__2.y);
        *(uint1*)(acc_s_cast + (i_11 * 2)) = __4;
      }
      tl::fence_proxy_async();
      tl::mbarrier_wait(_mbarrier[((bi_1 & 1) + 2)], ((bi_1 & 3) >> 1));
      if (((int*)buf_dyn_shmem)[(bi_1 + 10240)] < 256) {
        tl::gemm_rs<64, 64, 64, 4, 1, 0, 0, 0, 64, 64, 0, 0, true>((&(acc_s_cast[0])), (&(((half_t*)buf_dyn_shmem)[(((bi_1 & 1) * 4096) + 8192)])), (&(acc_o[0])));
      }
      tl::mbarrier_arrive(_mbarrier[((bi_1 & 1) + 6)], 0, (((int)threadIdx.x) == 0));
      #pragma unroll
      for (int i_12 = 0; i_12 < 2; ++i_12) {
        scores_sum[i_12] = 0.000000e+00f;
        #pragma unroll
        for (int rv_1 = 0; rv_1 < 16; ++rv_1) {
          scores_sum[i_12] = (scores_sum[i_12] + acc_s[((((rv_1 & 7) * 4) + (i_12 * 2)) + (rv_1 >> 3))]);
        }
        scores_sum[i_12] = tl::AllReduce<tl::SumOp, 4, 1, 0, 128>::run_hopper(scores_sum[i_12]);
      }
      #pragma unroll
      for (int i_13 = 0; i_13 < 2; ++i_13) {
        logsum[i_13] = ((logsum[i_13] * scores_scale[i_13]) + scores_sum[i_13]);
      }
    }
    #pragma unroll
    for (int i_14 = 0; i_14 < 32; ++i_14) {
      acc_o[i_14] = (acc_o[i_14] / logsum[((i_14 & 3) >> 1)]);
    }
    #pragma unroll
    for (int i_15 = 0; i_15 < 16; ++i_15) {
      uint1 __5;
      float2 v__3 = *(float2*)(acc_o + (i_15 * 2));
      ((half2*)(&(__5.x)))->x = (half_t)(v__3.x);
      ((half2*)(&(__5.x)))->y = (half_t)(v__3.y);
      *(uint1*)(Output + ((((((((int)blockIdx.x) * 4096) + ((((int)threadIdx.x) >> 5) * 1024)) + ((i_15 & 1) * 512)) + (((((int)threadIdx.x) & 31) >> 2) * 64)) + ((i_15 >> 1) * 8)) + ((((int)threadIdx.x) & 3) * 2))) = __5;
    }
  }
}  
    """
    return code


@tilelang.jit(out_idx=[4],)
def _tl_blocksparse_flashattn(batch, heads, seq_len, dim, is_causal, top_k=10, dtype="float16"):
    block_M = 64
    block_N = 64
    num_stages = 2
    threads = 128
    scale = (1.0 / dim)**0.5 * 1.44269504  # log2(e)
    shape = [batch, heads, seq_len, dim]
    top_k = min(top_k, seq_len // block_N)
    block_index_shape = [batch, heads, seq_len // block_M, top_k]

    accum_dtype = "float"
    block_index_dtype = "int32"

    def kernel_func(block_M, block_N, num_stages, threads):

        @T.prim_func
        def blocksparse_flashattn(
                Q: T.Tensor(shape, dtype),
                K: T.Tensor(shape, dtype),
                V: T.Tensor(shape, dtype),
                BlockIndex: T.Tensor(block_index_shape, block_index_dtype),
                Output: T.Tensor(shape, dtype),
        ):
            with T.Kernel(
                    T.ceildiv(seq_len, block_M), heads, batch, threads=threads) as (bx, by, bz):
                Q_shared = T.alloc_shared([block_M, dim], dtype)
                Q_local = T.alloc_fragment([block_M, dim], dtype)
                K_shared = T.alloc_shared([block_N, dim], dtype)
                V_shared = T.alloc_shared([block_N, dim], dtype)
                # O_shared = T.alloc_shared([block_M, dim], dtype)
                acc_s = T.alloc_fragment([block_M, block_N], accum_dtype)
                acc_s_cast = T.alloc_fragment([block_M, block_N], dtype)
                acc_o = T.alloc_fragment([block_M, dim], accum_dtype)
                scores_max = T.alloc_fragment([block_M], accum_dtype)
                scores_max_prev = T.alloc_fragment([block_M], accum_dtype)
                scores_scale = T.alloc_fragment([block_M], accum_dtype)
                scores_sum = T.alloc_fragment([block_M], accum_dtype)
                logsum = T.alloc_fragment([block_M], accum_dtype)
                block_index = T.alloc_shared([top_k], block_index_dtype)

                T.copy(Q[bz, by, bx * block_M:(bx + 1) * block_M, :], Q_shared)
                for i, j in T.Parallel(block_M, dim):
                    Q_local[i, j] = Q_shared[i, j] * scale

                T.fill(acc_o, 0)
                T.fill(logsum, 0)
                T.fill(scores_max, -T.infinity(accum_dtype))

                for vj in T.Parallel(top_k):
                    block_index[vj] = BlockIndex[bz, by, bx, vj]

                block_count = T.min(T.ceildiv((bx + 1) * block_M, block_N), top_k)

                for bi in T.Pipelined(block_count, num_stages=num_stages):
                    k = block_index[bi]
                    # if k < 256:
                    T.copy(K[bz, by, k * block_N:(k + 1) * block_N, :], K_shared) # load K
                    T.copy(V[bz, by, k * block_N:(k + 1) * block_N, :], V_shared) # load V
                    for i, j in T.Parallel(block_M, block_N):
                        acc_s[i, j] = T.if_then_else(bx * block_M + i >= k * block_N + j, 0,
                                                 -T.infinity(acc_s.dtype))
                    T.gemm(Q_local, K_shared, acc_s, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)

                    T.copy(scores_max, scores_max_prev)
                    T.reduce_max(acc_s, scores_max, dim=1, clear=False)

                    for i in T.Parallel(block_M):
                        scores_scale[i] = T.exp2(scores_max_prev[i] - scores_max[i])
                    for i, j in T.Parallel(block_M, block_N):
                        acc_s[i, j] = T.exp2(acc_s[i, j] - scores_max[i])

                    for i, j in T.Parallel(block_M, dim):
                        acc_o[i, j] = acc_o[i, j] * scores_scale[i]

                    T.copy(acc_s, acc_s_cast)
                    T.gemm(acc_s_cast, V_shared, acc_o, policy=T.GemmWarpPolicy.FullRow)

                    T.reduce_sum(acc_s, scores_sum, dim=1)

                    for i in T.Parallel(block_M):
                        logsum[i] = logsum[i] * scores_scale[i] + scores_sum[i]

                for i, j in T.Parallel(block_M, dim):
                    acc_o[i, j] /= logsum[i]
                T.copy(acc_o, Output[bz, by, bx * block_M:(bx + 1) * block_M, :])

        return blocksparse_flashattn

    return kernel_func(block_M, block_N, num_stages, threads)

@triton.jit
def _triton_block_sparse_attn_fwd_kernel(
    Q, K, V, seqlens, sm_scale,
    block_index,
    Out,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vn, stride_vk,
    stride_oz, stride_oh, stride_om, stride_ok,
    Z, H, N_CTX,
    NUM_ROWS, MAX_BLOCKS_PRE_ROW,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    dtype: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)

    seqlen = tl.load(seqlens + off_hz // H)
    if start_m * BLOCK_M >= seqlen:
        return

    # initialize offsets
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)

    qo_offset = (off_hz // H) * stride_qz + (off_hz % H) * stride_qh
    kv_offset = (off_hz // H) * stride_kz + (off_hz % H) * stride_kh

    q_ptrs = Q + qo_offset + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk
    k_ptrs = K + kv_offset + offs_d[:, None] * stride_kk
    v_ptrs = V + kv_offset + offs_d[None, :] * stride_vk
    o_ptrs = Out + qo_offset + offs_m[:, None] * stride_om + offs_d[None, :] * stride_ok

    blocks_ptr = block_index + (off_hz * NUM_ROWS + start_m) * MAX_BLOCKS_PRE_ROW

    # initialize pointer to m and l
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    # scale sm_scale by log_2(e) and use
    # 2^x instead of exp in the loop because CSE and LICM
    # don't work as expected with `exp` in the loop
    qk_scale = sm_scale * 1.44269504
    # load q: it will stay in SRAM throughout
    q = tl.load(q_ptrs)
    q = (q * qk_scale).to(dtype)

    # loop over k, v and update accumulator
    m_mask = offs_m[:, None] < seqlen
    block_count = tl.minimum((start_m + 1) * BLOCK_M // BLOCK_N, MAX_BLOCKS_PRE_ROW)


    # (1 + 2 + 3 + ... )

    for sparse_block_idx in range(block_count):
        real_block_idx = tl.load(blocks_ptr + sparse_block_idx)
        start_n = real_block_idx * BLOCK_N
        cols = start_n + offs_n
        # -- load k, v --
        k = tl.load(k_ptrs + cols[None, :] * stride_kn)
        v = tl.load(v_ptrs + cols[:, None] * stride_vn)
        # -- compute qk --
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        # if start_n + BLOCK_N < seqlen:
        #     qk = tl.where(m_mask, qk, float("-inf"))
        # else:
        causal_mask = cols[None, :] <= offs_m[:, None]
        qk = tl.where(m_mask & causal_mask, qk, float("-inf"))
        qk += tl.dot(q, k)
        # -- compute scaling constant --
        m_i_new = tl.maximum(m_i, tl.max(qk, 1))
        alpha = tl.math.exp2(m_i - m_i_new)
        p = tl.math.exp2(qk - m_i_new[:, None])
        # -- scale and update acc --
        acc_scale = l_i * 0 + alpha  # workaround some compiler bug
        acc *= acc_scale[:, None]
        acc += tl.dot(p.to(dtype), v)
        # -- update m_i and l_i --
        l_i = l_i * alpha + tl.sum(p, 1)
        m_i = m_i_new

    # write back O
    acc /= l_i[:, None]
    tl.store(o_ptrs, acc.to(dtype), mask=m_mask)


def _triton_block_sparse_attention(
    q,                 # [BATCH, N_HEADS, N_CTX, D_HEAD]
    k,                 # [BATCH, N_HEADS, N_CTX, D_HEAD]
    v,                 # [BATCH, N_HEADS, N_CTX, D_HEAD]
    seqlens,           # [BATCH, ]
    block_index,       # [BATCH, N_HEADS, cdiv(N_CTX, BLOCK_SIZE_M), MAX_BLOCKS_PRE_ROW]
    sm_scale,
    block_size_M=64,
    block_size_N=64,
) -> torch.Tensor:
    # shape constraints
    Lq, Lk, Lv = q.shape[-1], k.shape[-1], v.shape[-1]
    assert Lq == Lk and Lk == Lv
    assert Lk in {16, 32, 64, 128}
    o = torch.zeros_like(q)
    grid = (triton.cdiv(q.shape[2], block_size_M), q.shape[0] * q.shape[1], 1)
    dtype = tl.bfloat16 if q.dtype == torch.bfloat16 else tl.float16
    _triton_block_sparse_attn_fwd_kernel[grid](
        q, k, v, seqlens, sm_scale,
        block_index,
        o,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        o.stride(0), o.stride(1), o.stride(2), o.stride(3),
        q.shape[0], q.shape[1], q.shape[2],
        block_index.shape[-2], block_index.shape[-1],
        BLOCK_M=block_size_M, BLOCK_N=block_size_N,
        BLOCK_DMODEL=Lk,
        dtype=dtype,
        num_warps=4, num_stages=2,
    )

    return o


def _build_block_index(
    query: torch.Tensor,     # [BATCH, N_HEADS, N_CTX, D_HEAD]
    key: torch.Tensor,       # [BATCH, N_HEADS, N_CTX, D_HEAD]
    top_k: int,
    block_size_M: int = 64,
    block_size_N: int = 64,
):
    batch_size, num_heads, context_size, head_dim = query.shape
    query_pool = query.reshape((batch_size, num_heads, -1, block_size_M, head_dim)).mean(dim=-2)
    key_pool = key.reshape((batch_size, num_heads, -1, block_size_N, head_dim)).mean(dim=-2)
    arange_M = torch.arange(query_pool.shape[-2], dtype=torch.int32, device=query.device) * block_size_M
    arange_N = torch.arange(key_pool.shape[-2], dtype=torch.int32, device=key.device) * block_size_N
    p_pool = torch.einsum(f'bhmk, bhnk -> bhmn', query_pool, key_pool)
    p_pool = p_pool.where(arange_M[None, None, :, None] >= arange_N[None, None, None, :], -torch.inf)
    top_k = min(top_k, context_size // block_size_N)
    return torch.topk(p_pool, top_k, dim=-1).indices.to(torch.int32).sort(dim=-1).values


def block_sparse_attention(
    query: torch.Tensor,  # [BATCH, N_HEADS, N_CTX, D_HEAD]
    key: torch.Tensor,    # [BATCH, N_HEADS, N_CTX, D_HEAD]
    value: torch.Tensor,  # [BATCH, N_HEADS, N_CTX, D_HEAD]
    top_k: int,
    block_size_M: int = 64,
    block_size_N: int = 64,
):
    batch_size, num_heads, context_size, head_dim = query.shape
    pad = block_size_M - (query.shape[2] & (block_size_M - 1))
    pad = 0 if pad == block_size_M else pad
    query = torch.nn.functional.pad(query, [0, 0, 0, pad, 0, 0, 0, 0])
    key = torch.nn.functional.pad(key, [0, 0, 0, pad, 0, 0, 0, 0])
    value = torch.nn.functional.pad(value, [0, 0, 0, pad, 0, 0, 0, 0])
    seqlens = torch.tensor([context_size], dtype=torch.int32, device=query.device)
    sm_scale = head_dim ** -0.5
    block_index = _build_block_index(query, key, top_k, block_size_N, block_size_N)
    tl_kernel = _tl_blocksparse_flashattn(batch_size, num_heads, context_size, head_dim, True, top_k)

    def run(is_triton: bool = True):
        if is_triton:
            out = _triton_block_sparse_attention(query, key, value, seqlens, block_index, sm_scale, block_size_M, block_size_N)
        else:
            out = tl_kernel(query, key, value, block_index)
        return out[..., :context_size, :]
    return run

def test_correctness():
    BATCH, N_HEADS, SEQ_LEN, D_HEAD = 1, 1, 16384, 64
    TOK_LIST = [50, 20, 10]
    TOPK = TOK_LIST[0]

    # torch.manual_seed(0)
    q = torch.randn(BATCH, N_HEADS, SEQ_LEN, D_HEAD, device='cuda', dtype=torch.float16)
    k = torch.randn(BATCH, N_HEADS, SEQ_LEN, D_HEAD, device='cuda', dtype=torch.float16)
    v = torch.randn(BATCH, N_HEADS, SEQ_LEN, D_HEAD, device='cuda', dtype=torch.float16)

    run = block_sparse_attention(q, k, v, TOPK)

    triton_out = run(True)
    tilelang_out = run(False)
    torch.cuda.synchronize()
    torch.testing.assert_close(tilelang_out, triton_out, atol=1e-2, rtol=1e-2)
    print("Pass topk sparse attention test with qlen == klen")

    triton_time = do_bench(lambda: run(True))
    tilelang_time = do_bench(lambda: run(False))

    print("triton_time: ", triton_time)
    print("tilelang_time: ", tilelang_time)

test_correctness()