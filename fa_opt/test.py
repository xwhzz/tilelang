# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.

import torch
import torch.nn.functional as F
import tilelang
from tilelang.autotuner import *
import tilelang.language as T
import itertools
import argparse
from functools import partial
from tilelang.engine.callback import register_cuda_postproc_callback
tilelang.disable_cache()

def get_configs():
    block_M = [128]
    block_N = [128]
    num_stages = [2]
    threads = [256]
    _configs = list(itertools.product(block_M, block_N, num_stages, threads))

    configs = [{
        'block_M': c[0],
        'block_N': c[1],
        'num_stages': c[2],
        'threads': c[3]
    } for c in _configs]
    return configs

@register_cuda_postproc_callback
def tilelang_callback_cuda_postproc(code, _):
    # return code
    print(code)
    code = """
#include <math_constants.h>
#include <tl_templates/cuda/gemm.h>
#include <tl_templates/cuda/copy.h>
#include <tl_templates/cuda/reduce.h>
#include <tl_templates/cuda/ldsm.h>
#include <tl_templates/cuda/threadblock_swizzle.h>
#include <tl_templates/cuda/debug.h>

extern "C" __global__ void main_kernel(__grid_constant__ const CUtensorMap K_desc, __grid_constant__ const CUtensorMap Output_desc, __grid_constant__ const CUtensorMap Q_desc, __grid_constant__ const CUtensorMap V_desc);
extern "C" __global__ void __launch_bounds__(384, 1) main_kernel(__grid_constant__ const CUtensorMap K_desc, __grid_constant__ const CUtensorMap Output_desc, __grid_constant__ const CUtensorMap Q_desc, __grid_constant__ const CUtensorMap V_desc) {
  extern __shared__ __align__(1024) uchar buf_dyn_shmem[];
  float acc_o[64];
  float logsum[2];
  float scores_max[2];
  float acc_s[88];
  float scores_max_prev[2];
  float scores_scale[2];
  //float scores_sum[2];
  cutlass::bfloat16_t acc_s_cast[88];
  __shared__ uint64_t k_load_barrier[2];    // For K tensor loading (alternating)
  __shared__ uint64_t v_load_barrier[2];    // For V tensor loading (alternating)
  __shared__ uint64_t k_full_barrier[2];    // For K producer-consumer sync (alternating)
  __shared__ uint64_t v_full_barrier[2];    // For V producer-consumer sync (alternating)
  __shared__ uint64_t q_load_barrier;       // For Q tensor loading
  // if (((int)threadIdx.x) == 0) {

  if (cutlass::canonical_warp_idx_sync() == 0 && cute::elect_one_sync()) {
    tl::prefetch_tma_descriptor(Q_desc);
    tl::prefetch_tma_descriptor(K_desc);
    tl::prefetch_tma_descriptor(V_desc);
    tl::prefetch_tma_descriptor(Output_desc);
    tl::mbarrier_init(k_load_barrier[0], 1);
    tl::mbarrier_init(k_load_barrier[1], 1);
    tl::mbarrier_init(v_load_barrier[0], 1);
    tl::mbarrier_init(v_load_barrier[1], 1);
    tl::mbarrier_init(k_full_barrier[0], 2);
    tl::mbarrier_init(k_full_barrier[1], 2);
    tl::mbarrier_init(v_full_barrier[0], 2);
    tl::mbarrier_init(v_full_barrier[1], 2);
    tl::mbarrier_init(q_load_barrier, 1);
    //tl::mbarrier_init(_mbarrier[9], 256);
    //tl::mbarrier_init(_mbarrier[10], 256);
  }
  __syncthreads();
  if (256 <= ((int)threadIdx.x)) {
    tl::warpgroup_reg_dealloc<24>();
    //int warp_idx_in_warpgroup = __shfl_sync(0xffffffff, (threadIdx.x / 32) % 4, 0);
    //if (warp_idx_in_warpgroup == 0) {
    //  cutlass::arch::NamedBarrier::sync(160, static_cast<uint32_t>(2) /*id*/);
    //}
    // tl::mbarrier_wait(_mbarrier[4], 1);
      // if (((int)threadIdx.x) == 256) {
      if (__shfl_sync(0xffffffff, (threadIdx.x / 32) % 4, 0) == 0 && cute::elect_one_sync()) {
        tl::mbarrier_wait(k_full_barrier[0], 1);
        // tl::mbarrier_expect_tx(k_load_barrier[0], 45056);
        if (threadIdx.x % 128 == 0){
         tl::mbarrier_arrive_expect_tx(k_load_barrier[0], 45056);}

        tl::tma_load<tl::CacheHintSm90::EVICT_LAST>(K_desc, k_load_barrier[0], (&(((cutlass::bfloat16_t*)buf_dyn_shmem)[(16384)])), 0, ((int)blockIdx.y), 0, 0);
        tl::tma_load<tl::CacheHintSm90::EVICT_LAST>(K_desc, k_load_barrier[0], (&(((cutlass::bfloat16_t*)buf_dyn_shmem)[(27648)])), 64, ((int)blockIdx.y), 0, 0);
      }
      // tl::mbarrier_arrive(_mbarrier[0]);

    // if (((int)threadIdx.x) == 256) {
    if (__shfl_sync(0xffffffff, (threadIdx.x / 32) % 4, 0) == 0 && cute::elect_one_sync()) {
      // tl::mbarrier_expect_tx(q_load_barrier, 32768);
      tl::mbarrier_arrive_expect_tx(q_load_barrier, 32768);
      tl::tma_load<tl::CacheHintSm90::EVICT_FIRST>(Q_desc, q_load_barrier, (&(((cutlass::bfloat16_t*)buf_dyn_shmem)[0])), 0, ((int)blockIdx.y), (((int)blockIdx.x) * 128), 0);
      tl::tma_load<tl::CacheHintSm90::EVICT_FIRST>(Q_desc, q_load_barrier, (&(((cutlass::bfloat16_t*)buf_dyn_shmem)[8192])), 64, ((int)blockIdx.y), (((int)blockIdx.x) * 128), 0);
    }
    // tl::mbarrier_arrive(_mbarrier[8]);
    #pragma unroll 2
    for (int k = 0; k < 93; ++k) {
      int k_1 = k + 1;

      // tl::mbarrier_wait(_mbarrier[((k_1 & 1) + 4)], (((k_1 & 3) >> 1) ^ 1));
      // if (((int)threadIdx.x) == 256) {
      if (__shfl_sync(0xffffffff, (threadIdx.x / 32) % 4, 0) == 0 && cute::elect_one_sync()) {
        tl::mbarrier_wait(k_full_barrier[((k_1 & 1))], (((k_1 & 3) >> 1) ^ 1));
        // tl::mbarrier_expect_tx(k_load_barrier[(k_1 & 1)], 45056);
        if (threadIdx.x % 128 == 0){
         tl::mbarrier_arrive_expect_tx(k_load_barrier[(k_1 & 1)], 45056);}

        tl::tma_load<tl::CacheHintSm90::EVICT_LAST>(K_desc, k_load_barrier[(k_1 & 1)], (&(((cutlass::bfloat16_t*)buf_dyn_shmem)[(((k_1 & 1) * 22528) + 16384)])), 0, ((int)blockIdx.y), (k_1 * 176), 0);
        tl::tma_load<tl::CacheHintSm90::EVICT_LAST>(K_desc, k_load_barrier[(k_1 & 1)], (&(((cutlass::bfloat16_t*)buf_dyn_shmem)[(((k_1 & 1) * 22528) + 27648)])), 64, ((int)blockIdx.y), (k_1 * 176), 0);
      }
      // tl::mbarrier_arrive(_mbarrier[(k_1 & 1)]);
      
      // tl::mbarrier_wait(_mbarrier[((k & 1) + 6)], (((k & 3) >> 1) ^ 1));
      // if (((int)threadIdx.x) == 256) {
      if (__shfl_sync(0xffffffff, (threadIdx.x / 32) % 4, 0) == 0 && cute::elect_one_sync()) {
        tl::mbarrier_wait(v_full_barrier[((k & 1))], (((k & 3) >> 1) ^ 1));
        // tl::mbarrier_expect_tx(v_load_barrier[((k & 1))], 45056);
        if (threadIdx.x % 128 == 0){
         tl::mbarrier_arrive_expect_tx(v_load_barrier[((k & 1))], 45056);}

        tl::tma_load<tl::CacheHintSm90::EVICT_LAST>(V_desc, v_load_barrier[((k & 1))], (&(((cutlass::bfloat16_t*)buf_dyn_shmem)[(((k & 1) * 22528) + 61440)])), 0, ((int)blockIdx.y), (k * 176), 0);
        tl::tma_load<tl::CacheHintSm90::EVICT_LAST>(V_desc, v_load_barrier[((k & 1))], (&(((cutlass::bfloat16_t*)buf_dyn_shmem)[(((k & 1) * 22528) + 72704)])), 64, ((int)blockIdx.y), (k * 176), 0);
      }
      // tl::mbarrier_arrive(_mbarrier[((k & 1) + 2)]);
    }

    //  tl::mbarrier_wait(_mbarrier[7], 1);
    // if (((int)threadIdx.x) == 256) {
    if (__shfl_sync(0xffffffff, (threadIdx.x / 32) % 4, 0) == 0 && cute::elect_one_sync()) {
        tl::mbarrier_wait(v_full_barrier[1], 1);
        // tl::mbarrier_expect_tx(v_load_barrier[1], 45056);
        if (threadIdx.x % 128 == 0){
         tl::mbarrier_arrive_expect_tx(v_load_barrier[1], 45056);}

        tl::tma_load<tl::CacheHintSm90::EVICT_LAST>(V_desc, v_load_barrier[1], (&(((cutlass::bfloat16_t*)buf_dyn_shmem)[((22528) + 61440)])), 0, ((int)blockIdx.y), (93 * 176), 0);
        tl::tma_load<tl::CacheHintSm90::EVICT_LAST>(V_desc, v_load_barrier[1], (&(((cutlass::bfloat16_t*)buf_dyn_shmem)[((22528) + 72704)])), 64, ((int)blockIdx.y), (93 * 176), 0);
      }
      // tl::mbarrier_arrive(_mbarrier[3]);
  } else {
    tl::warpgroup_reg_alloc<240>();
    // int warp_group_idx = threadIdx.x / 128;
    // WG1 needs the very first signal to start
    // if (warp_group_idx == 0) {
    //     //cutlass::arch::NamedBarrier::arrive(160, static_cast<uint32_t>(2) /*id*/);
    //     cutlass::arch::NamedBarrier::arrive(256, static_cast<uint32_t>(0) /*id*/);
    // }

    #pragma unroll
    for (int i = 0; i < 32; ++i) {
      *(float2*)(acc_o + (i * 2)) = make_float2(0.000000e+00f, 0.000000e+00f);
    }
    #pragma unroll
    for (int i_1 = 0; i_1 < 2; ++i_1) {
     logsum[i_1] = 0.000000e+00f;
    }

    //tl::fence_proxy_async();
    tl::mbarrier_wait(q_load_barrier, 0);
    #pragma unroll
    for (int i_3 = 0; i_3 < 44; ++i_3) {
      *(float2*)(acc_s + (i_3 * 2)) = make_float2(0.000000e+00f, 0.000000e+00f);
    }
    //tl::fence_proxy_async();
    if (tl::mbarrier_try_wait(k_load_barrier[0],0) == 0){ tl::mbarrier_wait(k_load_barrier[0], 0); }
    // tl::mbarrier_wait(k_load_barrier[0], 0);
    tl::gemm_ss<128, 176, 128, 8, 1, 0, 1, 0, true>((&(((cutlass::bfloat16_t*)buf_dyn_shmem)[0])), (&(((cutlass::bfloat16_t*)buf_dyn_shmem)[16384])), (&(acc_s[0])));
    // tl::mbarrier_arrive(k_full_barrier[0]);
    tl::mbarrier_arrive(k_full_barrier[0], 0, uint32_t(threadIdx.x % 128 == 0));

    #pragma unroll
    for (int i_5 = 0; i_5 < 2; ++i_5) {
      scores_max[i_5] = -CUDART_INF_F;
    }
    #pragma unroll
    for (int i_6 = 0; i_6 < 2; ++i_6) {
      #pragma unroll
      for (int rv = 0; rv < 44; ++rv) {
        scores_max[i_6] = max(scores_max[i_6], acc_s[((((rv % 22) * 4) + (i_6 * 2)) + (rv / 22))]);
      }
      scores_max[i_6] = tl::AllReduce<tl::MaxOp, 4, 1>::run_hopper(scores_max[i_6]);
    }

    #pragma unroll
    for (int i_8 = 0; i_8 < 88; ++i_8) {
      acc_s[i_8] = exp2f(((acc_s[i_8] * 1.275174e-01f) - (scores_max[((i_8 & 3) >> 1)] * 1.275174e-01f)));
    }
    #pragma unroll
    for (int i_9 = 0; i_9 < 2; ++i_9) {
      //scores_sum[i_9] = 0.000000e+00f;
      #pragma unroll
      for (int rv_1 = 0; rv_1 < 44; ++rv_1) {
        logsum[i_9] = (logsum[i_9] + acc_s[((((rv_1 % 22) * 4) + (i_9 * 2)) + (rv_1 / 22))]);
      }
      //scores_sum[i_9] = tl::AllReduce<tl::SumOp, 4, 1>::run_hopper(scores_sum[i_9]);
           
    }

    #pragma unroll
    for (int i_11 = 0; i_11 < 44; ++i_11) {
      //uint1 __1;
      //float2 v_ = *(float2*)(acc_s + (i_11 * 2));
      //((nv_bfloat162*)(&(__1.x)))->x = (cutlass::bfloat16_t)(v_.x);
      //((nv_bfloat162*)(&(__1.x)))->y = (cutlass::bfloat16_t)(v_.y);
      //*(uint1*)(acc_s_cast + (i_11 * 2)) = __1;

      uint1 __1;
      float2 v_ = *(float2*)(acc_s + (i_11 * 2));
      reinterpret_cast<__nv_bfloat162 &>(__1) = __float22bfloat162_rn(reinterpret_cast<float2 const &>(v_));
      *(uint1*)(acc_s_cast + (i_11 * 2)) = __1; 
    }
    #pragma unroll 1
    for (int k_1 = 0; k_1 < 93; ++k_1) {  
      #pragma unroll
      for (int i_12 = 0; i_12 < 44; ++i_12) {
        *(float2*)(acc_s + (i_12 * 2)) = make_float2(0.000000e+00f, 0.000000e+00f);
      }
      //tl::fence_proxy_async();
      if (tl::mbarrier_try_wait(k_load_barrier[((k_1 + 1) & 1)], (((k_1 + 1) & 3) >> 1)) == 0) {
        tl::mbarrier_wait(k_load_barrier[((k_1 + 1) & 1)], (((k_1 + 1) & 3) >> 1));
      }
      // tl::mbarrier_wait(k_load_barrier[((k_1 + 1) & 1)], (((k_1 + 1) & 3) >> 1));  
      // int k_barrier_idx = ((k_1 + 1) & 1);
      // int k_barrier_phase = ((k_1 + 1) & 3) >> 1;
      // tl::mbarrier_wait(k_load_barrier[k_barrier_idx], k_barrier_phase);
      //warp_scheduler_barrier_sync();
      // cutlass::arch::NamedBarrier::sync(256, warp_group_idx/*id*/);
      tl::gemm_ss<128, 176, 128, 8, 1, 0, 1, 0, true, -1>((&(((cutlass::bfloat16_t*)buf_dyn_shmem)[0])), (&(((cutlass::bfloat16_t*)buf_dyn_shmem)[((((k_1 + 1) & 1) * 22528) + 16384)])), (&(acc_s[0])));

      //tl::fence_proxy_async();
      if (tl::mbarrier_try_wait(v_load_barrier[((k_1 & 1))], ((k_1 & 3) >> 1)) == 0) {
        tl::mbarrier_wait(v_load_barrier[((k_1 & 1))], ((k_1 & 3) >> 1));
      }
      // tl::mbarrier_wait(v_load_barrier[((k_1 & 1))], ((k_1 & 3) >> 1));
      tl::gemm_rs<128, 128, 176, 8, 1, 0, 0, 0, true, -1>((&(acc_s_cast[0])), (&(((cutlass::bfloat16_t*)buf_dyn_shmem)[(((k_1 & 1) * 22528) + 61440)])), (&(acc_o[0])));
      //warp_scheduler_barrier_arrive();
      // int const next_WG = 1 - warp_group_idx;
      // cutlass::arch::NamedBarrier::arrive(256, next_WG /*id*/);

      cute::warpgroup_wait<1>();
      // tl::mbarrier_arrive(k_full_barrier[(((k_1 + 1) & 1))]);
      // tl::mbarrier_arrive(k_full_barrier[(((k_1 + 1) & 1))], 0, uint32_t(threadIdx.x % 128 == 0));
      tl::mbarrier_arrive(k_full_barrier[(((k_1 + 1) & 1))], 0, uint32_t(threadIdx.x % 128 == 0));
      // int k_barrier_idx_2 = (((k_1 + 1) & 1));
      // tl::mbarrier_arrive(k_full_barrier[k_barrier_idx_2], 0, uint32_t(threadIdx.x % 128 == 0));
      #pragma unroll
      for (int i_14 = 0; i_14 < 2; ++i_14) {
        scores_max_prev[i_14] = scores_max[i_14];
      }

      #pragma unroll
      for (int i_16 = 0; i_16 < 2; ++i_16) {
        #pragma unroll
        for (int rv_2 = 0; rv_2 < 44; ++rv_2) {
          scores_max[i_16] = max(scores_max[i_16], acc_s[((((rv_2 % 22) * 4) + (i_16 * 2)) + (rv_2 / 22))]);
        }
        scores_max[i_16] = tl::AllReduce<tl::MaxOp, 4, 1>::run_hopper(scores_max[i_16]);
      }
      #pragma unroll
      for (int i_17 = 0; i_17 < 2; ++i_17) {
        scores_scale[i_17] = exp2f(((scores_max_prev[i_17] * 1.275174e-01f) - (scores_max[i_17] * 1.275174e-01f)));     
      }
      #pragma unroll
      for (int i_17 = 0; i_17 < 2; ++i_17) {
        logsum[i_17] *= scores_scale[i_17];
      }

      // #pragma unroll
      // for (int i_18 = 0; i_18 < 88; ++i_18) {
      //  acc_s[i_18] = exp2f(((acc_s[i_18] * 1.275174e-01f) - (scores_max[((i_18 & 3) >> 1)] * 1.275174e-01f)));
      // }
      for (int i_18_p3 = 0; i_18_p3 < 2; ++i_18_p3)
      {
          auto scales_max_local = scores_max[i_18_p3] * 1.275174e-01f;
          for (int i_18_p2 = 0; i_18_p2 < 22; ++i_18_p2)
          {
              auto i_18_p0 = i_18_p2 * 2 + i_18_p3;
              for (int i_18_p1 = 0; i_18_p1 < 2; ++i_18_p1)
              {
                  int i_18 = i_18_p0 * 2 + i_18_p1;
                  acc_s[i_18] = exp2f(acc_s[i_18] * 1.275174e-01f - scales_max_local);
              }
          }
      }
      #pragma unroll
      for (int i_19 = 0; i_19 < 2; ++i_19) {
        //scores_sum[i_19] = 0.000000e+00f;
        #pragma unroll
        for (int rv_3 = 0; rv_3 < 44; ++rv_3) {
          logsum[i_19] = (logsum[i_19] + acc_s[((((rv_3 % 22) * 4) + (i_19 * 2)) + (rv_3 / 22))]);
        }
        //scores_sum[i_19] = tl::AllReduce<tl::SumOp, 4, 1>::run_hopper(scores_sum[i_19]);
      }

      //tl::fence_proxy_async();
      cute::warpgroup_wait<0>();
      // tl::mbarrier_arrive(v_full_barrier[((k_1 & 1))]);
      tl::mbarrier_arrive(v_full_barrier[((k_1 & 1))], 0, uint32_t(threadIdx.x % 128 == 0));
      // int v_barrier_idx = ((k_1 & 1));
      // tl::mbarrier_arrive(v_full_barrier[v_barrier_idx], 0, uint32_t(threadIdx.x % 128 == 0));
      #pragma unroll
      for (int i_21 = 0; i_21 < 44; ++i_21) {
        //uint1 __2;
        //float2 v__1 = *(float2*)(acc_s + (i_21 * 2));
        //((nv_bfloat162*)(&(__2.x)))->x = (cutlass::bfloat16_t)(v__1.x);
        //((nv_bfloat162*)(&(__2.x)))->y = (cutlass::bfloat16_t)(v__1.y);
        //*(uint1*)(acc_s_cast + (i_21 * 2)) = __2;

        uint1 __2;
        float2 v__1 = *(float2*)(acc_s + (i_21 * 2));
        reinterpret_cast<__nv_bfloat162 &>(__2) = __float22bfloat162_rn(reinterpret_cast<float2 const &>(v__1));
        *(uint1*)(acc_s_cast + (i_21 * 2)) = __2; 
      }

      #pragma unroll
      for (int i_13 = 0; i_13 < 64; ++i_13) {
        acc_o[i_13] = (acc_o[i_13] * scores_scale[((i_13 & 3) >> 1)]);
      }
    }
    //cutlass::arch::NamedBarrier::arrive(160, static_cast<uint32_t>(2) /*id*/);

    //tl::fence_proxy_async();
    if (tl::mbarrier_try_wait(v_load_barrier[1], 0) == 0) {
      tl::mbarrier_wait(v_load_barrier[1], 0);
    }
    // tl::mbarrier_wait(v_load_barrier[1], 0);
    tl::gemm_rs<128, 128, 176, 8, 1, 0, 0, 0, true, -1>((&(acc_s_cast[0])), (&(((cutlass::bfloat16_t*)buf_dyn_shmem)[83968])), (&(acc_o[0])));
    // tl::mbarrier_arrive(v_full_barrier[1]);
    // tl::mbarrier_arrive(v_full_barrier[1], 0, uint32_t(threadIdx.x % 128 == 0));
    #pragma unroll
    for (int i_19 = 0; i_19 < 2; ++i_19) {
      logsum[i_19] = tl::AllReduce<tl::SumOp, 4, 1>::run_hopper(logsum[i_19]);
    }
    #pragma unroll
    for (int i_19 = 0; i_19 < 2; ++i_19) {
      scores_scale[i_19] = 1.f/ logsum[i_19];
      logsum[i_19] = scores_max[i_19] * (1.275174e-01f * 6.931472e+00f) + __logf(logsum[i_19]);
    }
    cute::warpgroup_wait<0>();
    tl::mbarrier_arrive(v_full_barrier[1], 0, uint32_t(threadIdx.x % 128 == 0));
    #pragma unroll
    for (int i_22 = 0; i_22 < 64; ++i_22) {
      acc_o[i_22] = (acc_o[i_22] * scores_scale[((i_22 & 3) >> 1)]);
    }
    //tl::syncthreads_partial(_mbarrier[9]);
    //flash::named_barrier_sync(256, cutlass::arch::ReservedNamedBarriers::EpilogueBarrier);

    asm volatile("bar.sync %0, %1;" : : "r"(1), "r"(256));
    // cutlass::arch::synclog_emit_named_barrier_arrive_and_wait(__LINE__, 256, 1);

    #pragma unroll
    for (int i_24 = 0; i_24 < 8; ++i_24) {
      //tl::ptx_stmatrix_x4((&(((cutlass::bfloat16_t*)buf_dyn_shmem)[(((((((int)threadIdx.x) >> 5) * 2048) + ((((int)threadIdx.x) & 15) * 128)) + (i_24 * 16)) + (((((int)threadIdx.x) & 31) >> 4) * 8))])), __pack_half2(((cutlass::bfloat16_t)acc_o[(i_24 * 8)]), ((cutlass::bfloat16_t)acc_o[((i_24 * 8) + 1)])), __pack_half2(((cutlass::bfloat16_t)acc_o[((i_24 * 8) + 2)]), ((cutlass::bfloat16_t)acc_o[((i_24 * 8) + 3)])), __pack_half2(((cutlass::bfloat16_t)acc_o[((i_24 * 8) + 4)]), ((cutlass::bfloat16_t)acc_o[((i_24 * 8) + 5)])), __pack_half2(((cutlass::bfloat16_t)acc_o[((i_24 * 8) + 6)]), ((cutlass::bfloat16_t)acc_o[((i_24 * 8) + 7)])));
      tl::ptx_stmatrix_x4((&(((bfloat16_t*)buf_dyn_shmem)[(((((((i_24 >> 2) * 8192) + ((((int)threadIdx.x) >> 5) * 1024)) + ((((int)threadIdx.x) & 15) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + ((i_24 & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (i_24 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))])), __pack_half2(((bfloat16_t)acc_o[(i_24 * 8)]), ((bfloat16_t)acc_o[((i_24 * 8) + 1)])), __pack_half2(((bfloat16_t)acc_o[((i_24 * 8) + 2)]), ((bfloat16_t)acc_o[((i_24 * 8) + 3)])), __pack_half2(((bfloat16_t)acc_o[((i_24 * 8) + 4)]), ((bfloat16_t)acc_o[((i_24 * 8) + 5)])), __pack_half2(((bfloat16_t)acc_o[((i_24 * 8) + 6)]), ((bfloat16_t)acc_o[((i_24 * 8) + 7)])));
    }
    tl::fence_proxy_async();
    //tl::syncthreads_partial(_mbarrier[10]);

    cutlass::arch::NamedBarrier::arrive(256 + 32, 1);
    int warp_idx_sync = __shfl_sync(0xffffffff, threadIdx.x / 32, 0);
    if (warp_idx_sync == 256 / 32 - 1) {
        cutlass::arch::NamedBarrier::sync(256 + 32,
                                            1);
    // }
    // if (((int)threadIdx.x) == 0) {
    if (cute::elect_one_sync()) {
      // tl::tma_store(Output_desc, _mbarrier[5], (&(((cutlass::bfloat16_t*)buf_dyn_shmem)[0])), 0, ((int)blockIdx.y), (((int)blockIdx.x) * 128), 0);
      tl::tma_store(Output_desc, (&(((cutlass::bfloat16_t*)buf_dyn_shmem)[0])), 0, ((int)blockIdx.y), (((int)blockIdx.x) * 128), 0);
      tl::tma_store_arrive();
      tl::tma_store_wait<0>();
      tl::tma_store(Output_desc, (&(((bfloat16_t*)buf_dyn_shmem)[8192])), 64, ((int)blockIdx.y), (((int)blockIdx.x) * 128), 0);
      tl::tma_store_arrive();
      tl::tma_store_wait<0>();
    }
  }
  }
}
    """
    return code


def flashattn(batch, heads, seq_len, dim, is_causal, tune=False, shufl=False):
    scale = (1.0 / dim)**0.5 * 1.44269504  # log2(e)
    shape = [batch, seq_len, heads, dim]
    dtype = "bfloat16"
    accum_dtype = "float"

    @tilelang.jit(out_idx=[3], compile_flags=["--use_fast_math", 
                "--expt-relaxed-constexpr", 
                "-DCUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED",
                "-DCUTLASS_ENABLE_GDC_FOR_SM90",
                "--resource-usage",
                "-O3", "-DENABLE_BF16 -DCUTE_ARCH_ELECT_ONE_SM90_ENABLED"], pass_configs={
                  "tl.disable_shuffle_elect": not shufl,
                })
    def kernel_func(block_M, block_N, num_stages, threads):

        @T.macro
        def MMA0(
            K: T.Tensor(shape, dtype),
            Q_shared: T.SharedBuffer([block_M, dim], dtype),
            K_shared: T.SharedBuffer([block_N, dim], dtype),
            acc_s: T.FragmentBuffer([block_M, block_N], accum_dtype),
            k: T.int32,
            bx: T.int32,
            by: T.int32,
            bz: T.int32,
        ):
            T.copy(K[bz, k * block_N:(k + 1) * block_N, by, :], K_shared)
            if is_causal:
                for i, j in T.Parallel(block_M, block_N):
                    acc_s[i, j] = T.if_then_else(bx * block_M + i >= k * block_N + j, 0,
                                                 -T.infinity(acc_s.dtype))
            else:
                T.clear(acc_s)
            T.gemm(Q_shared, K_shared, acc_s, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)

        @T.macro
        def MMA1(
            V: T.Tensor(shape, dtype),
            V_shared: T.SharedBuffer([block_M, dim], dtype),
            acc_s_cast: T.FragmentBuffer([block_M, block_N], dtype),
            acc_o: T.FragmentBuffer([block_M, dim], accum_dtype),
            k: T.int32,
            by: T.int32,
            bz: T.int32,
        ):
            T.copy(V[bz, k * block_N:(k + 1) * block_N, by, :], V_shared)
            T.gemm(acc_s_cast, V_shared, acc_o, policy=T.GemmWarpPolicy.FullRow)

        @T.macro
        def Softmax(
                acc_s: T.FragmentBuffer([block_M, block_N], accum_dtype),
                acc_s_cast: T.FragmentBuffer([block_M, block_N], dtype),
                scores_max: T.FragmentBuffer([block_M], accum_dtype),
                scores_max_prev: T.FragmentBuffer([block_M], accum_dtype),
                scores_scale: T.FragmentBuffer([block_M], accum_dtype),
                logsum: T.FragmentBuffer([block_M], accum_dtype),
        ):
            T.copy(scores_max, scores_max_prev)
            T.fill(scores_max, -T.infinity(accum_dtype))
            T.reduce_max(acc_s, scores_max, dim=1, clear=False)
            # To do causal softmax, we need to set the scores_max to 0 if it is -inf
            # This process is called Check_inf in FlashAttention3 code, and it only need to be done
            # in the first ceil_div(kBlockM, kBlockN) steps.
            # for i in T.Parallel(block_M):
            #     scores_max[i] = T.if_then_else(scores_max[i] == -T.infinity(accum_dtype), 0, scores_max[i])
            for i in T.Parallel(block_M):
                scores_scale[i] = T.exp2(scores_max_prev[i] * scale - scores_max[i] * scale)
            for i, j in T.Parallel(block_M, block_N):
                # Instead of computing exp(x - max), we compute exp2(x * log_2(e) -
                # max * log_2(e)) This allows the compiler to use the ffma
                # instruction instead of fadd and fmul separately.
                acc_s[i, j] = T.exp2(acc_s[i, j] * scale - scores_max[i] * scale)

            T.reduce_sum(acc_s, logsum, dim=1, clear=False, scale="intra-thread")
            # reducer.intra_thread()
            for i in T.Parallel(block_M):
                logsum[i] *= scores_scale[i]
            T.copy(acc_s, acc_s_cast)

        @T.macro
        def Rescale(
                acc_o: T.FragmentBuffer([block_M, dim], accum_dtype),
                scores_scale: T.FragmentBuffer([block_M], accum_dtype),
        ):
            for i, j in T.Parallel(block_M, dim):
                acc_o[i, j] *= scores_scale[i]

        @T.prim_func
        def main(
                Q: T.Tensor(shape, dtype),
                K: T.Tensor(shape, dtype),
                V: T.Tensor(shape, dtype),
                Output: T.Tensor(shape, dtype),
        ):
            with T.Kernel(
                    T.ceildiv(seq_len, block_M), heads, batch, threads=threads) as (bx, by, bz):
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
                logsum = T.alloc_fragment([block_M], accum_dtype)

                T.annotate_layout({
                    O_shared: tilelang.layout.make_swizzled_layout(O_shared),
                })

                T.copy(Q[bz, bx * block_M:(bx + 1) * block_M, by, :], Q_shared)
                T.fill(acc_o, 0)
                T.fill(logsum, 0)
                T.fill(scores_max, -T.infinity(accum_dtype))

                loop_range = (
                    T.min(T.ceildiv(seq_len, block_N), T.ceildiv(
                        (bx + 1) * block_M, block_N)) if is_causal else T.ceildiv(seq_len, block_N))

                for k in T.Pipelined(
                        loop_range,
                        num_stages=num_stages,
                        order=[-1, 0, 3, 1, -1, 2],
                        stage=[-1, 0, 0, 1, -1, 1],
                        group=[[0], [1, 2], [3, 4, 5, 6, 7, 8, 9, 10], [11], [12], [13]]
                        ):
                    
                    MMA0(K, Q_shared, K_shared, acc_s, k, bx, by, bz)
                    Softmax(acc_s, acc_s_cast, scores_max, scores_max_prev, scores_scale,
                            logsum)
                    Rescale(acc_o, scores_scale)
                    MMA1(V, V_shared, acc_s_cast, acc_o, k, by, bz)
                T.reduce_sum(acc_s, logsum, dim=1, clear=False, scale="inter-thread")
                for i, j in T.Parallel(block_M, dim):
                    acc_o[i, j] /= logsum[i]
                T.copy(acc_o, O_shared)
                T.copy(O_shared, Output[bz, bx * block_M:(bx + 1) * block_M, by, :])

        return main

    if tune:

        @autotune(configs=get_configs(), warmup=10, rep=10)
        @tilelang.jit(out_idx=[3])
        def kernel(block_M=None, block_N=None, num_stages=None, threads=None):
            return kernel_func(block_M, block_N, num_stages, threads)

        return kernel()
    else:

        def kernel(block_M, block_N, num_stages, threads):
            return kernel_func(block_M, block_N, num_stages, threads)

        return kernel


def ref_program(Q, K, V, is_causal):
    # dim = Q.size(-1)
    # scores = torch.einsum('bqhd,bkhd->bhqk', Q, K)
    # scores = scores / torch.sqrt(torch.tensor(dim, dtype=scores.dtype))
    # if is_causal:
    #     seq_len = Q.size(1)
    #     mask = torch.tril(torch.ones(seq_len, seq_len, device=scores.device))
    #     mask = mask.unsqueeze(0).unsqueeze(0)
    #     scores = scores.masked_fill(mask == 0, float('-inf'))
    # attention_weights = F.softmax(scores, dim=-1)
    # output = torch.einsum('bhqk,bkhd->bqhd', attention_weights, V)
    # return output
    import flash_attn_interface
    return flash_attn_interface.flash_attn_func(Q, K, V, causal=is_causal)


def main(
    batch: int = 8,
    heads: int = 32,
    seq_len: int = 4096,
    dim: int = 128,
    is_causal: bool = False,
    tune: bool = False,
    shl: bool = False,
):
    flops_per_matmul = 2.0 * batch * heads * seq_len * seq_len * dim
    total_flops = 2 * flops_per_matmul
    if is_causal:
        total_flops *= 0.5

    if (not tune):
        kernel = flashattn(
            batch, heads, seq_len, dim, is_causal, tune=tune, shufl=shl)(
                block_M=128, block_N=176, num_stages=2, threads=256)
        # print(kernel.get_kernel_source())
        # exit(0)
        """
        All checks pass.
        Ref: 20.26 ms
        Ref: 434.24 TFlops
        Tile-lang: 26.66 ms
        Tile-lang: 329.94 TFlops

        All checks pass.
Ref: 20.55 ms
Ref: 427.93 TFlops
Tile-lang: 22.71 ms
Tile-lang: 387.27 TFlops

        """
        ref_program_processed = partial(ref_program, is_causal=is_causal)
        profiler = kernel.get_profiler(tensor_supply_type=tilelang.TensorSupplyType.Randn)
        profiler.assert_allclose(ref_program_processed, rtol=0.01, atol=0.01)
        print("All checks pass.")
        latency = profiler.do_bench(warmup=50, n_repeat=100)
        print("Tile-lang: {:.2f} ms".format(latency))
        print("Tile-lang: {:.2f} TFlops".format(total_flops / latency * 1e-9))
        latency = profiler.do_bench(ref_program_processed, warmup=50, n_repeat=100)
        print("Ref: {:.2f} ms".format(latency))
        print("Ref: {:.2f} TFlops".format(total_flops / latency * 1e-9))
    else:
        best_result = flashattn(batch, heads, seq_len, dim, is_causal, tune=tune)
        best_latency = best_result.latency
        best_config = best_result.config
        ref_latency = best_result.ref_latency
        print(f"Best latency: {best_latency}")
        print(f"Best TFlops: {total_flops / best_latency * 1e-9}")
        print(f"Best config: {best_config}")
        print(f"Ref latency: {ref_latency}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch', type=int, default=8, help='batch size')
    parser.add_argument('--heads', type=int, default=32, help='heads')
    parser.add_argument('--seq_len', type=int, default=4096, help='sequence length')
    parser.add_argument('--dim', type=int, default=128, help='dim')
    parser.add_argument('--is_causal', action='store_true', help='causal')
    parser.add_argument('--tune', action='store_true', help='tune configs')
    parser.add_argument('--shl', action='store_true', help='use shl for tuning')
    args = parser.parse_args()
    main(args.batch, args.heads, args.seq_len, args.dim, args.is_causal, args.tune, args.shl)