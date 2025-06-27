
#include <tl_templates/aica/copy.h>
#include <tl_templates/aica/gemm.h>
#include <tl_templates/aica/reduce.h>
#include <tl_templates/aica/threadblock_swizzle.h>

__global__ void __launch_bounds__(128, 1) main_kernel(half_t* __restrict__ K, half_t* __restrict__ Output, half_t* __restrict__ Q, half_t* __restrict__ V) {
  extern __shared__ __align__(1024) uchar buf_dyn_shmem[];
  float acc_o[32];
  float logsum[2];
  float scores_max[2];
  float acc_s[32];
  float scores_max_prev[2];
  float scores_scale[2];
  float scores_sum[2];
  #pragma unroll
  for (int i = 0; i < 4; ++i) {
    *(uint4*)(((half_t*)buf_dyn_shmem) + ((((((i * 1024) + ((((int)threadIdx.x) >> 3) * 64)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 12288)) = *(uint4*)(Q + ((((((((int)blockIdx.z) * 262144) + (((int)blockIdx.x) * 131072)) + (i * 32768)) + ((((int)threadIdx.x) >> 3) * 2048)) + (((int)blockIdx.y) * 64)) + ((((int)threadIdx.x) & 7) * 8)));
  }
  #pragma unroll
  for (int i_1 = 0; i_1 < 32; ++i_1) {
    acc_o[i_1] = 0.000000e+00f;
  }
  #pragma unroll
  for (int i_2 = 0; i_2 < 2; ++i_2) {
    logsum[i_2] = 0.000000e+00f;
  }
  #pragma unroll
  for (int i_3 = 0; i_3 < 2; ++i_3) {
    scores_max[i_3] = -CUDART_INF_F;
  }
  #pragma unroll
  for (int i_4 = 0; i_4 < 4; ++i_4) {
    tl::cp_async_gs<16>(buf_dyn_shmem+(((((i_4 * 2048) + ((((int)threadIdx.x) >> 3) * 128)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 64)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 16)), K+(((((((int)blockIdx.z) * 262144) + (i_4 * 32768)) + ((((int)threadIdx.x) >> 3) * 2048)) + (((int)blockIdx.y) * 64)) + ((((int)threadIdx.x) & 7) * 8)));
  }
  tl::cp_async_commit();
  #pragma unroll
  for (int i_5 = 0; i_5 < 4; ++i_5) {
    tl::cp_async_gs<16>(buf_dyn_shmem+((((((i_5 * 2048) + ((((int)threadIdx.x) >> 3) * 128)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 64)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 16)) + 16384), V+(((((((int)blockIdx.z) * 262144) + (i_5 * 32768)) + ((((int)threadIdx.x) >> 3) * 2048)) + (((int)blockIdx.y) * 64)) + ((((int)threadIdx.x) & 7) * 8)));
  }
  tl::cp_async_commit();
  #pragma unroll
  for (int i_6 = 0; i_6 < 32; ++i_6) {
    acc_s[i_6] = 0.000000e+00f;
  }
  tl::cp_async_wait<1>();
  __syncthreads();
  tl::gemm_ss<64, 64, 64, 4, 1, 0, 1, 0>((&(((half_t*)buf_dyn_shmem)[12288])), (&(((half_t*)buf_dyn_shmem)[0])), (&(acc_s[0])));
  __syncthreads();
  #pragma unroll
  for (int i_7 = 0; i_7 < 4; ++i_7) {
    tl::cp_async_gs<16>(buf_dyn_shmem+(((((i_7 * 2048) + ((((int)threadIdx.x) >> 3) * 128)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 64)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 16)), K+((((((((int)blockIdx.z) * 262144) + (i_7 * 32768)) + ((((int)threadIdx.x) >> 3) * 2048)) + (((int)blockIdx.y) * 64)) + ((((int)threadIdx.x) & 7) * 8)) + 131072));
  }
  tl::cp_async_commit();
  #pragma unroll
  for (int i_8 = 0; i_8 < 2; ++i_8) {
    scores_max_prev[i_8] = scores_max[i_8];
  }
  #pragma unroll
  for (int i_9 = 0; i_9 < 2; ++i_9) {
    scores_max[i_9] = -CUDART_INF_F;
  }
  #pragma unroll
  for (int i_10 = 0; i_10 < 2; ++i_10) {
    #pragma unroll
    for (int rv = 0; rv < 16; ++rv) {
      scores_max[i_10] = max(scores_max[i_10], acc_s[((((rv & 7) * 4) + (i_10 * 2)) + (rv >> 3))]);
    }
    scores_max[i_10] = tl::AllReduce<tl::MaxOp, 4, 1>::run(scores_max[i_10]);
  }
  #pragma unroll
  for (int i_11 = 0; i_11 < 2; ++i_11) {
    scores_scale[i_11] = exp2f(((scores_max_prev[i_11] * 1.803369e-01f) - (scores_max[i_11] * 1.803369e-01f)));
  }
  #pragma unroll
  for (int i_12 = 0; i_12 < 32; ++i_12) {
    acc_s[i_12] = exp2f(((acc_s[i_12] * 1.803369e-01f) - (scores_max[((i_12 & 3) >> 1)] * 1.803369e-01f)));
  }
  #pragma unroll
  for (int i_13 = 0; i_13 < 2; ++i_13) {
    scores_sum[i_13] = 0.000000e+00f;
    #pragma unroll
    for (int rv_1 = 0; rv_1 < 16; ++rv_1) {
      scores_sum[i_13] = (scores_sum[i_13] + acc_s[((((rv_1 & 7) * 4) + (i_13 * 2)) + (rv_1 >> 3))]);
    }
    scores_sum[i_13] = tl::AllReduce<tl::SumOp, 4, 1>::run(scores_sum[i_13]);
  }
  #pragma unroll
  for (int i_14 = 0; i_14 < 2; ++i_14) {
    logsum[i_14] = ((logsum[i_14] * scores_scale[i_14]) + scores_sum[i_14]);
  }
  #pragma unroll
  for (int i_15 = 0; i_15 < 32; ++i_15) {
    ((half_t*)buf_dyn_shmem)[((((((((((((int)threadIdx.x) >> 5) * 1024) + (((i_15 & 3) >> 1) * 512)) + (((((int)threadIdx.x) & 31) >> 2) * 64)) + ((((i_15 >> 4) + ((((int)threadIdx.x) & 31) >> 4)) & 1) * 32)) + (((((i_15 & 15) >> 3) + ((((int)threadIdx.x) & 15) >> 3)) & 1) * 16)) + (((((i_15 & 7) >> 2) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 8)) + ((i_15 & 1) * 4)) + (((int)threadIdx.x) & 3)) + 4096)] = ((half_t)acc_s[i_15]);
  }
  #pragma unroll
  for (int i_16 = 0; i_16 < 32; ++i_16) {
    acc_o[i_16] = (acc_o[i_16] * scores_scale[((i_16 & 3) >> 1)]);
  }
  tl::cp_async_wait<1>();
  __syncthreads();
  tl::gemm_ss<64, 64, 64, 4, 1, 0, 0, 0>((&(((half_t*)buf_dyn_shmem)[4096])), (&(((half_t*)buf_dyn_shmem)[8192])), (&(acc_o[0])));
  __syncthreads();
  #pragma unroll
  for (int i_17 = 0; i_17 < 4; ++i_17) {
    tl::cp_async_gs<16>(buf_dyn_shmem+((((((i_17 * 2048) + ((((int)threadIdx.x) >> 3) * 128)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 64)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 16)) + 16384), V+((((((((int)blockIdx.z) * 262144) + (i_17 * 32768)) + ((((int)threadIdx.x) >> 3) * 2048)) + (((int)blockIdx.y) * 64)) + ((((int)threadIdx.x) & 7) * 8)) + 131072));
  }
  tl::cp_async_commit();
  #pragma unroll
  for (int i_18 = 0; i_18 < 32; ++i_18) {
    acc_s[i_18] = 0.000000e+00f;
  }
  tl::cp_async_wait<1>();
  __syncthreads();
  tl::gemm_ss<64, 64, 64, 4, 1, 0, 1, 0>((&(((half_t*)buf_dyn_shmem)[12288])), (&(((half_t*)buf_dyn_shmem)[0])), (&(acc_s[0])));
  #pragma unroll
  for (int i_19 = 0; i_19 < 2; ++i_19) {
    scores_max_prev[i_19] = scores_max[i_19];
  }
  #pragma unroll
  for (int i_20 = 0; i_20 < 2; ++i_20) {
    scores_max[i_20] = -CUDART_INF_F;
  }
  #pragma unroll
  for (int i_21 = 0; i_21 < 2; ++i_21) {
    #pragma unroll
    for (int rv_2 = 0; rv_2 < 16; ++rv_2) {
      scores_max[i_21] = max(scores_max[i_21], acc_s[((((rv_2 & 7) * 4) + (i_21 * 2)) + (rv_2 >> 3))]);
    }
    scores_max[i_21] = tl::AllReduce<tl::MaxOp, 4, 1>::run(scores_max[i_21]);
  }
  #pragma unroll
  for (int i_22 = 0; i_22 < 2; ++i_22) {
    scores_scale[i_22] = exp2f(((scores_max_prev[i_22] * 1.803369e-01f) - (scores_max[i_22] * 1.803369e-01f)));
  }
  #pragma unroll
  for (int i_23 = 0; i_23 < 32; ++i_23) {
    acc_s[i_23] = exp2f(((acc_s[i_23] * 1.803369e-01f) - (scores_max[((i_23 & 3) >> 1)] * 1.803369e-01f)));
  }
  #pragma unroll
  for (int i_24 = 0; i_24 < 2; ++i_24) {
    scores_sum[i_24] = 0.000000e+00f;
    #pragma unroll
    for (int rv_3 = 0; rv_3 < 16; ++rv_3) {
      scores_sum[i_24] = (scores_sum[i_24] + acc_s[((((rv_3 & 7) * 4) + (i_24 * 2)) + (rv_3 >> 3))]);
    }
    scores_sum[i_24] = tl::AllReduce<tl::SumOp, 4, 1>::run(scores_sum[i_24]);
  }
  #pragma unroll
  for (int i_25 = 0; i_25 < 2; ++i_25) {
    logsum[i_25] = ((logsum[i_25] * scores_scale[i_25]) + scores_sum[i_25]);
  }
  __syncthreads();
  #pragma unroll
  for (int i_26 = 0; i_26 < 32; ++i_26) {
    ((half_t*)buf_dyn_shmem)[((((((((((((int)threadIdx.x) >> 5) * 1024) + (((i_26 & 3) >> 1) * 512)) + (((((int)threadIdx.x) & 31) >> 2) * 64)) + ((((i_26 >> 4) + ((((int)threadIdx.x) & 31) >> 4)) & 1) * 32)) + (((((i_26 & 15) >> 3) + ((((int)threadIdx.x) & 15) >> 3)) & 1) * 16)) + (((((i_26 & 7) >> 2) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 8)) + ((i_26 & 1) * 4)) + (((int)threadIdx.x) & 3)) + 4096)] = ((half_t)acc_s[i_26]);
  }
  #pragma unroll
  for (int i_27 = 0; i_27 < 32; ++i_27) {
    acc_o[i_27] = (acc_o[i_27] * scores_scale[((i_27 & 3) >> 1)]);
  }
  tl::cp_async_wait<0>();
  __syncthreads();
  tl::gemm_ss<64, 64, 64, 4, 1, 0, 0, 0>((&(((half_t*)buf_dyn_shmem)[4096])), (&(((half_t*)buf_dyn_shmem)[8192])), (&(acc_o[0])));
  #pragma unroll
  for (int i_28 = 0; i_28 < 32; ++i_28) {
    acc_o[i_28] = (acc_o[i_28] / logsum[((i_28 & 3) >> 1)]);
  }
  __syncthreads();
  #pragma unroll
  for (int i_29 = 0; i_29 < 32; ++i_29) {
    ((half_t*)buf_dyn_shmem)[((((((((((int)threadIdx.x) >> 5) * 1024) + (((i_29 & 3) >> 1) * 512)) + (((((int)threadIdx.x) & 31) >> 2) * 64)) + ((i_29 >> 2) * 8)) + ((i_29 & 1) * 4)) + (((int)threadIdx.x) & 3)) + 12288)] = ((half_t)acc_o[i_29]);
  }
  __syncthreads();
  #pragma unroll
  for (int i_30 = 0; i_30 < 4; ++i_30) {
    *(uint4*)(Output + ((((((((int)blockIdx.z) * 262144) + (((int)blockIdx.x) * 131072)) + (i_30 * 32768)) + ((((int)threadIdx.x) >> 3) * 2048)) + (((int)blockIdx.y) * 64)) + ((((int)threadIdx.x) & 7) * 8))) = *(uint4*)(((half_t*)buf_dyn_shmem) + (((i_30 * 1024) + (((int)threadIdx.x) * 8)) + 12288));
  }
}

extern "C" int call(half_t* __restrict__ Q, half_t* __restrict__ K, half_t* __restrict__ V, half_t* __restrict__ Output) {
    
    main_kernel<<<dim3(2, 32, 8), dim3(128, 1, 1), 32768>>>(K, Output, Q, V);
    return 0;
}
