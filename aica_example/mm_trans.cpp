
#include <tl_templates/aica/copy.h>
#include <tl_templates/aica/gemm.h>
#include <tl_templates/aica/reduce.h>
#include <tl_templates/aica/threadblock_swizzle.h>

// __global__ void __launch_bounds__(128, 1) main_kernel(half_t* __restrict__ K, half_t* __restrict__ Output, half_t* __restrict__ Q, half_t* __restrict__ V) {
__global__ void __launch_bounds__(128, 1) main_kernel(half_t* __restrict__ A, half_t* __restrict__ B, half_t* __restrict__ C) {
  extern __shared__ __align__(1024) uchar buf_dyn_shmem[];
  float C_local[128];
  #pragma unroll
  for (int i = 0; i < 128; ++i) {
    C_local[i] = 0.000000e+00f;
  }
  #pragma unroll
  for (int i_1 = 0; i_1 < 4; ++i_1) {
    tl::cp_async_gs<16>(buf_dyn_shmem+(((((i_1 * 2048) + ((((int)threadIdx.x) >> 2) * 64)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 16)) + 16384), A+((((((int)blockIdx.y) * 131072) + (i_1 * 32768)) + ((((int)threadIdx.x) >> 2) * 1024)) + ((((int)threadIdx.x) & 3) * 8)));
  }
  #pragma unroll
  for (int i_2 = 0; i_2 < 4; ++i_2) {
    tl::cp_async_gs<16>(buf_dyn_shmem+((((i_2 * 2048) + ((((int)threadIdx.x) >> 2) * 64)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 16)), B+((((((int)blockIdx.x) * 131072) + (i_2 * 32768)) + ((((int)threadIdx.x) >> 2) * 1024)) + ((((int)threadIdx.x) & 3) * 8)));
  }
  tl::cp_async_commit();
  for (int k = 0; k < 31; ++k) {
    __syncthreads();
    #pragma unroll
    for (int i_3 = 0; i_3 < 4; ++i_3) {
      tl::cp_async_gs<16>(buf_dyn_shmem+((((((((k + 1) & 1) * 8192) + (i_3 * 2048)) + ((((int)threadIdx.x) >> 2) * 64)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 16)) + 16384), A+((((((((int)blockIdx.y) * 131072) + (i_3 * 32768)) + ((((int)threadIdx.x) >> 2) * 1024)) + (k * 32)) + ((((int)threadIdx.x) & 3) * 8)) + 32));
    }
    #pragma unroll
    for (int i_4 = 0; i_4 < 4; ++i_4) {
      tl::cp_async_gs<16>(buf_dyn_shmem+(((((((k + 1) & 1) * 8192) + (i_4 * 2048)) + ((((int)threadIdx.x) >> 2) * 64)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 16)), B+((((((((int)blockIdx.x) * 131072) + (i_4 * 32768)) + ((((int)threadIdx.x) >> 2) * 1024)) + (k * 32)) + ((((int)threadIdx.x) & 3) * 8)) + 32));
    }
    tl::cp_async_commit();
    tl::cp_async_wait<1>();
    __syncthreads();
    tl::gemm_ss<128, 128, 32, 2, 2, 0, 1, 0>((&(((half_t*)buf_dyn_shmem)[(((k & 1) * 4096) + 8192)])), (&(((half_t*)buf_dyn_shmem)[((k & 1) * 4096)])), (&(C_local[0])));
  }
  tl::cp_async_wait<0>();
  __syncthreads();
  tl::gemm_ss<128, 128, 32, 2, 2, 0, 1, 0>((&(((half_t*)buf_dyn_shmem)[12288])), (&(((half_t*)buf_dyn_shmem)[4096])), (&(C_local[0])));
  __syncthreads();
  #pragma unroll
  for (int i_5 = 0; i_5 < 128; ++i_5) {
    ((half_t*)buf_dyn_shmem)[((((((((((i_5 & 15) >> 2) * 4096) + (((((int)threadIdx.x) & 63) >> 5) * 2048)) + (((i_5 & 3) >> 1) * 1024)) + (((((int)threadIdx.x) & 31) >> 2) * 128)) + ((i_5 >> 4) * 16)) + ((((int)threadIdx.x) >> 6) * 8)) + ((i_5 & 1) * 4)) + (((int)threadIdx.x) & 3))] = ((half_t)C_local[i_5]);
  }
  __syncthreads();
  #pragma unroll
  for (int i_6 = 0; i_6 < 16; ++i_6) {
    *(uint4*)(C + (((((((int)blockIdx.y) * 131072) + (i_6 * 8192)) + ((((int)threadIdx.x) >> 4) * 1024)) + (((int)blockIdx.x) * 128)) + ((((int)threadIdx.x) & 15) * 8))) = *(uint4*)(((half_t*)buf_dyn_shmem) + ((i_6 * 1024) + (((int)threadIdx.x) * 8)));
  }
}
extern "C" int call(half_t* __restrict__ A, half_t* __restrict__ B, half_t* __restrict__ C) {
        main_kernel<<<dim3(8, 8, 1), dim3(128, 1, 1), 32768>>>(A, B, C);
        return 0;
}
