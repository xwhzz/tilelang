
#include <tl_templates/aica/copy.h>
#include <tl_templates/aica/gemm.h>
#include <tl_templates/aica/reduce.h>
#include <tl_templates/aica/threadblock_swizzle.h>

__global__ void __launch_bounds__(128, 1) main_kernel(float* __restrict__ A, float* __restrict__ B) {
  extern __shared__ __align__(1024) uchar buf_dyn_shmem[];
  float A_local[8];
  float A_pow_local[8];
  float A_powsum[1];
  #pragma unroll
  for (int i = 0; i < 2; ++i) {
    *(float4*)(((float*)buf_dyn_shmem) + ((i * 512) + (((int)threadIdx.x) * 4))) = *(float4*)(A + (((((int)blockIdx.x) * 1024) + (i * 512)) + (((int)threadIdx.x) * 4)));
  }
  __syncthreads();
  #pragma unroll
  for (int i_1 = 0; i_1 < 2; ++i_1) {
    *(float4*)(A_local + (i_1 * 4)) = *(float4*)(((float*)buf_dyn_shmem) + ((i_1 * 512) + (((int)threadIdx.x) * 4)));
  }
  #pragma unroll
  for (int i_2 = 0; i_2 < 8; ++i_2) {
    A_pow_local[i_2] = (A_local[i_2] * A_local[i_2]);
  }
  __syncthreads();
  #pragma unroll
  for (int i_3 = 0; i_3 < 1; ++i_3) {
    A_powsum[0] = 0.000000e+00f;
    #pragma unroll
    for (int rv = 0; rv < 8; ++rv) {
      A_powsum[0] = (A_powsum[0] + A_pow_local[(((rv & 1) * 4) + (rv >> 1))]);
    }
    A_powsum[0] = tl::AllReduce<tl::SumOp, 128, 1>::run(A_powsum[0], (&(((float*)buf_dyn_shmem)[0])));
  }
  #pragma unroll
  for (int i_4 = 0; i_4 < 1; ++i_4) {
    A_powsum[0] = ((1.000000e+00f / sqrtf((A_powsum[0] * 9.765625e-04f))) + 1.000000e-12f);
  }
  #pragma unroll
  for (int i_5 = 0; i_5 < 8; ++i_5) {
    A_local[i_5] = (A_local[i_5] * A_powsum[0]);
  }
  #pragma unroll
  for (int i_6 = 0; i_6 < 2; ++i_6) {
    *(float4*)(B + (((((int)blockIdx.x) * 1024) + (i_6 * 512)) + (((int)threadIdx.x) * 4))) = *(float4*)(A_local + (i_6 * 4));
  }
}


extern "C" int call(float* __restrict__ A, float* __restrict__ B) {
        main_kernel<<<dim3(1024, 1, 1), dim3(128, 1, 1), 4096>>>(A, B);
        return 0;
}