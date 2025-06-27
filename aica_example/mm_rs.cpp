
#include <tl_templates/aica/copy.h>
#include <tl_templates/aica/gemm.h>
#include <tl_templates/aica/threadblock_swizzle.h>

__global__ void __launch_bounds__(128, 1) gemm_kernel(half_t* __restrict__ A, half_t* __restrict__ B, half_t* __restrict__ C) {
  float C_local[128];
  extern __shared__ __align__(1024) half_t B_shared[];
  half_t A_local[64];
  #pragma unroll
  for (int i = 0; i < 128; ++i) {
    C_local[i] = 0.000000e+00f;
  }
  #pragma unroll
  for (int i_1 = 0; i_1 < 4; ++i_1) {
    tl::cp_async_gs<16>(B_shared+((((((((((int)threadIdx.x) & 15) >> 3) * 2048) + (i_1 * 512)) + ((((int)threadIdx.x) >> 4) * 64)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)), B+((((i_1 * 16384) + ((((int)threadIdx.x) >> 4) * 2048)) + (((int)blockIdx.x) * 128)) + ((((int)threadIdx.x) & 15) * 8)));
  }
  tl::cp_async_commit();
  #pragma unroll
  for (int i_2 = 0; i_2 < 4; ++i_2) {
    tl::cp_async_gs<16>(B_shared+(((((((((((int)threadIdx.x) & 15) >> 3) * 2048) + (i_2 * 512)) + ((((int)threadIdx.x) >> 4) * 64)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 4096), B+(((((i_2 * 16384) + ((((int)threadIdx.x) >> 4) * 2048)) + (((int)blockIdx.x) * 128)) + ((((int)threadIdx.x) & 15) * 8)) + 65536));
  }
  tl::cp_async_commit();
  for (int k = 0; k < 161; ++k) {
    __syncthreads();
    #pragma unroll
    for (int i_3 = 0; i_3 < 4; ++i_3) {
      tl::cp_async_gs_conditional<16>(B_shared+(((((((((k + 2) % 3) * 4096) + (((((int)threadIdx.x) & 15) >> 3) * 2048)) + (i_3 * 512)) + ((((int)threadIdx.x) >> 4) * 64)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)), B+((((((k * 65536) + (i_3 * 16384)) + ((((int)threadIdx.x) >> 4) * 2048)) + (((int)blockIdx.x) * 128)) + ((((int)threadIdx.x) & 15) * 8)) + 131072), ((((k * 4) + i_3) < 641) && (((k * 4) + i_3) < 641)));
    }
    tl::cp_async_commit();
    #pragma unroll
    for (int i_4 = 0; i_4 < 32; ++i_4) {
      *(uint1*)(A_local + (i_4 * 2)) = *(uint1*)(A + (((((((((((int)blockIdx.y) * 664576) + (((i_4 & 15) >> 2) * 166144)) + (((((int)threadIdx.x) & 63) >> 5) * 83072)) + ((i_4 & 1) * 41536)) + (((((int)threadIdx.x) & 31) >> 2) * 5192)) + (k * 32)) + ((i_4 >> 4) * 16)) + (((i_4 & 3) >> 1) * 8)) + ((((int)threadIdx.x) & 3) * 2)));
    }
    tl::cp_async_wait<2>();
    __syncthreads();
    tl::gemm_rs<128, 128, 32, 2, 2, 0, 0, 0>((&(A_local[0])), (&(B_shared[((k % 3) * 4096)])), (&(C_local[0])));
  }
  #pragma unroll
  for (int i_5 = 0; i_5 < 32; ++i_5) {
    *(uint1*)(A_local + (i_5 * 2)) = *(uint1*)(A + (((((((((((int)blockIdx.y) * 664576) + (((i_5 & 15) >> 2) * 166144)) + (((((int)threadIdx.x) & 63) >> 5) * 83072)) + ((i_5 & 1) * 41536)) + (((((int)threadIdx.x) & 31) >> 2) * 5192)) + ((i_5 >> 4) * 16)) + (((i_5 & 3) >> 1) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + 5152));
  }
  tl::cp_async_wait<1>();
  __syncthreads();
  tl::gemm_rs<128, 128, 32, 2, 2, 0, 0, 0>((&(A_local[0])), (&(B_shared[8192])), (&(C_local[0])));
  #pragma unroll
  for (int i_6 = 0; i_6 < 32; ++i_6) {
    uint1 condval;
    if (((((i_6 >> 4) * 2) + ((i_6 & 3) >> 1)) < 1)) {
      condval = *(uint1*)(A + (((((((((((int)blockIdx.y) * 664576) + (((i_6 & 15) >> 2) * 166144)) + (((((int)threadIdx.x) & 63) >> 5) * 83072)) + ((i_6 & 1) * 41536)) + (((((int)threadIdx.x) & 31) >> 2) * 5192)) + ((i_6 >> 4) * 16)) + (((i_6 & 3) >> 1) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + 5184));
    } else {
      condval = make_uint1(__pack_half2(half_t(0.000000e+00f), half_t(0.000000e+00f)));
    }
    *(uint1*)(A_local + (i_6 * 2)) = condval;
  }
  tl::cp_async_wait<0>();
  __syncthreads();
  tl::gemm_rs<128, 128, 32, 2, 2, 0, 0, 0>((&(A_local[0])), (&(B_shared[0])), (&(C_local[0])));
  #pragma unroll
  for (int i_7 = 0; i_7 < 128; ++i_7) {
    C[((((((((((((int)blockIdx.y) * 262144) + (((i_7 & 15) >> 2) * 65536)) + (((((int)threadIdx.x) & 63) >> 5) * 32768)) + (((i_7 & 3) >> 1) * 16384)) + (((((int)threadIdx.x) & 31) >> 2) * 2048)) + (((int)blockIdx.x) * 128)) + ((i_7 >> 4) * 16)) + ((((int)threadIdx.x) >> 6) * 8)) + ((i_7 & 1) * 4)) + (((int)threadIdx.x) & 3))] = ((half_t)C_local[i_7]);
  }
}

extern "C" int call(half_t* __restrict__ A, half_t* __restrict__ B, half_t* __restrict__ C) {
    gemm_kernel<<<dim3(16, 8, 1), dim3(128, 1, 1), 49152>>>(A, B, C);
    return 0;
}