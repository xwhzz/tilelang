// Copyright (c) Tile-AI Corporation.
// Licensed under the MIT License.
#pragma once

#include "common.h"
#include <cutlass/arch/memory_sm80.h>

// TODO(xwh): only support SM80 for aica
namespace tl {

using namespace cutlass;

TL_DEVICE void cp_async_commit() {
  arch::cp_async_fence();
}

template <int N> TL_DEVICE void cp_async_wait() {
  arch::cp_async_wait<N>();
}

template <int N>
TL_DEVICE void cp_async_gs(void *smem_addr, void const *global_ptr) {
  unsigned int addr = smem_ptr_to_uint(smem_addr);
  #pragma unroll
  for (int i = 0; i < N; i+=4){
    arch::cp_async<4, arch::CacheOperation::Always>(reinterpret_cast<void *>(addr + i),
                                                    (const char *)global_ptr + i);
  }
}

template <int N>
TL_DEVICE void cp_async_gs_conditional(void *smem_addr, void const *global_ptr, bool cond) {
  unsigned int addr = smem_ptr_to_uint(smem_addr);
  #pragma unroll
  for (int i = 0; i < N; i+=4){
    arch::cp_async_zfill<4, arch::CacheOperation::Always>(reinterpret_cast<void *>(addr + i),
                                                    (const char *)global_ptr + i, cond);
  }
}

} // namespace tl