#pragma once

#if __CUDA_ARCH_LIST__ >= 890
#include "./cuda_fp8.h"
#endif

#include "common.h"
#ifndef __CUDACC_RTC__
#include <cstdint>
#include <cstdio>
#endif

template <typename T> struct PrintTraits {
  static __device__ const char *type_name() { return "unknown"; }
  static __device__ const char *fmt() { return "%p"; }
  static __device__ const void *cast(T val) { return (const void *)&val; }
};

#define DEFINE_PRINT_TRAIT(TYPE, NAME, FORMAT, CAST_TYPE)                      \
  template <> struct PrintTraits<TYPE> {                                       \
    static __device__ const char *type_name() { return NAME; }                 \
    static __device__ const char *fmt() { return FORMAT; }                     \
    static __device__ CAST_TYPE cast(TYPE val) { return (CAST_TYPE)val; }      \
  }

DEFINE_PRINT_TRAIT(char, "char", "%d", int);
DEFINE_PRINT_TRAIT(signed char, "signed char", "%d", int);
DEFINE_PRINT_TRAIT(unsigned char, "unsigned char", "%u", unsigned int);
DEFINE_PRINT_TRAIT(short, "short", "%d", int);
DEFINE_PRINT_TRAIT(unsigned short, "unsigned short", "%u", unsigned int);
DEFINE_PRINT_TRAIT(int, "int", "%d", int);
DEFINE_PRINT_TRAIT(unsigned int, "uint", "%u", unsigned int);
DEFINE_PRINT_TRAIT(long, "long", "%ld", long);
DEFINE_PRINT_TRAIT(unsigned long, "ulong", "%lu", unsigned long);
DEFINE_PRINT_TRAIT(long long, "long long", "%lld", long long);

DEFINE_PRINT_TRAIT(float, "float", "%f", float);
DEFINE_PRINT_TRAIT(double, "double", "%lf", double);
DEFINE_PRINT_TRAIT(half, "half", "%f", float);
DEFINE_PRINT_TRAIT(half_t, "half_t", "%f", float);
DEFINE_PRINT_TRAIT(bfloat16_t, "bfloat16_t", "%f", float);

#if __CUDA_ARCH_LIST__ >= 890
DEFINE_PRINT_TRAIT(fp8_e4_t, "fp8_e4_t", "%f", float);
DEFINE_PRINT_TRAIT(fp8_e5_t, "fp8_e5_t", "%f", float);
#endif

template <> struct PrintTraits<bool> {
  static __device__ const char *type_name() { return "bool"; }
  static __device__ const char *fmt() { return "%s"; }
  static __device__ const char *cast(bool val) {
    return val ? "true" : "false";
  }
};

template <typename T> struct PrintTraits<T *> {
  static __device__ const char *type_name() { return "pointer"; }
  static __device__ const char *fmt() { return "%p"; }
  static __device__ const void *cast(T *val) { return (const void *)val; }
};

__device__ inline void build_fmt(char *dst, const char *part1,
                                 const char *part2, const char *part3) {
  int i = 0;
  for (int j = 0; part1[j] != '\0'; ++j)
    dst[i++] = part1[j];
  for (int j = 0; part2[j] != '\0'; ++j)
    dst[i++] = part2[j];
  for (int j = 0; part3[j] != '\0'; ++j)
    dst[i++] = part3[j];
  dst[i] = '\0';
}

template <typename T> __device__ void debug_print_var(const char *msg, T var) {
  using Traits = PrintTraits<T>;

  const char *prefix =
      "msg='%s' BlockIdx=(%d, %d, %d), ThreadIdx=(%d, %d, %d): dtype=%s value=";
  const char *suffix = "\n";

  char fmt_buffer[256];
  build_fmt(fmt_buffer, prefix, Traits::fmt(), suffix);

  printf(fmt_buffer, msg, blockIdx.x, blockIdx.y, blockIdx.z, threadIdx.x,
         threadIdx.y, threadIdx.z, Traits::type_name(), Traits::cast(var));
}

template <typename T>
__device__ void debug_print_buffer_value(const char *msg, const char *buf_name,
                                         int index, T var) {
  using Traits = PrintTraits<T>;

  const char *prefix = "msg='%s' BlockIdx=(%d, %d, %d), ThreadIdx=(%d, %d, "
                       "%d): buffer=%s, index=%d, dtype=%s value=";
  const char *suffix = "\n";

  char fmt_buffer[256];
  build_fmt(fmt_buffer, prefix, Traits::fmt(), suffix);

  printf(fmt_buffer, msg, blockIdx.x, blockIdx.y, blockIdx.z, threadIdx.x,
         threadIdx.y, threadIdx.z, buf_name, index, Traits::type_name(),
         Traits::cast(var));
}

TL_DEVICE void device_assert(bool cond) { assert(cond); }

TL_DEVICE void device_assert_with_msg(bool cond, const char *msg) {
  if (!cond) {
    printf("Device assert failed: %s\n", msg);
    assert(0);
  }
}