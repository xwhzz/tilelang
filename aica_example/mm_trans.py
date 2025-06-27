# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
import tilelang
import tilelang.language as T
# `make_mma_swizzle_layout` is a python defined layout function
# specifically designed for MMA operations
# which ensures the consistency with the nvidia CUTLASS Library.
# to avoid bank conflicts and maximize the performance.
from tilelang.intrinsics import (
    make_mma_swizzle_layout as make_swizzle_layout,)  # noqa: F401


def matmul(M, N, K, block_M, block_N, block_K, dtype="float16", accum_dtype="float"):

        @T.prim_func
        def main(
                A: T.Tensor((M, K), dtype),
                B: T.Tensor((N, K), dtype),
                C: T.Tensor((M, N), dtype),
        ):

            with T.Kernel(
                    T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):

                # Allocate shared memory for A sub-block of shape (block_M, block_K)
                A_shared = T.alloc_shared((block_M, block_K), dtype)
                # Allocate shared memory for B sub-block of shape (block_N, block_K)
                B_shared = T.alloc_shared((block_N, block_K), dtype)
                # Allocate a local fragment for intermediate accumulation
                C_local = T.alloc_fragment((block_M, block_N), accum_dtype)
                # Allocate a shared memory for C sub-block of shape (block_M, block_N)
                C_shared = T.alloc_shared((block_M, block_N), dtype)


                # Clear out the accumulation buffer
                T.clear(C_local)

                # Loop over sub-blocks in K dimension, pipelined by num_stages
                for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=2):
                    # Load a sub-block of A from global memory into A_shared
                    T.copy(A[by * block_M, k * block_K], A_shared)
                    # Load a sub-block of B from global memory into B_shared
                    T.copy(B[bx * block_N, k * block_K], B_shared)
                    # Perform a partial matrix multiplication:
                    #   C_local += A_shared @ B_shared^T
                    T.gemm(
                        A_shared,
                        B_shared,
                        C_local,
                        transpose_B=True,
                    )
                # Write back the results from C_local to the global memory C
                T.copy(C_local, C_shared)
                T.copy(C_shared, C[by * block_M, bx * block_N])

        return main


M = 1024  # M = T.symbolic("m") if you want to use dynamic shape
N = 1024
K = 1024
block_M = 128
block_N = 128
block_K = 32

# 1. Define the kernel (matmul) and compile/lower it into an executable module
func = matmul(M, N, K, block_M, block_N, block_K)

# 2. Compile the kernel into a torch function
# out_idx specifies the index of the output buffer in the argument list
# if out_idx is specified, the tensor will be created during runtime
# target currently can be "cuda" or "hip" or "cpu".
jit_kernel = tilelang.compile(func, out_idx=[2], target="cuda", execution_backend="cython")
# jit_kernel = tilelang.compile(func, out_idx=[2], target="cuda", execution_backend="dlpack")

print(jit_kernel.get_kernel_source())
