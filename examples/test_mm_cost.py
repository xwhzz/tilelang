import tilelang
import tilelang.language as T
tilelang.disable_cache()

@tilelang.jit
def matmul(M, N, K, block_M, block_N, block_K, dtype="float16", accum_dtype="float"):

    @T.prim_func
    def matmul_kernel(
            A: T.Tensor((M, K), dtype),
            B: T.Tensor((K, N), dtype),
            C: T.Tensor((M, N), dtype),
    ):
        # Initialize Kernel Context
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_K, block_N), dtype)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)

            # Enable rasterization for better L2 cache locality (Optional)
            # T.use_swizzle(panel_size=10, enable=True)

            # Clear local accumulation
            T.clear(C_local)

            for ko in T.Pipelined(T.ceildiv(K, block_K), num_stages=3):
                # Copy tile of A
                # This is a sugar syntax for parallelized copy
                T.copy(A[by * block_M, ko * block_K], A_shared)

                # Copy tile of B
                T.copy(B[ko * block_K, bx * block_N], B_shared)

                # Perform a tile-level GEMM on the shared buffers
                # Currently we dispatch to the cute/hip on Nvidia/AMD GPUs
                T.gemm(A_shared, B_shared, C_local)

            # Copy result back to global memory
            T.copy(C_local, C[by * block_M, bx * block_N])

    return matmul_kernel


M = 8192
N = 1024
K = 8192
block_M = 128
block_N = 128
block_K = 64

# Define the kernel (matmul) and compile/lower it into an executable module
matmul_kernel = matmul(M, N, K, block_M, block_N, block_K)
# Test the kernel in Python with PyTorch data
import torch

# Create random input tensors on the GPU
a = torch.randn(M, K, device="cuda", dtype=torch.float16)
b = torch.randn(K, N, device="cuda", dtype=torch.float16)
c = torch.empty(M, N, device="cuda", dtype=torch.float16)

# Run the kernel through the Profiler
matmul_kernel(a, b, c)

print(c)

latency = tilelang.profiler.do_bench(lambda: matmul_kernel(a, b, c), backend="cupti")

print(f"tilelang latency: {latency} ms")
