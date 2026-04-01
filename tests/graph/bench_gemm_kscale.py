"""K-dimension scaling to isolate compute vs epilogue overhead."""

import tilelang
import tilelang.language as T
import torch

M, N = 512, 4096


def bench_ms(fn, w=20, r=200):
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    for _ in range(w):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(r):
        torch.cuda.synchronize()
        s.record()
        fn()
        e.record()
        torch.cuda.synchronize()
        ts.append(s.elapsed_time(e))
    return min(ts)


def make_gemm(K_val):
    @T.prim_func
    def gemm(
        A: T.Buffer((M, K_val), "float16"),
        B: T.Buffer((N, K_val), "float16"),
        C: T.Buffer((M, N), "float16"),
    ):
        with T.Kernel(T.ceildiv(N, 128), T.ceildiv(M, 128), threads=256) as (bx, by):
            A_sh = T.alloc_shared((128, 64), "float16")
            B_sh = T.alloc_shared((128, 64), "float16")
            C_frag = T.alloc_fragment((128, 128), "float32")
            T.use_swizzle(panel_size=10)
            T.clear(C_frag)
            for k in T.Pipelined(T.ceildiv(K_val, 64), num_stages=4):
                T.copy(A[by * 128, k * 64], A_sh)
                T.copy(B[bx * 128, k * 64], B_sh)
                T.gemm(A_sh, B_sh, C_frag, transpose_B=True)
            T.copy(C_frag, C[by * 128, bx * 128])

    return gemm


def main():
    print("K-dimension scaling (M=512, N=4096):")
    print(f"{'K':>6s} {'cuBLAS':>8s} {'TileLang':>10s} {'ratio':>7s} {'cuBLAS_T':>9s} {'TL_T':>9s}")
    print("-" * 55)
    for K in [64, 128, 256, 512, 1024, 2048, 4096, 8192]:
        A = torch.randn(M, K, device="cuda", dtype=torch.float16)
        B = torch.randn(N, K, device="cuda", dtype=torch.float16)
        cublas = bench_ms(lambda: A @ B.T)
        func = make_gemm(K)
        kernel = tilelang.compile(func, out_idx=[-1], target="auto")
        _ = kernel(A, B)
        tl = bench_ms(lambda: kernel(A, B))
        flops = 2 * M * N * K
        tl_t = flops / (tl * 1e-3) / 1e12
        cb_t = flops / (cublas * 1e-3) / 1e12
        print(f"{K:>6d} {cublas:>7.3f}ms {tl:>9.3f}ms {tl/cublas:>6.2f}x {cb_t:>8.0f}T {tl_t:>8.0f}T")


if __name__ == "__main__":
    main()
