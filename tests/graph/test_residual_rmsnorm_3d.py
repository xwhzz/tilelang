"""Test 3D residual_rmsnorm kernel standalone."""
import torch
import tilelang
import tilelang.language as T
from tilelang.profiler import do_bench

B, S, N = 1, 512, 4096

@T.prim_func
def kernel_3d(
    hidden: T.Tensor((B, S, N), "float16"),
    residual: T.Tensor((B, S, N), "float16"),
    w: T.Tensor((N,), "float16"),
    out: T.Tensor((B, S, N), "float16"),
):
    with T.Kernel(S, threads=128) as bx:
        h_frag = T.alloc_fragment((N,), "float16")
        r_frag = T.alloc_fragment((N,), "float16")
        x_sq = T.alloc_fragment((N,), "float32")
        sq_sum = T.alloc_fragment((1,), "float32")
        T.copy(hidden[0, bx, 0:N], h_frag)
        T.copy(residual[0, bx, 0:N], r_frag)
        for j in T.Parallel(N):
            val = T.cast(h_frag[j], "float32") + T.cast(r_frag[j], "float32")
            h_frag[j] = T.cast(val, "float16")
            x_sq[j] = val * val
        T.reduce_sum(x_sq, sq_sum, dim=0)
        for i in T.Parallel(1):
            sq_sum[i] = T.rsqrt(sq_sum[i] / N + 1e-6)
        for j in T.Parallel(N):
            h_frag[j] = T.cast(T.cast(h_frag[j], "float32") * sq_sum[0], "float16") * w[j]
        T.copy(h_frag, out[0, bx, 0:N])

if __name__ == "__main__":
    func = kernel_3d.with_attr("tir.is_scheduled", True).with_attr("tir.is_tilelang_kernel", True)
    kernel = tilelang.compile(func, target="auto", out_idx=[-1])

    h = torch.randn(B, S, N, device="cuda", dtype=torch.float16)
    r = torch.randn(B, S, N, device="cuda", dtype=torch.float16)
    wt = torch.randn(N, device="cuda", dtype=torch.float16)
    out_tl = kernel(h, r, wt)

    x = h.float() + r.float()
    ref = (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6) * wt.float()).half()
    err = (out_tl - ref).abs().max().item()
    print(f"3D kernel (B={B}, S={S}, N={N}): max_err={err:.6f}")
    t = do_bench(lambda: kernel(h, r, wt))
    print(f"Latency: {t:.4f} ms ({t*1000:.1f} us)")
