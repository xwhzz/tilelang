"""Standalone residual add + RMSNorm kernel: TileLang DSL."""

import torch
import tilelang
import tilelang.language as T
from tilelang.profiler import do_bench


@tilelang.jit(out_idx=[-1])
def residual_rmsnorm_tl(M, N):
    @T.prim_func
    def main(
        hidden: T.Tensor((M, N), "float16"),
        residual: T.Tensor((M, N), "float16"),
        w: T.Tensor((N,), "float16"),
        out: T.Tensor((M, N), "float16"),
    ):
        with T.Kernel(M, threads=128) as bx:
            h_frag = T.alloc_fragment((N,), "float16")
            r_frag = T.alloc_fragment((N,), "float16")
            x_sq = T.alloc_fragment((N,), "float32")
            sq_sum = T.alloc_fragment((1,), "float32")

            # Load both rows into registers
            T.copy(hidden[bx, 0:N], h_frag)
            T.copy(residual[bx, 0:N], r_frag)

            # Add + square in fp32
            for j in T.Parallel(N):
                val = T.cast(h_frag[j], "float32") + T.cast(r_frag[j], "float32")
                h_frag[j] = T.cast(val, "float16")  # reuse h_frag for the sum
                x_sq[j] = val * val

            # Reduce sum of squares
            T.reduce_sum(x_sq, sq_sum, dim=0)

            # rsqrt(mean + eps)
            for i in T.Parallel(1):
                sq_sum[i] = T.rsqrt(sq_sum[i] / N + 1e-6)

            # x * scale * w
            for j in T.Parallel(N):
                h_frag[j] = T.cast(
                    T.cast(h_frag[j], "float32") * sq_sum[0], "float16"
                ) * w[j]

            T.copy(h_frag, out[bx, 0:N])

    return main


if __name__ == "__main__":
    M, N = 512, 4096
    h = torch.randn(M, N, device="cuda", dtype=torch.float16)
    r = torch.randn(M, N, device="cuda", dtype=torch.float16)
    wt = torch.randn(N, device="cuda", dtype=torch.float16)

    out_tl = residual_rmsnorm_tl(M, N)(h, r, wt)

    # Reference
    x = h.float() + r.float()
    rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6)
    ref = (x * rms * wt.float()).half()

    err = (out_tl - ref).abs().max().item()
    print(f"Correctness: max_err={err:.6f}")
    assert err < 0.05, f"Error too large: {err}"

    t = do_bench(lambda: residual_rmsnorm_tl(M, N)(h, r, wt))
    print(f"Fused residual+RMSNorm: {t:.4f} ms ({t*1000:.1f} us)")

    # Compare: separate add + RMSNorm
    def sep_fn():
        x = h + r
        xf = x.float()
        rms_v = torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + 1e-6)
        return (xf * rms_v * wt.float()).half()

    t_sep = do_bench(sep_fn)
    print(f"PyTorch separate:       {t_sep:.4f} ms ({t_sep*1000:.1f} us)")
    print(f"Speedup: {t_sep/t:.2f}x")

    # Expected saving in model: 64 calls (32 layers × 2 norms)
    print(f"\nModel saving estimate: {(t_sep - t) * 64:.2f} ms for 64 calls")
