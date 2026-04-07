"""Benchmark: te.compute RoPE vs TileLang DSL RoPE with T.copy."""
import torch
import tilelang
import tilelang.language as T
from tilelang.profiler import do_bench

B, H, S, D = 1, 32, 512, 128
half = D // 2
# Input: (B, S, H*D) — the linear output before reshape+permute
HD = H * D


@tilelang.jit(out_idx=[-1])
def rope_dsl(B_, H_, S_, D_):
    half_ = D_ // 2

    @T.prim_func
    def main(
        x: T.Tensor((B_, S_, H_ * D_), "float16"),
        cos: T.Tensor((1, 1, S_, D_), "float16"),
        sin: T.Tensor((1, 1, S_, D_), "float16"),
        out: T.Tensor((B_, H_, S_, D_), "float16"),
    ):
        with T.Kernel(H_ * S_, threads=128) as bx:
            h = bx // S_
            s = bx % S_
            x_frag = T.alloc_fragment((D_,), "float16")
            cos_frag = T.alloc_fragment((D_,), "float16")
            sin_frag = T.alloc_fragment((D_,), "float16")

            # Vectorized load: reads contiguous D elements from x
            T.copy(x[0, s, h * D_:h * D_ + D_], x_frag)
            T.copy(cos[0, 0, s, 0:D_], cos_frag)
            T.copy(sin[0, 0, s, 0:D_], sin_frag)

            for d in T.Parallel(D_):
                val = T.cast(x_frag[d], "float32")
                paired_d = (d + half_) % D_
                # Read paired element from global memory (cross-thread fragment access not supported)
                paired_val = T.cast(x[0, s, h * D_ + paired_d], "float32")
                sign = T.float32(1) - T.float32(2) * T.cast(d < half_, "float32")
                c = T.cast(cos_frag[d], "float32")
                sn = T.cast(sin_frag[d], "float32")
                x_frag[d] = T.cast(val * c + sign * paired_val * sn, "float16")

            T.copy(x_frag, out[0, h, s, 0:D_])

    return main


if __name__ == "__main__":
    x = torch.randn(B, S, HD, device="cuda", dtype=torch.float16)
    cos = torch.randn(1, 1, S, D, device="cuda", dtype=torch.float16)
    sin = torch.randn(1, 1, S, D, device="cuda", dtype=torch.float16)

    out = rope_dsl(B, H, S, D)(x, cos, sin)

    # Reference
    def ref_rope(x, cos, sin):
        xr = x.reshape(B, S, H, D).permute(0, 2, 1, 3).float()
        c = cos.float()
        s = sin.float()
        x1 = xr[..., :half]
        x2 = xr[..., half:]
        rotated = torch.cat([-x2, x1], dim=-1)
        return (xr * c + rotated * s).half()

    ref = ref_rope(x, cos, sin)
    err = (out - ref).abs().max().item()
    print(f"Correctness: max_err={err:.6f}")
    assert err < 0.02

    t_dsl = do_bench(lambda: rope_dsl(B, H, S, D)(x, cos, sin))
    print(f"DSL RoPE:     {t_dsl:.4f} ms ({t_dsl/64*1000:.1f} us/call equiv)")

    # Compare: 64 calls (Q+K × 32 layers)
    print(f"64-call est:  {t_dsl * 64:.3f} ms")
