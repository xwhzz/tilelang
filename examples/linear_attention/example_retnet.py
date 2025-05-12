import argparse
import torch
import tilelang
import tilelang.language as T


def retnet(batch, heads, seq_len, dim_qk, dim_v, block_M, block_N):
    qk_shape = [batch, seq_len, heads, dim_qk]
    v_shape = [batch, seq_len, heads, dim_v]
    dtype = "float16"
    accum_dtype = "float"

    @T.prim_func
    def main(
            Q: T.Tensor(qk_shape, dtype),
            K: T.Tensor(qk_shape, dtype),
            V: T.Tensor(v_shape, dtype),
            mask: T.Tensor([heads, seq_len, seq_len], dtype),
            Output: T.Tensor(v_shape, dtype),
    ):
        with T.Kernel(T.ceildiv(seq_len, block_M), heads, batch, threads=128 * 2) as (bx, by, bz):
            Q_shared = T.alloc_shared([block_M, dim_qk], dtype)
            K_shared = T.alloc_shared([block_N, dim_qk], dtype)
            V_shared = T.alloc_shared([block_N, dim_v], dtype)
            mask_shared = T.alloc_shared([block_M, block_N], dtype)
            acc_o_shared = T.alloc_shared([block_M, dim_v], dtype)
            mask_local = T.alloc_fragment([block_M, block_N], dtype)
            acc_s = T.alloc_fragment([block_M, block_N], accum_dtype)
            acc_s_1 = T.alloc_fragment([block_M, block_N], accum_dtype)
            acc_s_shared = T.alloc_shared([block_M, block_N], dtype)
            acc_s_cast = T.alloc_fragment([block_M, block_N], dtype)
            acc_o = T.alloc_fragment([block_M, dim_v], accum_dtype)
            abs_sum = T.alloc_fragment([block_M], accum_dtype)
            r_wo_clamp = T.alloc_fragment([block_M], accum_dtype)
            r = T.alloc_fragment([block_M], accum_dtype)
            r_new = T.alloc_fragment([block_M], accum_dtype)

            T.annotate_layout({
                Q_shared: tilelang.layout.make_swizzled_layout(Q_shared),
                mask_shared: tilelang.layout.make_swizzled_layout(mask_shared),
                acc_s_shared: tilelang.layout.make_swizzled_layout(acc_s_shared),
                acc_o_shared: tilelang.layout.make_swizzled_layout(acc_o_shared)
            })

            T.copy(Q[bz, bx * block_M:(bx + 1) * block_M, by, :], Q_shared)

            T.fill(r, 0)
            T.fill(r_new, 0)
            T.fill(r_wo_clamp, 0)
            T.fill(acc_o, 0)
            loop_range = T.ceildiv(seq_len, block_N)
            for k in T.Pipelined(loop_range, num_stages=1):
                T.copy(K[bz, k * block_N:(k + 1) * block_N, by, :], K_shared)
                T.clear(acc_s)
                T.gemm(Q_shared, K_shared, acc_s, transpose_B=True, policy=T.GemmWarpPolicy.FullCol)
                T.copy(mask[by, bx * block_M:(bx + 1) * block_M, k * block_N:(k + 1) * block_N],
                       mask_shared)
                T.copy(mask_shared, mask_local)
                for i, j in T.Parallel(block_M, block_N):
                    acc_s[i, j] = acc_s[i, j] * mask_local[i, j]
                T.copy(acc_s, acc_s_shared)
                T.copy(acc_s_shared, acc_s_1)
                T.reduce_abssum(acc_s_1, abs_sum, dim=1)
                for i in T.Parallel(block_M):
                    r_wo_clamp[i] = r_wo_clamp[i] + abs_sum[i]
                for i in T.Parallel(block_M):
                    r_new[i] = T.max(r_wo_clamp[i], 1)
                for i, j in T.Parallel(block_M, dim_v):
                    acc_o[i, j] = T.if_then_else(k > 0, acc_o[i, j] * r[i] / r_new[i], acc_o[i, j])
                T.copy(r_new, r)
                for i, j in T.Parallel(block_M, block_N):
                    acc_s_1[i, j] = acc_s_1[i, j] / r_new[i]
                T.copy(acc_s_1, acc_s_cast)
                T.copy(V[bz, k * block_N:(k + 1) * block_N, by, :], V_shared)
                T.gemm(acc_s_cast, V_shared, acc_o, policy=T.GemmWarpPolicy.FullCol)
            T.copy(acc_o, acc_o_shared)
            T.copy(acc_o_shared, Output[bz, bx * block_M:(bx + 1) * block_M, by, :])

    return main


def ref_program(Q, K, V, mask):
    qk = torch.einsum('bqhd,bkhd->bhqk', Q, K)
    qkm = qk * mask
    r = qkm.detach().abs().sum(dim=-1, keepdim=True).clamp(min=1.0)
    o = torch.einsum('bhqk,bkhd->bqhd', qkm / r, V)
    return o.to(dtype=torch.float16)


def ref_inference(Q, K, V, prev_kv, prev_scale, decay):
    # Q : batch, seqlen, num_heads, head_dimqk
    # K : batch, seqlen, num_heads, head_dimqk
    # V : batch, seqlen, num_heads, head_dimv
    # prev_kv : batch, num_heads, head_dimv, head_dimqk
    # prev_scale : num_heads, 1, 1
    # decay : num_heads, 1, 1
    seqlen = V.size(1)
    num_heads = V.size(2)
    assert seqlen == 1, "Only support seqlen == 1"

    qr = Q.transpose(1, 2).contiguous()  # batch, num_heads, 1, head_dimqk
    kr = K.transpose(1, 2).contiguous()  # batch, num_heads, 1, head_dimqk
    v = V.transpose(1, 2).transpose(2, 3).contiguous()  # batch, num_heads, head_dimv, 1

    kv = kr * v  # batch, num_heads, head_dimv, head_dimqk
    scale = prev_scale * decay + 1  # num_heads, 1, 1
    kv = prev_kv * (prev_scale.sqrt() * decay / scale.sqrt()).view(
        num_heads, 1, 1) + kv / scale.sqrt().view(num_heads, 1, 1)
    output = torch.sum(qr * kv, dim=3)
    return output


def retnet_inference(batch, heads, dim_qk, dim_v, block_M):
    qk_shape = [batch, 1, heads, dim_qk]
    v_shape = [batch, 1, heads, dim_v]
    dtype = "float16"
    accum_dtype = "float"

    @T.prim_func
    def main(
            Q: T.Tensor(qk_shape, dtype),
            K: T.Tensor(qk_shape, dtype),
            V: T.Tensor(v_shape, dtype),
            prev_kv: T.Tensor([batch, heads, dim_v, dim_qk], dtype),
            prev_scale: T.Tensor([heads], dtype),
            decay: T.Tensor([heads], dtype),
            Output: T.Tensor([batch, heads, dim_v], dtype),
    ):
        with T.Kernel(T.ceildiv(dim_v, block_M), heads, batch, threads=128) as (bx, by, bz):
            Q_local = T.alloc_fragment([1, dim_qk], dtype)
            K_local = T.alloc_fragment([dim_qk], dtype)
            V_local = T.alloc_fragment([block_M], dtype)
            kv_local = T.alloc_fragment([block_M, dim_qk], accum_dtype)
            prev_kv_local = T.alloc_fragment([block_M, dim_qk], dtype)
            prev_scale_local = T.alloc_fragment([1], dtype)
            decay_local = T.alloc_fragment([1], accum_dtype)
            # scale_local = T.alloc_fragment([1], accum_dtype)
            qkv_local = T.alloc_fragment([block_M, dim_qk], accum_dtype)
            o_local = T.alloc_fragment([block_M], accum_dtype)

            T.annotate_layout({
                prev_scale_local: T.Layout(prev_scale_local.shape, lambda i: i),
                decay_local: T.Layout(decay_local.shape, lambda i: i),
                # scale_local: T.Layout(scale_local.shape, lambda i : i),
                kv_local: T.Fragment(kv_local.shape, lambda i, j: j // 8),
            })

            T.copy(Q[bz, 0, by, :], Q_local)
            T.copy(K[bz, 0, by, :], K_local)
            T.copy(V[bz, 0, by, bx * block_M:(bx + 1) * block_M], V_local)
            T.copy(prev_kv[bz, by, bx * block_M:(bx + 1) * block_M, :], prev_kv_local)
            prev_scale_local[0] = prev_scale[by]
            decay_local[0] = decay[by]
            for i, j in T.Parallel(block_M, dim_qk):
                kv_local[i, j] = K_local[j] * V_local[i]
            for i, j in T.Parallel(block_M, dim_qk):
                kv_local[i, j] += kv_local[i, j]
            for i, j in T.Parallel(block_M, dim_qk):
                kv_local[i, j] += prev_kv_local[i, j] * T.sqrt(prev_scale[by]) * decay[by]
            for i, j in T.Parallel(block_M, dim_qk):
                kv_local[i, j] = kv_local[i, j] / T.sqrt(prev_scale[by] * decay[by] + 1)
            for i, j in T.Parallel(block_M, dim_qk):
                qkv_local[i, j] = Q_local[0, j] * kv_local[i, j]
            T.reduce_sum(qkv_local, o_local, dim=1)
            T.copy(o_local, Output[bz, by, bx * block_M:(bx + 1) * block_M])

    return main


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch', type=int, default=1, help='Batch size')
    parser.add_argument('--h', type=int, default=10, help='Number of heads')
    parser.add_argument('--n_ctx', type=int, default=4096, help='Context size')
    parser.add_argument('--dim_qk', type=int, default=256, help='Head dimension')
    parser.add_argument('--dim_v', type=int, default=448, help='Head dimension')
    args = parser.parse_args()
    BATCH, H, N_CTX, dim_qk, dim_v = args.batch, args.h, args.n_ctx, args.dim_qk, args.dim_v
    total_flops = 2.0 * BATCH * H * N_CTX * N_CTX * (dim_qk + dim_v)
    BLOCK_M = 64
    BLOCK_N = 64
    program = retnet(BATCH, H, N_CTX, dim_qk, dim_v, BLOCK_M, BLOCK_N)
    kernel = tilelang.compile(program, out_idx=[4])
    profiler = kernel.get_profiler(tilelang.TensorSupplyType.Normal)

    ins = profiler._get_inputs()

    ref_outs = ref_program(*ins)
    lib_outs = kernel(*ins)

    profiler.assert_allclose(ref_program, rtol=0.01, atol=0.01)
    latency = profiler.do_bench(n_warmup=10, n_repeat=10, profiler="torch")
    print("tilelang: {:.2f} ms".format(latency))
    print("tilelang: {:.2f} TFlops".format(total_flops / latency * 1e-9))
