# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
import argparse
import torch
import itertools
import tilelang as tl
import tilelang.language as T
from tilelang.autotuner import AutoTuner
from tilelang.carver.template import MatmulTemplate
from tilelang.carver.arch import CUDA
from tilelang.carver.roller.rasterization import NoRasterization


def ref_program(A, B):
    return A @ B.T


def get_configs(M, N, K, with_roller=False, topk=20):
    if with_roller:
        arch = CUDA("cuda")
        carve_template = MatmulTemplate(
            M=M,
            N=N,
            K=K,
            in_dtype="float16",
            out_dtype="float16",
            accum_dtype="float",
        ).with_arch(arch)

        func = carve_template.equivalent_function()
        assert func is not None, "Function is None"
        roller_hints = carve_template.recommend_hints(topk=topk)
        if roller_hints is None:
            raise ValueError("No Roller Hints Found for TensorCore Scheduling")
        configs = []
        for hint in roller_hints:
            config = {}
            block_m, block_n = hint.block
            warp_m, warp_n = hint.warp
            # block_rows, block_cols represents warp partitioning
            block_rows, block_cols = block_m // warp_m, block_n // warp_n
            config["block_M"] = block_m
            config["block_N"] = block_n
            config["block_K"] = hint.rstep[0]
            config["num_stages"] = hint.pipeline_stage if hint.pipeline_stage > 1 else 0
            config["thread_num"] = block_rows * block_cols * 32
            config["enable_rasteration"] = hint.rasterization_plan is not NoRasterization
            configs.append(config)
        for config in configs:
            print(config)
    else:
        block_M = [64, 128, 256]
        block_N = [64, 128, 256]
        block_K = [32, 64]
        num_stages = [0, 1, 2, 3]
        thread_num = [128, 256]
        enable_rasterization = [True, False]
        _configs = list(
            itertools.product(
                block_M,
                block_N,
                block_K,
                num_stages,
                thread_num,
                enable_rasterization,
            ))

        configs = [
            {
                "block_M": c[0],
                "block_N": c[1],
                "block_K": c[2],
                "num_stages": c[3],
                "thread_num": c[4],
                "enable_rasteration": c[5],  # keep param name for backward-compat
            } for c in _configs
        ]
    return configs


def get_best_config(M, N, K, with_roller=False):

    def kernel(
        block_M=None,
        block_N=None,
        block_K=None,
        num_stages=None,
        thread_num=None,
        enable_rasteration=None,
    ):
        dtype = "float16"
        accum_dtype = "float"

        @T.prim_func
        def main(
                A: T.Tensor((M, K), dtype),
                B: T.Tensor((N, K), dtype),
                C: T.Tensor((M, N), dtype),
        ):
            with T.Kernel(
                    T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=thread_num) as (bx, by):
                A_shared = T.alloc_shared((block_M, block_K), dtype)
                B_shared = T.alloc_shared((block_N, block_K), dtype)
                C_local = T.alloc_fragment((block_M, block_N), accum_dtype)
                C_shared = T.alloc_shared((block_M, block_N), dtype)
                T.use_swizzle(panel_size=10, enable=enable_rasteration)
                T.clear(C_local)
                for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=num_stages):
                    T.copy(A[by * block_M, k * block_K], A_shared)
                    T.copy(B[bx * block_N, k * block_K], B_shared)
                    T.gemm(
                        A_shared,
                        B_shared,
                        C_local,
                        transpose_B=True,
                    )
                T.copy(C_local, C_shared)
                T.copy(C_shared, C[by * block_M, bx * block_N])

        return main

    autotuner = AutoTuner.from_kernel(
        kernel=kernel, configs=get_configs(M, N, K, with_roller)).set_compile_args(
            out_idx=[-1],
            supply_type=tl.TensorSupplyType.Integer,
            ref_prog=ref_program,
            skip_check=False,
            target="auto",
        )
    return autotuner.run(warmup=3, rep=20)


def get_heuristic_config() -> dict:
    # Get CUDA device properties
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    device = torch.cuda.current_device()
    sm_major, sm_minor = torch.cuda.get_device_capability(device)
    sm_version = sm_major * 10 + sm_minor
    print(f"CUDA device capability: {sm_version}")
    if sm_version in {80}:
        return {
            "block_M": 128,
            "block_N": 256,
            "block_K": 32,
            "num_stages": 2,
            "thread_num": 128,
            "enable_rasteration": True
        }
    elif sm_version in {90}:
        return {
            "block_M": 128,
            "block_N": 256,
            "block_K": 64,
            "num_stages": 3,
            "thread_num": 256,
            "enable_rasteration": True
        }
    else:
        return {
            "block_M": 128,
            "block_N": 256,
            "block_K": 32,
            "num_stages": 0,
            "thread_num": 128,
            "enable_rasteration": True
        }


def matmul(M,
           N,
           K,
           block_M,
           block_N,
           block_K,
           num_stages,
           thread_num,
           enable_rasteration,
           dtype="float16",
           accum_dtype="float"):

    @T.prim_func
    def main(
            A: T.Tensor((M, K), dtype),
            B: T.Tensor((N, K), dtype),
            C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=thread_num) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_N, block_K), dtype)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)
            C_shared = T.alloc_shared((block_M, block_N), dtype)
            T.use_swizzle(panel_size=10, enable=enable_rasteration)
            T.clear(C_local)
            for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=num_stages):
                T.copy(A[by * block_M, k * block_K], A_shared)
                T.copy(B[bx * block_N, k * block_K], B_shared)
                T.gemm(
                    A_shared,
                    B_shared,
                    C_local,
                    transpose_B=True,
                )
            T.copy(C_local, C_shared)
            T.copy(C_shared, C[by * block_M, bx * block_N])

    return main


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Autotuned MatMul Benchmark")
    parser.add_argument("--m", type=int, default=16384, help="Matrix dimension M")
    parser.add_argument("--n", type=int, default=16384, help="Matrix dimension N")
    parser.add_argument("--k", type=int, default=16384, help="Matrix dimension K")
    parser.add_argument(
        "--use_autotune",
        action="store_true",
        default=False,
        help="Whether to use autotune for matmul configs")
    parser.add_argument(
        "--with_roller",
        action="store_true",
        default=True,
        help="Whether to enable BitBLAS roller for search space")
    args = parser.parse_args()
    M, N, K = args.m, args.n, args.k
    a = torch.randn(M, K).cuda().half()
    b = torch.randn(N, K).cuda().half()
    use_autotune = args.use_autotune
    use_autotune = True
    with_roller = args.with_roller
    if use_autotune:
        result = get_best_config(M, N, K, with_roller)
        print(result.config)
        kernel = result.kernel
    else:
        config = get_heuristic_config()
        kernel = tl.compile(matmul(M, N, K, **config), out_idx=-1)

    # benchmark
    profiler = kernel.get_profiler(tensor_supply_type=tl.TensorSupplyType.Auto)
    tilelang_latency = profiler.do_bench()
    ref_latency = profiler.do_bench(ref_program)
    profiler.assert_allclose(ref_program, atol=1e-2, rtol=1e-2)
    print(f"TileLang latency: {tilelang_latency}")
    print(f"Ref latency: {ref_latency}")
    print(f"TileLang TFlops: {2 * M * N * K / tilelang_latency * 1e-9}")
    print(f"Ref TFlops: {2 * M * N * K / ref_latency * 1e-9}")
