import tilelang.testing.benchmark
import example_tilelang_gemm_splitk
import example_tilelang_gemm_splitk_vectorize_atomicadd


def bench_example_tilelang_gemm_splitk():
    tilelang.testing.benchmark.process_func(example_tilelang_gemm_splitk.run_regression_perf)


def bench_example_tilelang_gemm_splitk_vectorize_atomicadd():
    tilelang.testing.benchmark.process_func(example_tilelang_gemm_splitk_vectorize_atomicadd.run_regression_perf)


if globals().get("__name__") == "__main__":
    tilelang.testing.bench()
