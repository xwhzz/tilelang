import tilelang.testing.benchmark
import example_tilelang_gemm_streamk


def bench_example_tilelang_gemm_streamk():
    tilelang.testing.benchmark.process_func(example_tilelang_gemm_streamk.run_regression_perf)


if globals().get("__name__") == "__main__":
    tilelang.testing.bench()
