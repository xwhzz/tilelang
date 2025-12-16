import tilelang.testing.benchmark
import example_tilelang_gemm_fp8
import example_tilelang_gemm_fp8_2xAcc
import example_tilelang_gemm_fp8_intrinsic


def bench_example_tilelang_gemm_fp8_2xAcc():
    tilelang.testing.benchmark.process_func(example_tilelang_gemm_fp8_2xAcc.run_regression_perf)


def bench_example_tilelang_gemm_fp8_intrinsic():
    tilelang.testing.benchmark.process_func(example_tilelang_gemm_fp8_intrinsic.run_regression_perf)


def bench_example_tilelang_gemm_fp8():
    tilelang.testing.benchmark.process_func(example_tilelang_gemm_fp8.run_regression_perf)


if globals().get("__name__") == "__main__":
    tilelang.testing.bench()
