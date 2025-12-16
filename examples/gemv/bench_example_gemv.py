import tilelang.testing.benchmark
import example_gemv


def bench_example_gemv():
    tilelang.testing.benchmark.process_func(example_gemv.run_regression_perf)


if globals().get("__name__") == "__main__":
    tilelang.testing.bench()
