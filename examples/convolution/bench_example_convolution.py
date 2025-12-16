import tilelang.testing.benchmark
import example_convolution
import example_convolution_autotune


def bench_example_convolution():
    tilelang.testing.benchmark.process_func(example_convolution.run_regression_perf)


def bench_example_convolution_autotune():
    tilelang.testing.benchmark.process_func(example_convolution_autotune.run_regression_perf)


if globals().get("__name__") == "__main__":
    tilelang.testing.bench()
