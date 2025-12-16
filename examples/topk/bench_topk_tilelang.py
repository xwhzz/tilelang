import tilelang.testing.benchmark
import example_topk


def bench_example_topk():
    tilelang.testing.benchmark.process_func(example_topk.run_regression_perf)


if globals().get("__name__") == "__main__":
    tilelang.testing.bench()
