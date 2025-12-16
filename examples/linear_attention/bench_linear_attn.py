import tilelang.testing.benchmark
import example_linear_attn_bwd
import example_linear_attn_fwd


def bench_example_linear_attn_fwd():
    tilelang.testing.benchmark.process_func(example_linear_attn_fwd.run_regression_perf)


def bench_example_linear_attn_bwd():
    tilelang.testing.benchmark.process_func(example_linear_attn_bwd.run_regression_perf)


if globals().get("__name__") == "__main__":
    tilelang.testing.bench()
