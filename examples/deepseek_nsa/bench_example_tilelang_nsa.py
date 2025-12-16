import tilelang.testing.benchmark
import example_tilelang_nsa_fwd
import example_tilelang_nsa_decode


def bench_example_tilelang_nsa_fwd():
    tilelang.testing.benchmark.process_func(example_tilelang_nsa_fwd.run_regression_perf)


def bench_example_tilelang_nsa_fwd_decode():
    tilelang.testing.benchmark.process_func(example_tilelang_nsa_decode.run_regression_perf)


if globals().get("__name__") == "__main__":
    tilelang.testing.bench()
