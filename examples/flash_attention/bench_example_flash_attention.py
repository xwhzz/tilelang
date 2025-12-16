import tilelang.testing.benchmark
import example_gqa_fwd_bshd
import example_gqa_fwd_bshd_wgmma_pipelined
import example_mha_fwd_bhsd
import example_mha_fwd_bhsd_wgmma_pipelined
import example_mha_fwd_bshd
import example_mha_fwd_bshd_wgmma_pipelined
import example_mha_fwd_varlen
import example_gqa_bwd_tma_reduce_varlen
import example_gqa_bwd
import example_gqa_bwd_wgmma_pipelined
import example_mha_bwd_bshd
import example_mha_bwd_bhsd
import example_mha_bwd_bshd_wgmma_pipelined


def bench_example_gqa_bwd_tma_reduce_varlen():
    tilelang.testing.benchmark.process_func(example_gqa_bwd_tma_reduce_varlen.run_regression_perf, name="example_gqa_bwd_tma_reduce_varlen")


def bench_example_gqa_bwd():
    tilelang.testing.benchmark.process_func(example_gqa_bwd.run_regression_perf, name="example_gqa_bwd")


def bench_example_gqa_bwd_wgmma_pipelined():
    tilelang.testing.benchmark.process_func(example_gqa_bwd_wgmma_pipelined.run_regression_perf, name="example_gqa_bwd_wgmma_pipelined")


def bench_example_mha_bwd_bshd():
    tilelang.testing.benchmark.process_func(example_mha_bwd_bshd.run_regression_perf, name="example_mha_bwd_bshd")


def bench_example_mha_bwd_bhsd():
    tilelang.testing.benchmark.process_func(example_mha_bwd_bhsd.run_regression_perf, name="example_mha_bwd_bhsd")


def bench_example_mha_bwd_bshd_wgmma_pipelined():
    tilelang.testing.benchmark.process_func(
        example_mha_bwd_bshd_wgmma_pipelined.run_regression_perf, name="example_mha_bwd_bshd_wgmma_pipelined"
    )


def bench_example_gqa_fwd_bshd_wgmma_pipelined():
    tilelang.testing.benchmark.process_func(
        example_gqa_fwd_bshd_wgmma_pipelined.run_regression_perf, batch=1, heads=16, seq_len=1024, dim=128, is_causal=False, groups=16
    )


def bench_example_gqa_fwd_bshd():
    tilelang.testing.benchmark.process_func(
        example_gqa_fwd_bshd.run_regression_perf, batch=1, heads=16, seq_len=1024, dim=128, is_causal=False, groups=16
    )


def bench_example_mha_fwd_bhsd_wgmma_pipelined():
    tilelang.testing.benchmark.process_func(example_mha_fwd_bhsd_wgmma_pipelined.run_regression_perf)


def bench_example_mha_fwd_bhsd():
    tilelang.testing.benchmark.process_func(example_mha_fwd_bhsd.run_regression_perf)


def bench_example_mha_fwd_bshd_wgmma_pipelined():
    tilelang.testing.benchmark.process_func(example_mha_fwd_bshd_wgmma_pipelined.run_regression_perf, batch=1, heads=32, seq_len=256)


def bench_example_mha_fwd_bshd():
    tilelang.testing.benchmark.process_func(example_mha_fwd_bshd.run_regression_perf, batch=1, seq_len=256)


def bench_example_mha_fwd_varlen():
    tilelang.testing.benchmark.process_func(example_mha_fwd_varlen.run_regression_perf, batch=4, heads=16, seq_len=512, dim=64)


if globals().get("__name__") == "__main__":
    tilelang.testing.bench()
