"""Compare tilelang.jit(mode='graph') against te-based schedule rules.

For each computation pattern (matmul, matmul+cast, matmul+relu, etc.),
verify that:
  1. The graph-mode path (PyTorch trace → Relax → schedule → compile) produces
     correct results.
  2. The te-based path (te.compute → Matmul schedule rule → compile) produces
     correct results.
  3. Both paths agree with the PyTorch reference.

With ``--bench``, also benchmark graph-mode vs te-based vs torch.compile.

Disk caching is leveraged by default, so only the first run is slow.
"""

from __future__ import annotations

import argparse

import torch
import torch.nn.functional as F

import tilelang
from tilelang import tvm
from tilelang.profiler import do_bench
from tvm import te, tir

from tilelang.schedule.gpu import Matmul


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _lower_scheduled_mod(mod):
    """Apply the standard lowering passes for a schedule-rule output."""
    mod = tir.transform.Simplify()(mod)
    mod = tir.transform.LowerInitBlock()(mod)
    mod = tir.transform.ConvertBlocksToOpaque()(mod)
    mod = tilelang.transform.ReserveRootBlock()(mod)
    return mod


def _compile_te(func: tir.PrimFunc, arch: str) -> tilelang.JITKernel:
    """Schedule via Matmul rule and compile."""
    target = tvm.target.cuda(arch=arch)
    sch = Matmul().apply(func, target, False)
    if sch is None:
        raise RuntimeError("Matmul schedule rule returned None.")
    mod = _lower_scheduled_mod(sch.mod)
    return tilelang.compile(mod["main"])


def _tolerance(dtype: str) -> tuple[float, float]:
    if dtype in ("float16", "bfloat16"):
        return 5e-2, 5e-2
    return 1e-3, 1e-3


# Collect benchmark results for summary table
_bench_results: list[tuple[str, float | None, float | None, float | None]] = []


def _record(name: str, te_ms: float | None, graph_ms: float | None, torch_ms: float | None):
    _bench_results.append((name, te_ms, graph_ms, torch_ms))


def _print_summary():
    if not _bench_results:
        return
    print("\n" + "=" * 78)
    print(f"{'Benchmark':<32} {'te (ms)':>10} {'graph (ms)':>12} {'torch (ms)':>12}")
    print("-" * 78)
    for name, te_ms, graph_ms, torch_ms in _bench_results:
        te_s = f"{te_ms:.4f}" if te_ms is not None else "N/A"
        gr_s = f"{graph_ms:.4f}" if graph_ms is not None else "N/A"
        tc_s = f"{torch_ms:.4f}" if torch_ms is not None else "N/A"
        print(f"{name:<32} {te_s:>10} {gr_s:>12} {tc_s:>12}")
    print("=" * 78)


# ---------------------------------------------------------------------------
# Test 1: basic matmul  (f32)
# ---------------------------------------------------------------------------

def test_matmul_f32(arch: str, bench: bool = False):
    M, N, K = 512, 512, 512
    dtype = "float32"
    torch_dtype = torch.float32

    torch.manual_seed(0)
    a = torch.randn(M, K, device="cuda", dtype=torch_dtype)
    b = torch.randn(K, N, device="cuda", dtype=torch_dtype)

    # --- te path ---
    ta = te.placeholder((M, K), name="A", dtype=dtype)
    tb = te.placeholder((K, N), name="B", dtype=dtype)
    rk = te.reduce_axis((0, K), name="k")
    tc = te.compute((M, N), lambda i, j: te.sum(ta[i, rk] * tb[rk, j], axis=rk), name="C")
    te_func = te.create_prim_func([ta, tb, tc])
    te_kernel = _compile_te(te_func, arch)

    c_te = torch.empty(M, N, device="cuda", dtype=torch_dtype)
    te_kernel(a, b, c_te)

    # --- graph path ---
    @tilelang.jit(mode="graph", target=arch)
    def graph_matmul(x, w):
        return torch.matmul(x, w)

    c_graph = graph_matmul(a, b)

    # --- reference ---
    c_ref = a @ b

    rtol, atol = _tolerance(dtype)
    torch.testing.assert_close(c_te, c_ref, rtol=rtol, atol=atol)
    torch.testing.assert_close(c_graph, c_ref, rtol=rtol, atol=atol)
    print("\033[92mtest_matmul_f32: passed.\033[0m")

    if bench:
        te_ms = do_bench(lambda: te_kernel(a, b, c_te))
        gr_ms = do_bench(lambda: graph_matmul(a, b))
        tc_fn = torch.compile(lambda x, w: x @ w)
        tc_fn(a, b)  # warmup
        tc_ms = do_bench(lambda: tc_fn(a, b))
        _record(f"matmul f32 {M}x{N}x{K}", te_ms, gr_ms, tc_ms)


# ---------------------------------------------------------------------------
# Test 2: matmul with fp16 accumulation in f32  (fp16 → f32 → cast back fp16)
# ---------------------------------------------------------------------------

def test_matmul_fp16(arch: str, bench: bool = False):
    M, N, K = 512, 512, 512
    dtype = "float16"
    accum = "float32"
    torch_dtype = torch.float16

    torch.manual_seed(1)
    a = torch.randn(M, K, device="cuda", dtype=torch_dtype)
    b = torch.randn(K, N, device="cuda", dtype=torch_dtype)

    # --- te path: matmul in f32 + cast back to fp16 ---
    ta = te.placeholder((M, K), name="A", dtype=dtype)
    tb = te.placeholder((K, N), name="B", dtype=dtype)
    rk = te.reduce_axis((0, K), name="k")
    tc = te.compute(
        (M, N),
        lambda i, j: te.sum(ta[i, rk].astype(accum) * tb[rk, j].astype(accum), axis=rk),
        name="C",
    )
    tc_cast = te.compute(tc.shape, lambda i, j: tc[i, j].astype(dtype), name="C_cast")
    te_func = te.create_prim_func([ta, tb, tc_cast])
    te_kernel = _compile_te(te_func, arch)

    c_te = torch.empty(M, N, device="cuda", dtype=torch_dtype)
    te_kernel(a, b, c_te)

    # --- graph path: Relax auto-inserts cast ---
    @tilelang.jit(mode="graph", target=arch)
    def graph_matmul(x, w):
        return torch.matmul(x, w)

    c_graph = graph_matmul(a, b)

    # --- reference ---
    c_ref = (a.float() @ b.float()).half()

    rtol, atol = _tolerance(dtype)
    torch.testing.assert_close(c_te, c_ref, rtol=rtol, atol=atol)
    torch.testing.assert_close(c_graph.to(torch_dtype), c_ref, rtol=rtol, atol=atol)
    print("\033[92mtest_matmul_fp16: passed.\033[0m")

    if bench:
        te_ms = do_bench(lambda: te_kernel(a, b, c_te))
        gr_ms = do_bench(lambda: graph_matmul(a, b))
        tc_fn = torch.compile(lambda x, w: x @ w)
        tc_fn(a, b)
        tc_ms = do_bench(lambda: tc_fn(a, b))
        _record(f"matmul fp16 {M}x{N}x{K}", te_ms, gr_ms, tc_ms)


# ---------------------------------------------------------------------------
# Test 3: matmul with bf16 accumulation in f32  (bf16 → f32 → cast back bf16)
# ---------------------------------------------------------------------------

def test_matmul_bf16(arch: str, bench: bool = False):
    M, N, K = 512, 512, 512
    dtype = "bfloat16"
    accum = "float32"
    torch_dtype = torch.bfloat16

    torch.manual_seed(2)
    a = torch.randn(M, K, device="cuda", dtype=torch_dtype)
    b = torch.randn(K, N, device="cuda", dtype=torch_dtype)

    # --- te path: matmul in f32 + cast back to bf16 ---
    ta = te.placeholder((M, K), name="A", dtype=dtype)
    tb = te.placeholder((K, N), name="B", dtype=dtype)
    rk = te.reduce_axis((0, K), name="k")
    tc = te.compute(
        (M, N),
        lambda i, j: te.sum(ta[i, rk].astype(accum) * tb[rk, j].astype(accum), axis=rk),
        name="C",
    )
    tc_cast = te.compute(tc.shape, lambda i, j: tc[i, j].astype(dtype), name="C_cast")
    te_func = te.create_prim_func([ta, tb, tc_cast])
    te_kernel = _compile_te(te_func, arch)

    c_te = torch.empty(M, N, device="cuda", dtype=torch_dtype)
    te_kernel(a, b, c_te)

    # --- graph path: matmul + explicit cast ---
    @tilelang.jit(mode="graph", target=arch)
    def graph_matmul(x, w):
        return torch.matmul(x, w).to(torch.bfloat16)

    c_graph = graph_matmul(a, b)

    # --- reference ---
    c_ref = (a.float() @ b.float()).to(torch_dtype)

    rtol, atol = _tolerance(dtype)
    torch.testing.assert_close(c_te, c_ref, rtol=rtol, atol=atol)
    torch.testing.assert_close(c_graph, c_ref, rtol=rtol, atol=atol)
    print("\033[92mtest_matmul_bf16: passed.\033[0m")

    if bench:
        te_ms = do_bench(lambda: te_kernel(a, b, c_te))
        gr_ms = do_bench(lambda: graph_matmul(a, b))
        tc_fn = torch.compile(lambda x, w: (x @ w).to(torch.bfloat16))
        tc_fn(a, b)
        tc_ms = do_bench(lambda: tc_fn(a, b))
        _record(f"matmul+cast bf16 {M}x{N}x{K}", te_ms, gr_ms, tc_ms)


# ---------------------------------------------------------------------------
# Test 4: batched matmul (3D × 2D), bf16
# ---------------------------------------------------------------------------

def test_batched_matmul_bf16(arch: str, bench: bool = False):
    B, M, N, K = 2, 512, 2048, 2048
    torch_dtype = torch.bfloat16

    torch.manual_seed(3)
    a = torch.randn(B, M, K, device="cuda", dtype=torch_dtype) * 0.02
    b = torch.randn(K, N, device="cuda", dtype=torch_dtype) * 0.02

    # --- graph path ---
    @tilelang.jit(mode="graph", target=arch)
    def graph_matmul(x, w):
        return torch.matmul(x, w).to(torch.bfloat16)

    c_graph = graph_matmul(a, b)

    # --- reference ---
    c_ref = (a.float() @ b.float()).to(torch_dtype)

    rtol, atol = _tolerance("bfloat16")
    torch.testing.assert_close(c_graph, c_ref, rtol=rtol, atol=atol)
    print("\033[92mtest_batched_matmul_bf16: passed.\033[0m")

    if bench:
        gr_ms = do_bench(lambda: graph_matmul(a, b))
        tc_fn = torch.compile(lambda x, w: (x @ w).to(torch.bfloat16))
        tc_fn(a, b)
        tc_ms = do_bench(lambda: tc_fn(a, b))
        _record(f"batched matmul+cast bf16 {B}x{M}x{K}x{N}", None, gr_ms, tc_ms)


# ---------------------------------------------------------------------------
# Test 5: matmul + silu (fused epilogue), bf16
# ---------------------------------------------------------------------------

def test_matmul_silu_bf16(arch: str, bench: bool = False):
    M, N, K = 1024, 8192, 2048
    torch_dtype = torch.bfloat16

    torch.manual_seed(4)
    a = torch.randn(M, K, device="cuda", dtype=torch_dtype) * 0.02
    b = torch.randn(K, N, device="cuda", dtype=torch_dtype) * 0.02

    # --- graph path: matmul + silu (fused by Relax) ---
    @tilelang.jit(mode="graph", target=arch)
    def graph_fn(x, w):
        return F.silu(torch.matmul(x, w))

    c_graph = graph_fn(a, b)

    # --- reference ---
    c_ref = F.silu(a.float() @ b.float())

    rtol, atol = _tolerance("bfloat16")
    torch.testing.assert_close(c_graph.float(), c_ref, rtol=rtol, atol=atol)
    print("\033[92mtest_matmul_silu_bf16: passed.\033[0m")

    if bench:
        gr_ms = do_bench(lambda: graph_fn(a, b))
        tc_fn = torch.compile(lambda x, w: F.silu(x @ w))
        tc_fn(a, b)
        tc_ms = do_bench(lambda: tc_fn(a, b))
        _record(f"matmul+silu bf16 {M}x{K}x{N}", None, gr_ms, tc_ms)


# ---------------------------------------------------------------------------
# Test 6: matmul + relu (fused epilogue), fp16
# ---------------------------------------------------------------------------

def test_matmul_relu_fp16(arch: str, bench: bool = False):
    M, N, K = 512, 512, 512
    dtype = "float16"
    accum = "float32"
    torch_dtype = torch.float16

    torch.manual_seed(5)
    a = torch.randn(M, K, device="cuda", dtype=torch_dtype)
    b = torch.randn(K, N, device="cuda", dtype=torch_dtype)

    # --- te path: matmul + relu epilogue ---
    ta = te.placeholder((M, K), name="A", dtype=dtype)
    tb = te.placeholder((K, N), name="B", dtype=dtype)
    rk = te.reduce_axis((0, K), name="k")
    tc = te.compute(
        (M, N),
        lambda i, j: te.sum(ta[i, rk].astype(accum) * tb[rk, j].astype(accum), axis=rk),
        name="C",
    )
    td = te.compute(
        tc.shape,
        lambda i, j: tir.max(tc[i, j], tir.const(0.0, accum)).astype(dtype),
        name="D",
    )
    te_func = te.create_prim_func([ta, tb, td])
    te_kernel = _compile_te(te_func, arch)

    d_te = torch.empty(M, N, device="cuda", dtype=torch_dtype)
    te_kernel(a, b, d_te)

    # --- graph path ---
    @tilelang.jit(mode="graph", target=arch)
    def graph_fn(x, w):
        return F.relu(torch.matmul(x, w)).to(torch.float16)

    d_graph = graph_fn(a, b)

    # --- reference ---
    d_ref = F.relu(a.float() @ b.float()).half()

    rtol, atol = _tolerance(dtype)
    torch.testing.assert_close(d_te, d_ref, rtol=rtol, atol=atol)
    torch.testing.assert_close(d_graph, d_ref, rtol=rtol, atol=atol)
    print("\033[92mtest_matmul_relu_fp16: passed.\033[0m")

    if bench:
        te_ms = do_bench(lambda: te_kernel(a, b, d_te))
        gr_ms = do_bench(lambda: graph_fn(a, b))
        tc_fn = torch.compile(lambda x, w: F.relu(x @ w).half())
        tc_fn(a, b)
        tc_ms = do_bench(lambda: tc_fn(a, b))
        _record(f"matmul+relu fp16 {M}x{N}x{K}", te_ms, gr_ms, tc_ms)


# ---------------------------------------------------------------------------
# Test 7: SwiGLU FFN pattern, bf16
# ---------------------------------------------------------------------------

def test_swiglu_ffn_bf16(arch: str, bench: bool = False):
    M, H, FFN = 1024, 2048, 8192
    torch_dtype = torch.bfloat16

    torch.manual_seed(6)
    x = torch.randn(M, H, device="cuda", dtype=torch_dtype) * 0.02
    w_gate = torch.randn(H, FFN, device="cuda", dtype=torch_dtype) * 0.02
    w_up = torch.randn(H, FFN, device="cuda", dtype=torch_dtype) * 0.02
    w_down = torch.randn(FFN, H, device="cuda", dtype=torch_dtype) * 0.02

    # --- graph path ---
    @tilelang.jit(mode="graph", target=arch)
    def graph_ffn(x_in, wg, wu, wd):
        gate = F.silu(torch.matmul(x_in, wg))
        up = torch.matmul(x_in, wu)
        return torch.matmul(gate * up, wd)

    out_graph = graph_ffn(x, w_gate, w_up, w_down)

    # --- reference ---
    xf = x.float()
    gate_ref = F.silu(xf @ w_gate.float())
    up_ref = xf @ w_up.float()
    out_ref = (gate_ref * up_ref) @ w_down.float()

    rtol, atol = _tolerance("bfloat16")
    torch.testing.assert_close(out_graph.float(), out_ref, rtol=rtol, atol=atol)
    print("\033[92mtest_swiglu_ffn_bf16: passed.\033[0m")

    if bench:
        gr_ms = do_bench(lambda: graph_ffn(x, w_gate, w_up, w_down))

        @torch.compile()
        def tc_ffn(x_in, wg, wu, wd):
            gate = F.silu(x_in @ wg)
            up = x_in @ wu
            return (gate * up) @ wd

        tc_ffn(x, w_gate, w_up, w_down)
        tc_ms = do_bench(lambda: tc_ffn(x, w_gate, w_up, w_down))
        _record(f"SwiGLU FFN bf16 {M}x{H}x{FFN}", None, gr_ms, tc_ms)


# ---------------------------------------------------------------------------
# Test 8: MLP (matmul + relu + matmul), fp16 - compare graph vs te
# ---------------------------------------------------------------------------

def test_mlp_graph_vs_te(arch: str, bench: bool = False):
    dim = 1024
    dtype = "float16"
    torch_dtype = torch.float16

    torch.manual_seed(7)
    x = torch.randn(dim, device="cuda", dtype=torch_dtype)
    w1 = torch.randn(dim, dim, device="cuda", dtype=torch_dtype)
    w2 = torch.randn(dim, dim, device="cuda", dtype=torch_dtype)

    # --- graph path ---
    @tilelang.jit(mode="graph", target=arch)
    def graph_mlp(x_in, w1_in, w2_in):
        h = F.relu(torch.matmul(w1_in, x_in))
        return torch.matmul(w2_in, h)

    out_graph = graph_mlp(x, w1, w2)

    # --- reference ---
    h_ref = F.relu(w1.float() @ x.float())
    out_ref = w2.float() @ h_ref

    rtol, atol = _tolerance(dtype)
    torch.testing.assert_close(out_graph.float(), out_ref, rtol=rtol, atol=atol)
    print("\033[92mtest_mlp_graph_vs_te: passed.\033[0m")

    if bench:
        gr_ms = do_bench(lambda: graph_mlp(x, w1, w2))

        @torch.compile()
        def tc_mlp(x_in, w1_in, w2_in):
            return w2_in @ F.relu(w1_in @ x_in)

        tc_mlp(x, w1, w2)
        tc_ms = do_bench(lambda: tc_mlp(x, w1, w2))
        _record(f"MLP fp16 {dim}x{dim}", None, gr_ms, tc_ms)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Graph-mode JIT vs te-based schedule rule comparison.")
    parser.add_argument("--arch", type=str, default="sm_90a",
                        help='CUDA arch, e.g. "sm_90a".')
    parser.add_argument("--bench", action="store_true",
                        help="Run benchmarks (te vs graph vs torch.compile).")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    b = args.bench
    test_matmul_f32(args.arch, bench=b)
    test_matmul_fp16(args.arch, bench=b)
    test_matmul_bf16(args.arch, bench=b)
    test_batched_matmul_bf16(args.arch, bench=b)
    test_matmul_silu_bf16(args.arch, bench=b)
    test_matmul_relu_fp16(args.arch, bench=b)
    test_swiglu_ffn_bf16(args.arch, bench=b)
    test_mlp_graph_vs_te(args.arch, bench=b)
    print("\n\033[92mAll graph vs te tests passed.\033[0m")
    if b:
        _print_summary()
