"""Softmax reduction benchmark — TileLang vs Inductor vs eager.

Workloads (dim=-1, ``torch.nn.functional.softmax``):

    label              shape                  dtypes
    -----------------  ---------------------  --------------------
    attn-weights-4k    (32, 32, 4096)         float16, bfloat16
    attn-weights-32k   (32, 32, 32768)        bfloat16
    lm-head-logits     (4, 102400)            float16, bfloat16

For each backend we measure the GPU latency of a single softmax forward
pass with CUDA events (no profiler — torch.profiler's "Command Buffer
Full" inflates numbers for tiny kernels).  The script also verifies
that the TileLang backend was actually invoked and produced at least
one TIR kernel — a silent fallback to eager would otherwise show up
as identical numbers across the three columns.
"""

import os
os.environ.setdefault("TQDM_DISABLE", "1")

import warnings; warnings.filterwarnings("ignore")
import argparse
import torch
import torch._dynamo
import torch.nn.functional as F

import tilelang  # noqa: F401
from tilelang.graph import backend_config
from tilelang.graph.cache import clear_cache
import tilelang.graph.backend as _be_mod
import tilelang.graph.vm_build as _vm_mod
from tvm import tir as _tir


# ---------------------------------------------------------------------------
# Workload spec — keep in sync with the YAML in the spec file
# ---------------------------------------------------------------------------

SHAPES = [
    (1024, 4096), (1024, 10240), (1024, 20480),
]

DTYPE_MAP = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}


# ---------------------------------------------------------------------------
# Backend probe — same shape as the LLaMA decode bench: refuse to report
# numbers when dynamo silently bypasses the TileLang backend.
# ---------------------------------------------------------------------------

class _BackendProbe:
    def __init__(self):
        self.backend_calls = 0
        self.tir_kernels = 0
        self._orig_backend = None
        self._orig_compile_tir = None

    def install(self):
        from torch._dynamo.backends.registry import _COMPILER_FNS
        self._orig_backend = _be_mod.tilelang_backend
        self._orig_compile_tir = _vm_mod._compile_tir_for_vm

        def _tracked_backend(gm, example_inputs):
            self.backend_calls += 1
            return self._orig_backend(gm, example_inputs)

        def _tracked_compile_tir(tir_mod, target):
            for _gv, fn in tir_mod.functions.items():
                if isinstance(fn, _tir.PrimFunc):
                    if (fn.attrs is not None
                            and fn.attrs.get("tir.is_host_func")):
                        continue
                    self.tir_kernels += 1
            return self._orig_compile_tir(tir_mod, target)

        _COMPILER_FNS["tilelang"] = _tracked_backend
        _be_mod.tilelang_backend = _tracked_backend
        _vm_mod._compile_tir_for_vm = _tracked_compile_tir

    def uninstall(self):
        from torch._dynamo.backends.registry import _COMPILER_FNS
        if self._orig_backend is not None:
            _COMPILER_FNS["tilelang"] = self._orig_backend
            _be_mod.tilelang_backend = self._orig_backend
        if self._orig_compile_tir is not None:
            _vm_mod._compile_tir_for_vm = self._orig_compile_tir

    def reset(self):
        self.backend_calls = 0
        self.tir_kernels = 0


# ---------------------------------------------------------------------------
# Benchmark helpers
# ---------------------------------------------------------------------------


def _make_binary_elementwise_module(op: callable) -> torch.nn.Module:
    class _BinaryElementwise(torch.nn.Module):
        def __init__(self, op):
            super().__init__()
            self.op = op

        def forward(self, x, y):
            return self.op(x.to(torch.float32), y.to(torch.float32)).to(x.dtype)
    return _BinaryElementwise(op)


def _build_runner(model: torch.nn.Module, backend: str):
    if backend == "eager":
        return model
    if backend == "inductor":
        torch._dynamo.reset()
        return torch.compile(model, backend="inductor", mode="default")
    if backend == "tilelang":
        clear_cache()
        torch._dynamo.reset()
        backend_config.vm_clone_output = False
        return torch.compile(model, backend="tilelang")
    raise ValueError(f"unknown backend {backend!r}")


def _time_kernel(runner, x: torch.Tensor, y: torch.Tensor,
                 n_warmup: int = 10, n_bench: int = 100) -> float:
    """Average per-call latency in microseconds (do_bench returns ms)."""
    from tilelang.profiler import do_bench
    return do_bench(lambda: runner(x, y)) * 1000.0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backends", default="eager,inductor,tilelang",
                    help="Comma-separated list (subset of eager,inductor,tilelang).")
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--rtol", type=float, default=1e-2,
                    help="Relative tolerance for correctness vs eager.")
    ap.add_argument("--atol", type=float, default=1e-2,
                    help="Absolute tolerance for correctness vs eager.")
    args = ap.parse_args()

    backends = [b.strip() for b in args.backends.split(",") if b.strip()]
    print(f"GPU:       {torch.cuda.get_device_name()}")
    print(f"Backends:  {backends}")
    print(f"Warmup:    {args.warmup}    Iters: {args.iters}\n")

    rows = []  # (label, shape, dtype, {backend: us}, ok_flags, probe_info)
    ops = [torch.sub, torch.mul, torch.div, torch.max, torch.min]
    # ops = [torch.div]
    dtypes = ["float16", "bfloat16"]
    probe = _BackendProbe()
    probe.install()
    try:
        # for shape in SHAPES:
        #     for dtype_name in dtypes:
        from itertools import product
        for op, shape, dtype in product(ops, SHAPES, dtypes):
                print(f"=== shape={tuple(shape)}  op={op.__name__}  dtype={dtype} ===",
                      flush=True)

                # Realistic distribution: zero-mean unit-ish so softmax
                # doesn't degenerate.  Stable across dtypes.
                torch.manual_seed(0)
                x = torch.randn(*shape, dtype=DTYPE_MAP[dtype], device="cuda")
                y = torch.randn(*shape, dtype=DTYPE_MAP[dtype], device="cuda")

                model = _make_binary_elementwise_module(op).cuda().eval()

                # Eager reference output for correctness checking
                with torch.no_grad():
                    ref = model(x, y)

                results: dict[str, float] = {}
                ok: dict[str, str] = {}
                tl_probe_calls = 0
                tl_probe_kernels = 0

                for backend in backends:
                    runner = _build_runner(model, backend)
                    if backend == "tilelang":
                        probe.reset()
                    # Warm-up + correctness check (single call so the
                    # compiled graph exists before we time it).
                    with torch.no_grad():
                        out = runner(x, y)
                    if backend == "tilelang":
                        tl_probe_calls = probe.backend_calls
                        tl_probe_kernels = probe.tir_kernels

                    # Correctness vs eager
                    out_f = out.float()
                    ref_f = ref.float()
                    err = (out_f - ref_f).abs().max().item()
                    if torch.allclose(out_f, ref_f, rtol=args.rtol, atol=args.atol):
                        ok[backend] = "ok"
                    else:
                        ok[backend] = f"err={err:.2e}"

                    us = _time_kernel(runner, x, y, args.warmup, args.iters)
                    results[backend] = us
                    print(f"  {backend:<10s} {us:>10.2f} us   ({ok[backend]})",
                          flush=True)

                rows.append((op.__name__, tuple(shape), dtype, results, ok,
                             tl_probe_calls, tl_probe_kernels))
                print()
    finally:
        probe.uninstall()

    # ---- Verify TileLang was actually exercised ----
    if "tilelang" in backends:
        for _, shape, dtype, _r, _ok, calls, kernels in rows:
            if calls == 0:
                raise AssertionError(
                    f"[{shape}] TileLang backend was never "
                    "invoked — dynamo bypassed it (silent fallback to eager).")
            if kernels == 0:
                raise AssertionError(
                    f"[{shape}] TileLang backend ran but "
                    "produced zero TIR kernels — every op fell back to torch.")

    # ---- Summary table ----
    print("\n=== Summary ===")
    header_cols = [
        "op", "shape", "dtype",
        *(f"{b} (us)" for b in backends),
    ]
    if "eager" in backends and "tilelang" in backends:
        header_cols.append("tl/eager")
    if "inductor" in backends and "tilelang" in backends:
        header_cols.append("tl/ind")

    widths = [max(len(c), 14) for c in header_cols]
    fmt = "  ".join(f"{{:<{w}s}}" if i < 3 else f"{{:>{w}s}}"
                    for i, w in enumerate(widths))
    print(fmt.format(*header_cols))
    print("  ".join("-" * w for w in widths))

    for op, shape, dtype, r, _ok, _calls, _kernels in rows:
        cells = [op, str(shape), dtype]
        for b in backends:
            cells.append(f"{r[b]:.1f}")
        if "eager" in backends and "tilelang" in backends:
            cells.append(f"{r['tilelang'] / r['eager']:.2f}x")
        if "inductor" in backends and "tilelang" in backends:
            cells.append(f"{r['tilelang'] / r['inductor']:.2f}x")
        print(fmt.format(*cells))


if __name__ == "__main__":
    main()
