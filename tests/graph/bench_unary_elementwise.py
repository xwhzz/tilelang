"""Unary elementwise benchmark — TileLang vs Inductor.

Workloads: exp, gelu on 1-D tensors with fp16/bf16.
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


SHAPES = [262_144, 1_048_576, 4_000_000]

DTYPE_MAP = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


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


def _make_unary_module(op_name: str) -> torch.nn.Module:
    class _Exp(torch.nn.Module):
        def forward(self, x):
            return torch.exp(x.float()).to(x.dtype)

    class _GELU(torch.nn.Module):
        def forward(self, x):
            return F.gelu(x.float()).to(x.dtype)

    return {"exp": _Exp, "gelu": _GELU}[op_name]()


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


def _time_kernel(runner, x: torch.Tensor,
                 n_warmup: int = 10, n_bench: int = 100) -> float:
    from tilelang.profiler import do_bench
    return do_bench(lambda: runner(x)) * 1000.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backends", default="eager,inductor,tilelang",
                    help="Comma-separated list.")
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--rtol", type=float, default=1e-2)
    ap.add_argument("--atol", type=float, default=1e-2)
    args = ap.parse_args()

    backends = [b.strip() for b in args.backends.split(",") if b.strip()]
    print(f"GPU:       {torch.cuda.get_device_name()}")
    print(f"Backends:  {backends}")
    print(f"Warmup:    {args.warmup}    Iters: {args.iters}\n")

    ops = ["exp", "gelu"]
    dtypes = ["float16", "bfloat16"]
    rows = []

    probe = _BackendProbe()
    probe.install()
    try:
        from itertools import product
        for op_name, shape, dtype_name in product(ops, SHAPES, dtypes):
            dtype = DTYPE_MAP[dtype_name]
            print(f"=== op={op_name}  N={shape}  dtype={dtype_name} ===",
                  flush=True)

            torch.manual_seed(0)
            x = torch.randn(shape, dtype=dtype, device="cuda")

            model = _make_unary_module(op_name).cuda().eval()

            with torch.no_grad():
                ref = model(x)

            results: dict[str, float] = {}
            ok: dict[str, str] = {}
            tl_probe_calls = 0
            tl_probe_kernels = 0

            for backend in backends:
                runner = _build_runner(model, backend)
                if backend == "tilelang":
                    probe.reset()
                with torch.no_grad():
                    out = runner(x)
                if backend == "tilelang":
                    tl_probe_calls = probe.backend_calls
                    tl_probe_kernels = probe.tir_kernels

                out_f = out.float()
                ref_f = ref.float()
                mask = ref_f.isfinite() & out_f.isfinite()
                if mask.sum() > 0:
                    err = (out_f[mask] - ref_f[mask]).abs().max().item()
                else:
                    err = 0.0
                if torch.allclose(out_f[mask], ref_f[mask],
                                  rtol=args.rtol, atol=args.atol):
                    ok[backend] = "ok"
                else:
                    ok[backend] = f"err={err:.2e}"

                us = _time_kernel(runner, x, args.warmup, args.iters)
                results[backend] = us
                print(f"  {backend:<10s} {us:>10.2f} us   ({ok[backend]})",
                      flush=True)

            rows.append((op_name, shape, dtype_name, results, ok,
                         tl_probe_calls, tl_probe_kernels))
            print()
    finally:
        probe.uninstall()

    # ---- Verify TileLang was actually exercised ----
    if "tilelang" in backends:
        for op_name, shape, dtype_name, _r, _ok, calls, kernels in rows:
            if calls == 0:
                raise AssertionError(
                    f"[{op_name} N={shape} {dtype_name}] TileLang backend "
                    "was never invoked.")
            if kernels == 0:
                raise AssertionError(
                    f"[{op_name} N={shape} {dtype_name}] TileLang produced "
                    "zero TIR kernels.")

    # ---- Summary table ----
    print("\n=== Summary ===")
    header_cols = [
        "op", "N", "dtype",
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

    for op_name, shape, dtype_name, r, _ok, _calls, _kernels in rows:
        cells = [op_name, str(shape), dtype_name]
        for b in backends:
            cells.append(f"{r[b]:.1f}")
        if "eager" in backends and "tilelang" in backends:
            cells.append(f"{r['tilelang'] / r['eager']:.2f}x")
        if "inductor" in backends and "tilelang" in backends:
            cells.append(f"{r['tilelang'] / r['inductor']:.2f}x")
        print(fmt.format(*cells))


if __name__ == "__main__":
    main()
