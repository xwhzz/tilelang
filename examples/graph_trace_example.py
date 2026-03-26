"""Example: Graph compilation tracing and kernel source inspection.

Demonstrates:
  1. get_compilation_traces() — access compilation traces
  2. trace.summary()          — compilation summary (schedule rules, timing)
  3. trace.show_tir()         — per-kernel unscheduled/scheduled TIR
"""

import torch
import torch._dynamo

import tilelang  # noqa: F401  (triggers backend registration)
from tilelang.jit.backend import clear_compilation_traces, get_compilation_traces

# ── 1. Define a small MLP and compile via tilelang backend ────────────

torch._dynamo.reset()
clear_compilation_traces()

dim = 256

@torch.compile(backend="tilelang")
def swiglu_ffn(x, w_gate, w_up, w_down):
    """SwiGLU FFN: down_proj(silu(gate_proj(x)) * up_proj(x))"""
    gate = x @ w_gate  # (B, dim) @ (dim, ffn_dim)
    return gate.to(torch.bfloat16)


# Trigger compilation with concrete shapes
B, ffn_dim = 32, dim * 4
x = torch.randn(B, dim, device="cuda", dtype=torch.bfloat16)
w_gate = torch.randn(dim, ffn_dim, device="cuda", dtype=torch.bfloat16)
w_up = torch.randn(dim, ffn_dim, device="cuda", dtype=torch.bfloat16)
w_down = torch.randn(ffn_dim, dim, device="cuda", dtype=torch.bfloat16)

out = swiglu_ffn(x, w_gate, w_up, w_down)
ref = x @ w_gate
torch.testing.assert_close(out, ref, rtol=1e-2, atol=1e-2)

# ── 2. Show compilation traces ───────────────────────────────────────

traces = get_compilation_traces()
print("=" * 70)
print("COMPILATION TRACES")
print("=" * 70)
for i, trace in enumerate(traces):
    print(f"--- Subgraph {i} ---")
    print(trace.summary())
    print()

# ── 3. Programmatic trace access ────────────────────────────────────

if traces:
    trace = traces[0]
    print("=" * 70)
    print("PROGRAMMATIC TRACE ACCESS")
    print("=" * 70)
    print(f"Compilation path : {trace.compilation_path}")
    print(f"Architecture     : {trace.arch}")
    print(f"Number of kernels: {len(trace.kernels)}")
    print()
    for kt in trace.kernels:
        print(f"  Kernel: {kt.name}")
        print(f"    Schedule rule: {kt.schedule_rule}")
        print(f"    Output dtype : {kt.output_dtype}")
        print()

    # Schedule rule matching summary
    print("Schedule matches:")
    for func_name, rule in trace.schedule_matches.items():
        print(f"  {func_name:40s} → {rule}")

    # ── 4. Per-kernel TIR ────────────────────────────────────────────

    print()
    print("=" * 70)
    print("PER-KERNEL TIR (all kernels)")
    print("=" * 70)
    trace.show_tir()

print()
print("\033[92mDone.\033[0m")
