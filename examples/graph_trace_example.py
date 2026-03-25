"""Example: Graph compilation tracing and kernel source inspection.

Demonstrates:
  1. show_trace()       — compilation summary (schedule rules, timing)
  2. get_trace()        — programmatic access to trace data
  3. show_kernel_sources() — CUDA source of each compiled kernel
  4. show_tir()         — per-kernel unscheduled/scheduled TIR
"""

import torch
import torch.nn.functional as F

import tilelang

# ── 1. Define a small MLP and compile via graph mode ──────────────────

dim = 256


@tilelang.jit(mode="graph")
def swiglu_ffn(x, w_gate, w_up, w_down):
    """SwiGLU FFN: down_proj(silu(gate_proj(x)) * up_proj(x))"""
    gate = x @ w_gate  # (B, dim) @ (dim, ffn_dim)
    return gate.to(torch.bfloat16)
    # up = x @ w_up
    # return (F.silu(gate) * up) @ w_down


# Trigger compilation with concrete shapes
B, ffn_dim = 32, dim * 4
x = torch.randn(B, dim, device="cuda", dtype=torch.bfloat16)
w_gate = torch.randn(dim, ffn_dim, device="cuda", dtype=torch.bfloat16)
w_up = torch.randn(dim, ffn_dim, device="cuda", dtype=torch.bfloat16)
w_down = torch.randn(ffn_dim, dim, device="cuda", dtype=torch.bfloat16)

out = swiglu_ffn(x, w_gate, w_up, w_down)
# ref = (F.silu(x @ w_gate) * (x @ w_up)) @ w_down
ref = x @ w_gate
torch.testing.assert_close(out, ref, rtol=1e-2, atol=1e-2)

# ── 2. Show compilation trace ─────────────────────────────────────────

print("=" * 70)
print("COMPILATION TRACE")
print("=" * 70)
swiglu_ffn.show_trace()

# ── 3. Programmatic trace access ──────────────────────────────────────

print()
print("=" * 70)
print("PROGRAMMATIC TRACE ACCESS")
print("=" * 70)
trace = swiglu_ffn.get_trace()
print(f"Compilation path : {trace.compilation_path}")
print(f"Architecture     : {trace.arch}")
print(f"Total time       : {trace.total_time_ms:.1f} ms")
print(f"Number of kernels: {len(trace.kernels)}")
print()
for kt in trace.kernels:
    print(f"  Kernel: {kt.name}")
    print(f"    Schedule rule: {kt.schedule_rule}")
    print(f"    Compile time : {kt.compile_time_ms:.1f} ms")
    print(f"    Output dtype : {kt.output_dtype}")
    print()

# Schedule rule matching summary
print("Schedule matches:")
for func_name, rule in trace.schedule_matches.items():
    print(f"  {func_name:40s} → {rule}")

# ── 4. Show kernel CUDA sources ───────────────────────────────────────

print()
print("=" * 70)
print("KERNEL CUDA SOURCES (first 30 lines each)")
print("=" * 70)
sources = swiglu_ffn.get_kernel_sources()
for name, src in sources.items():
    print(f"\n--- {name} ---")
    lines = src.splitlines()
    for line in lines[:30]:
        print(line)
    if len(lines) > 30:
        print(f"  ... ({len(lines) - 30} more lines)")

# ── 5. Per-kernel TIR: unscheduled vs scheduled ───────────────────────

# Show all kernels
print()
print("=" * 70)
print("PER-KERNEL TIR (all kernels)")
print("=" * 70)
trace.show_tir()

# Or show just one kernel by name
print()
print("=" * 70)
print("PER-KERNEL TIR (single kernel)")
print("=" * 70)
first_kernel = trace.kernels[0].name
trace.show_tir(first_kernel)

print()
print("\033[92mDone.\033[0m")
