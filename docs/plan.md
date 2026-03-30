# TileLang torch.compile Backend: Refactor, Relocate, and LLaMA-2 E2E

## Goal Description

Refactor the TileLang `torch.compile` backend to replace ad-hoc mechanisms with principled abstractions, relocate the module from `tilelang/jit/torch_compile/` to `tilelang/torch_compile/` (removing old shims), and establish a LLaMA-2-7B end-to-end correctness benchmark using native HuggingFace SDPA as a permanent extern op.

## Acceptance Criteria

Following TDD philosophy, each criterion includes positive and negative tests for deterministic verification.

- AC-1: Module lives at `tilelang/torch_compile/` and old `tilelang/jit/{backend,graph,codegen,runtime}.py` shims are removed
  - Positive Tests (expected to PASS):
    - `from tilelang.torch_compile.api import tilelang_backend` succeeds
    - `torch.compile(model, backend="tilelang")` works after relocation
    - Existing tests in `tests/end2end/` pass without modification (they use public API, not shim paths)
  - Negative Tests (expected to FAIL):
    - `from tilelang.jit.backend import tilelang_backend` raises `ImportError`
    - `from tilelang.jit.graph import compile_subgraph_direct` raises `ImportError`

- AC-2: No wrapper codegen path fabricates outputs for failed kernels
  - Positive Tests (expected to PASS):
    - A subgraph where one TIR function fails compilation and has no extern backing → entire subgraph falls back to eager, producing correct results
    - A subgraph with extern ops (SDPA, embedding) + compiled TIR kernels → correct mixed execution
  - Negative Tests (expected to FAIL):
    - A subgraph with a failed kernel producing `torch.empty()` placeholder → this codepath no longer exists
    - `_TORCH_FALLBACK_PATTERNS` dictionary no longer exists in the codebase

- AC-3: Ad-hoc mechanisms replaced with principled abstractions
  - AC-3a: ExternPolicy replaces `_ALWAYS_EXTERN` and `_infer_failed_op_name`
    - Positive: SDPA identified as extern via target/qualname identity (not `__name__` string)
    - Positive: Custom `torch.library` ops auto-detected as extern
    - Negative: An op not in the extern policy but absent from TVM's convert_map → auto-stubbed (not silently dropped)
  - AC-3b: CompileCapability replaces non-float buffer pre-filter
    - Positive: TIR function with int64 buffers → detected and excluded before compilation
    - Negative: TIR function with only float16/float32 buffers → not excluded
  - AC-3c: FX graph canonicalization passes preserved as separate pre-processing
    - Positive: `cat([torch.tensor([]), x])` simplified to `x` before `from_fx`
    - Positive: Diamond breaking and scalar folding still applied

- AC-4: LLaMA-2-7B produces correct output with native SDPA as extern op
  - AC-4.1: Forward correctness
    - Positive: `torch.testing.assert_close(tl_logits, eager_logits, rtol=1e-2, atol=0.05)` passes
    - Negative: A model with random weights (0 layers) → different from pretrained output
  - AC-4.2: Token-match for text generation
    - Positive: Greedy decode with fixed seed/prompt produces identical tokens across eager and tilelang
    - Negative: Different random seeds → different tokens (demonstrates the test is meaningful)
  - AC-4.3: CI smoke test with synthetic model
    - Positive: 7B-shape synthetic model (2 layers) passes correctness check
    - Negative: Deliberately wrong compilation (skipped scheduling) → fails correctness

- AC-5: `GraphCompileTrace` reports per-subgraph compilation composition
  - Positive Tests (expected to PASS):
    - Trace contains `n_compiled`, `n_extern`, `n_fallback_eager` counts for compiled subgraphs
    - Cache-hit traces have these counts as `None`
  - Negative Tests (expected to FAIL):
    - A subgraph with 5 compiled + 2 extern kernels reports `n_compiled=3` → count mismatch detected

## Path Boundaries

### Upper Bound (Maximum Acceptable Scope)

The implementation relocates the module, replaces all identified ad-hoc mechanisms (ExternPolicy, CompileCapability, removal of `_TORCH_FALLBACK_PATTERNS` and `_fallback_outputs` bridge), adds structured trace reporting, includes the LLaMA-2-7B pretrained benchmark with both forward correctness and token-match generation tests, and adds a CI smoke test with a synthetic model.

### Lower Bound (Minimum Acceptable Scope)

The implementation relocates the module, removes shims, enforces the no-fabricated-output failure contract (failed TIR → eager subgraph fallback), removes `_TORCH_FALLBACK_PATTERNS`, and passes the LLaMA-2-7B forward correctness test with SDPA as extern op. Trace reporting and CI smoke test are deferred.

### Allowed Choices

- Can use: `torch.library` for custom extern op registration, TVM's `from_fx` with `keep_params_as_input=True`, `FuseTransposeMatmul` Relax pass
- Can use: `@dataclass` or plain dict for ExternPolicy/CompileCapability (implementation choice)
- Cannot use: `_TORCH_FALLBACK_PATTERNS` or any pattern-matched TIR-name-based fallback
- Cannot use: `torch.empty()` as a placeholder for failed kernel outputs

## Feasibility Hints and Suggestions

> **Note**: This section is for reference and understanding only.

### Conceptual Approach

```
ExternPolicy:
  - A set of (target_qualname) tuples identifying ops that bypass TIR
  - Populated from: (1) permanent entries (SDPA), (2) auto-discovered ops not in TVM convert_map
  - Keyed by torch._ops.OpOverloadPacket identity or full qualname, not __name__

CompileCapability:
  - Inspects lowered TIR function buffer_map dtypes
  - Rejects functions with any non-float buffer (int64, bool, uint8)
  - Returns: compilable dict + pre_failed set

Failure contract:
  compile_subgraph_direct():
    1. Extract call sequence from Relax IR
    2. Separate extern stubs
    3. Pre-filter non-compilable functions (CompileCapability)
    4. Bulk TIR compile remaining functions
    5. If bulk fails, per-function probe → collect compilable + failed
    6. If any failed function has NO extern_op backing → return None (eager fallback)
    7. Otherwise, compile compilable batch, mark failed as torch_fallback with extern_op

No codegen path emits torch.empty() for unknown ops.
```

### Relevant References

- `tilelang/jit/torch_compile/analysis.py` — current FX lowering, extern op stubs, TIR compilation
- `tilelang/jit/torch_compile/codegen.py` — wrapper code generation, `_TORCH_FALLBACK_PATTERNS`
- `tilelang/jit/torch_compile/compiler.py` — compilation orchestration, caching, CUDA graphs
- `tilelang/jit/torch_compile/runtime.py` — compiled graph module execution
- `tilelang/jit/torch_compile/api.py` — public backend registration
- `tilelang/schedule/gpu/transpose.py` — transpose schedule rule (try/except fix preserved)
- `tests/end2end/bench_llama_pretrained.py` — existing pretrained benchmark script
- `tests/end2end/bench_llama_tileops_attn.py` — tileops attention benchmark

## Dependencies and Sequence

### Milestones

1. **Module Relocation**: Move `tilelang/jit/torch_compile/` → `tilelang/torch_compile/`
   - Phase A: Copy module to new location, update internal imports
   - Phase B: Delete old `tilelang/jit/{backend,graph,codegen,runtime}.py` shims
   - Phase C: Update `tilelang/__init__.py` and backend registration
   - Phase D: Verify all existing tests pass

2. **Mechanism Refactor**: Replace ad-hoc mechanisms
   - Phase A: Implement ExternPolicy (replaces `_ALWAYS_EXTERN`, `_infer_failed_op_name`)
   - Phase B: Implement CompileCapability (replaces non-float pre-filter)
   - Phase C: Enforce failure contract — remove `_TORCH_FALLBACK_PATTERNS`, `_fallback_outputs` bridge, and `torch.empty()` fabrication
   - Phase D: Add structured trace counts (`n_compiled`, `n_extern`, `n_fallback_eager`)

3. **LLaMA-2-7B E2E**: Correctness benchmark
   - Phase A: Clean benchmark script with native HF SDPA as extern op
   - Phase B: Forward correctness test (assert_close)
   - Phase C: Token-match generation test (greedy decode, fixed seed/prompt)
   - Phase D: CI smoke test with synthetic 7B-shape model (2 layers)

Milestone 1 is independent and can proceed first. Milestone 2 depends on Milestone 1 (new module location). Milestone 3 depends on Milestone 2 (correct failure contract needed for E2E).

## Task Breakdown

| Task ID | Description | Target AC | Tag | Depends On |
|---------|-------------|-----------|-----|------------|
| task1 | Analyze current import graph and all downstream references to tilelang.jit.{backend,graph,codegen,runtime} | AC-1 | analyze | - |
| task2 | Move tilelang/jit/torch_compile/ → tilelang/torch_compile/, update internal imports | AC-1 | coding | task1 |
| task3 | Delete tilelang/jit/{backend,graph,codegen,runtime}.py shims, update tilelang/__init__.py | AC-1 | coding | task2 |
| task4 | Verify all existing tests pass after relocation | AC-1 | coding | task3 |
| task5 | Analyze ExternPolicy design: identify all op-identity mechanisms in current code | AC-3a | analyze | task4 |
| task6 | Implement ExternPolicy with target/qualname identity, replace _ALWAYS_EXTERN and _infer_failed_op_name | AC-3a | coding | task5 |
| task7 | Implement CompileCapability check, replace non-float buffer pre-filter | AC-3b | coding | task5 |
| task8 | Remove _TORCH_FALLBACK_PATTERNS, _fallback_outputs bridge, torch.empty() fabrication; enforce eager-subgraph fallback | AC-2 | coding | task6, task7 |
| task9 | Analyze trace schema requirements for compilation composition reporting | AC-5 | analyze | task8 |
| task10 | Add n_compiled/n_extern/n_fallback_eager to GraphCompileTrace | AC-5 | coding | task9 |
| task11 | Write clean LLaMA-2-7B benchmark script (native HF SDPA, forward correctness, token-match) | AC-4 | coding | task8 |
| task12 | Add CI smoke test with synthetic 7B-shape model | AC-4.3 | coding | task11 |
| task13 | Run full pretrained LLaMA-2-7B E2E and verify correctness | AC-4.1, AC-4.2 | coding | task11 |

## Claude-Codex Deliberation

### Agreements
- Module relocation with shim removal is safe and appropriate
- The `torch.empty()` fabrication path is the highest correctness risk and must be removed
- `_TORCH_FALLBACK_PATTERNS` is legacy and should be removed entirely
- SDPA should be permanently extern (single optimized kernel, no fusion benefit)
- Non-float buffer pre-filter should become a named CompileCapability check
- FX graph canonicalization (_simplify_fx_graph, _break_fx_diamonds, _fold_scalar_inputs) is separate from op policy
- Mixed execution is valid for extern ops; only TIR compile failures without semantic fallback trigger eager subgraph fallback
- LLaMA benchmark should be manual/nightly; CI uses synthetic model

### Resolved Disagreements
- **Single OpPolicy vs 3 concerns**: Claude proposed a unified OpClassification; Codex argued for separation into FX lowering policy, TIR compile-capability, and graph canonicalization. Resolution: accept 3-concern split (matches actual code boundaries).
- **ExternPolicy identity**: Claude used op `__name__`; Codex required target/qualname identity. Resolution: accept qualname identity (prevents name collisions).
- **_TORCH_FALLBACK_PATTERNS scope**: Claude included it in OpPolicy; Codex identified it as TIR-function semantic fallback, not FX-op classification. Resolution: remove it entirely under the new failure contract (no more pattern-based fallback).
- **Unclassified from_fx failures**: Claude proposed catching specific exception types; Codex required catch-all → eager fallback as well. Resolution: accept both (try structured handling, catch-all as safety net).

### Convergence Status
- Final Status: `converged` (2 rounds, no REQUIRED_CHANGES remaining)

## Pending User Decisions

- DEC-1: LLaMA target path
  - Claude Position: Native HF SDPA as primary target
  - Codex Position: Tileops custom-op path is safer
  - Tradeoff Summary: SDPA tests general extern mechanism; tileops tests custom-op path specifically
  - Decision Status: **Native HF SDPA** (user decided)

- DEC-4: Benchmark CI placement
  - Claude Position: Manual benchmark + CI smoke test
  - Codex Position: Pretrained 7B should not be CI default
  - Decision Status: **Manual/nightly with synthetic CI smoke test** (user decided)

- DEC-5: SDPA extern permanence
  - Claude Position: Permanent
  - Codex Position: Need explicit decision
  - Decision Status: **Permanent extern** (user decided)

- DEC-6: Import shim deprecation
  - Claude Position: Keep with deprecation warning
  - Codex Position: Need explicit timeline
  - Decision Status: **Remove immediately** (user decided)

## Implementation Notes

### Code Style Requirements
- Implementation code and comments must NOT contain plan-specific terminology such as "AC-", "Milestone", "Step", "Phase", or similar workflow markers
- These terms are for plan documentation only, not for the resulting codebase
- Use descriptive, domain-appropriate naming in code instead

--- Original Design Draft Start ---

Now, we implement a new backend "tilelang" of torch.compile. We also meet the correctness problem when benchmarking llama-2. Its core logic is at `tilelang/jit/torch_compile`. We expect to put this module within the root package rather than jit submodule. And we also want to refactor our implementation, because it includes many adhoc implementation, like _TORCH_FALLBACK_PATTERNS in `tilelang/jit/torch_compile/codegen.py`, _ALWAYS_EXTERN in `tilelang/jit/torch_compile/analysis.py`. We need to reflect our current design firstly, and then refactor its mechanisms. And finally we want to benchmark llama-2-7b as a end2end test.

--- Original Design Draft End ---
