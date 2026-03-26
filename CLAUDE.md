# TileLang Project Guide

## What is TileLang?

TileLang is a **tile-based DSL** built on top of TVM's Tensor IR (TIR). It provides a Pythonic interface for writing high-performance GPU kernels using tile-level primitives (T.copy, T.gemm, T.reduce, T.fill, etc.) that map naturally to GPU hardware capabilities like shared memory, tensor cores, and TMA.

## Two Compilation Paths

### Path 1: Direct DSL (Manual Kernel Writing)
Write kernels directly using `@T.prim_func` with tile primitives:
```python
@T.prim_func
def kernel(A: T.Tensor(...), C: T.Tensor(...)):
    with T.Kernel(grid_x, grid_y, threads=128) as (bx, by):
        A_shared = T.alloc_shared((tile_M, tile_K), dtype)
        C_local = T.alloc_fragment((tile_M, tile_N), accum_dtype)
        T.clear(C_local)
        for k in T.Pipelined(K // tile_K, num_stages=3):
            T.copy(A[...], A_shared)
            T.gemm(A_shared, B_shared, C_local)
        T.copy(C_local, C[...])
```
Compiled via `tilelang.jit` or `tilelang.compile`.

### Path 2: Schedule Rules (Automatic Transformation) -- OUR FOCUS
Transform loop-level `te.compute` expressions into optimized tilelang programs via schedule rules:
```python
# 1. Define computation as te.compute
a = te.placeholder((M, N, K), name="a")
rk = te.reduce_axis((0, K), name="rk")
c = te.compute((M, N), lambda i, j: te.sum(a[i, j, rk], axis=rk), name="c")
func = te.create_prim_func([a, c])

# 2. Apply schedule rule (automatic transformation)
sch = Reduction().apply(func, target, None)

# 3. Lower and compile
mod = sch.mod
mod = tvm.tir.transform.Simplify()(mod)
# ... lowering passes ...
kernel = tilelang.compile(mod["main"])
```

## Schedule Rule Architecture

### Base Class
All rules extend `GPUScheduleRule` (from `tvm.dlight.ScheduleRule`):
```python
class MyRule(GPUScheduleRule):
    def apply(self, func, target, _) -> None | Schedule | list[Schedule]:
        ...
```

### Available Schedule Rules (priority order)
1. **Matmul** - Matrix multiplication (tensorized)
2. **GEMV** - Matrix-vector products (split-K fast path)
3. **Reduction** - Simple reductions (sum, max, min, bitwise)
4. **GeneralReduction** - Complex multi-step reductions
5. **Transpose** - Tensor transpose
6. **ElementWise** - Injective element-wise ops
7. **Fallback** - Generic catch-all

### TileSchedule Primitives (the schedule vocabulary)
These are the key schedule methods that transform loop-level IR into tile-level IR:

| Primitive | Purpose |
|-----------|---------|
| `sch.launch_thread(root_block, N)` | Set thread count for the kernel |
| `sch.bind(loop, "blockIdx.x")` | Bind loop to GPU grid dimension |
| `sch.parallel(loop)` | Mark loop for T.Parallel lowering |
| `sch.pipeline(loop, stages)` | Mark loop for T.Pipelined lowering |
| `sch.cache_read_at(loop, block, buf_idx, scope)` | Stage input tile into fast memory |
| `sch.cache_write_at(loop, block, buf_idx, scope)` | Stage output tile in fast memory |
| `sch.cache_reduce_at(loop, block, buf_idx, scope, init)` | Allocate + init + write-back accumulator |
| `sch.fill_at(loop, block, buf_idx, value)` | Initialize buffer with value |
| `sch.reduce_at(loop, block, ...)` | Replace loop body with tile-level T.reduce |
| `sch.gemm_at(loop, block, ...)` | Replace loop body with tile-level T.gemm |
| `sch.copy_at(loop, block, ...)` | Replace loop body with tile-level T.copy |
| `sch.annotate_layout(block, name, layout)` | Set swizzle layout for shared memory |

### Standard Schedule Pattern (ElementWise example)
```
1. normalize_prim_func(sch)     -- analyze block structure
2. try_inline(sch, block_infos) -- inline trivial blocks
3. Classify loops (S=spatial, R=reduction, O=opaque)
4. Fuse spatial loops, split into [blockIdx, inner_tile]
5. cache_read_at / cache_write_at into local.fragment
6. sch.parallel(inner) + sch.bind(bx, "blockIdx.x")
7. sch.launch_thread(root, num_threads)
```

### Standard Schedule Pattern (Reduction example)
```
1. normalize + try_inline_contiguous_spatial
2. Classify S-loops and R-loops, fuse each group
3. Split spatial → [blockIdx, inner_s(=1)]
4. Split reduction → [outer_chunk, inner_chunk]
5. cache_read_at(outer_chunk, ..., "local.fragment")  -- stage input
6. cache_reduce_at(blockIdx, ..., "local.fragment", init_val)  -- accumulator
7. reduce_at(outer_chunk, ..., replace_loop_body=True)  -- tile-level reduce
8. bind + launch_thread
```

## Execution Model: Thread-Level vs Tile-Level

TileLang's core idea is adding **tile-level (cooperative) operations** into thread-level programming.

- **Plain `for` loop (serial)**: Every thread executes the entire loop body independently. This is thread-level code — each thread does its own work.
- **`T.Parallel` / `sch.parallel(loop)`**: The work inside the loop is **partitioned across all threads** in the block. All threads cooperatively execute the operation together. This is tile-level code.

This distinction is fundamental to understanding memory scopes:

## Memory Hierarchy (CUDA)

| Scope | View Level | Description | Typical Use |
|-------|-----------|-------------|-------------|
| `global` | -- | Device DRAM | Input/output tensors |
| `shared.dyn` | Thread-block | Dynamic shared memory (L1) | Tiled input staging |
| `local.fragment` | **Thread-block** (tile-level) | Logically a tile owned by the whole block, physically distributed across thread registers | Accumulators, tiled inputs for T.copy/T.reduce/T.gemm |
| `local` | **Single thread** (thread-level) | Each thread's private memory | Per-thread scalar variables, thread-local storage |

**Key insight**: `local.fragment` shapes are written from the **thread-block's perspective** (e.g., a `(128, 128)` accumulator tile), but the data is physically partitioned across threads in registers. Tile-level operations (T.copy, T.gemm, T.reduce, T.fill) operate on fragments cooperatively. In contrast, `local` buffers have shapes from a **single thread's perspective** — each thread owns its own independent copy.

## Layout System

The layout system is the mechanism that bridges tile-level semantics and thread-level execution. It answers: **how is a logical tile distributed across threads and registers?**

### Layout = Affine Index Mapping

A `Layout` (`src/layout/layout.h`) is a pure function from logical indices to physical indices:

```
Layout {
  input_size_: shape of logical domain (e.g., [128, 128])
  forward_index_: output expressions as functions of input vars
}
```

- `Forward(vars)` — evaluate the mapping (logical → physical)
- `Inverse()` — compute the reverse mapping via TVM's `DetectIterMap` (physical → logical)
- `OutputShape()` — derive physical shape by analyzing forward_index ranges
- `Reshape(new_shape, rescale_num, rescale_den)` — change logical shape with dtype scaling

### Fragment = Layout + Thread Assignment

A `Fragment` extends Layout with thread distribution:

```
Fragment {
  forward_index_:  logical indices → register index (where in thread's registers)
  forward_thread_: logical indices → thread_id (which thread owns it)
  replicate_size_: copies per thread (1=unique, N=broadcast)
}
```

Example — an 8x8 GEMM warp fragment (`src/layout/gemm_layouts.cc`):
```
Input: [i ∈ [0,8), j ∈ [0,8)]
forward_thread = floor(j/2) + 4*i    → 32 threads
forward_index  = [j mod 2]            → 2 registers per thread
```
So the logical (8,8) tile maps to: 32 threads, each holding 2 elements.

Key Fragment operations:
- `Repeat(repeats, on_thread)` — tile to cover larger blocks. `on_thread=true` increases thread count; `on_thread=false` increases registers per thread.
- `Replicate(N)` — every thread gets a full copy (for broadcast values like masks)
- `DeReplicate()` — compress redundant replication
- `FullyReplicated(shape, threads)` — all threads hold the entire buffer

### How Layouts Drive Lowering

The complete flow:

```
1. Annotation:     T.annotate_layout() seeds kLayoutMap on blocks
2. LayoutReducer:  Converts local.reducer → local.fragment with Fragment layouts
3. LayoutInference: 3-phase BFS propagation assigns layouts to ALL buffers
   - Strict:  only explicit layouts from tile ops (T.gemm imposes WMMA layout)
   - Common:  BFS propagation through buffer use-def chains
   - Free:    relax constraints for remaining unresolved buffers
4. LowerTileOp:   Rewrites buffer shapes from logical (input) to physical (output)
                   Converts local.fragment scope → local scope
5. PartitionLoop:  Converts T.Parallel loops to serial per-thread loops via Inverse()
```

### Loop Partition = Layout Inversion (`src/op/loop_partition.cc`)

This is the key lowering step that turns cooperative tile work into per-thread serial code:

1. Given a `T.Parallel` loop and its inferred Fragment layout
2. Compute `Inverse()` — from `(serial_iters, thread_var)` → original loop vars
3. Generate serial loops over the Fragment's OutputShape
4. Insert guards for non-bijective mappings (when inverse isn't provably in-bounds)

```
Before: for i in parallel(64):       # 64 elements, cooperative
           fragment[i] = ...

After:  for i_local in serial(2):    # 2 elements per thread, serial
           if i_local * 32 + threadIdx.x < 64:
               local_buf[i_local] = ...
```

### Swizzle Layouts (Shared Memory Bank Conflicts)

For shared memory, `SwizzledLayout` uses XOR-based index reordering to avoid bank conflicts:
- Full (128B), Half (64B), Quarter (32B) swizzle granularities
- Selected automatically based on buffer alignment and element size
- `makeGemmABLayout()` chooses the optimal swizzle for GEMM operands

### GEMM Fragment Layouts (`src/layout/gemm_layouts.cc`)

Block-level GEMM fragments are composed hierarchically:
1. **Base warp tile**: 8x8 or 16x8 WMMA fragment (hardware-defined)
2. **Warp tiling**: `Repeat(on_thread=true)` — distribute across warps
3. **Block tiling**: `Repeat(on_thread=false)` — unroll within each thread

Example: `makeGemmFragmentC(block_m=128, block_n=128, warp_m=64, warp_n=64)`:
- Base: 16x8 → Repeat to 64x64 (on_thread) → Repeat to 128x128 (registers)
- Result: 128 threads, each holding multiple 16x8 tiles in registers

## Lowering Pipeline

### For tile-primitive schedules (Path 2):
Standard: `Simplify → LowerCrossThreadReduction → LowerInitBlock → PlanAndUpdateBufferAllocationLocation → ConvertBlocksToOpaque → UnifyThreadBinding → CompactBufferAllocation → Simplify → ReserveRootBlock`

**Critical ordering constraints:**
- `LowerCrossThreadReduction` MUST precede `LowerInitBlock` (needs T.init() block structure)
- `UnifyThreadBinding` MUST precede `CompactBufferAllocation` (GPU buffer compaction)

### For direct DSL kernels (Path 1):
Three phases: `PreLowerSemanticCheck → LowerAndLegalize → OptimizeForTarget`
(Handled automatically by `tilelang.compile`)

## Key APIs

```python
tilelang.compile(func)          # Compile PrimFunc → executable kernel
tilelang.jit(out_idx=[-1])      # JIT decorator for direct DSL
tilelang.disable_cache()        # Disable kernel caching (for benchmarking)
tilelang.register_cuda_postproc(fn)  # Modify CUDA source before nvcc
```

## Build & Test

```bash
cd /data/xwh/imp/tilelang/build && ninja   # Build C++ components
python -m pytest tests/reduction/           # Run reduction tests
python -m pytest tests/elementwise/         # Run elementwise tests
python -m pytest tests/gemv/                # Run GEMV tests
```

## Project Layout

```
tilelang/
├── language/          # DSL frontend (T.* primitives)
├── schedule/
│   ├── __init__.py    # TileSchedule class (extends tvm.tir.Schedule)
│   └── gpu/           # Schedule rules (ElementWise, Reduction, GEMV, ...)
├── engine/            # Compilation pipeline (lower.py, phase.py)
├── transform/         # Custom IR transform passes
├── jit/               # JIT compilation & kernel execution
├── layout/            # Python layout API
├── tileop/            # Tile operation implementations
├── intrinsics/        # Backend-specific intrinsics
└── profiler/          # Benchmarking utilities
src/
├── layout/
│   ├── layout.h/cc        # Layout & Fragment class hierarchy
│   ├── gemm_layouts.cc    # Concrete GEMM fragment layouts (WMMA, CDNA, Hopper)
│   ├── swizzle.h/cc       # Bank-conflict-free shared memory layouts
│   └── utils.h/cc         # Iterator analysis helpers
├── op/
│   ├── copy.cc            # T.copy lowering (SIMT, TMA, bulk copy)
│   ├── reduce.cc          # T.reduce lowering (local accum + AllReduce)
│   ├── fill.cc            # T.fill lowering
│   ├── loop_partition.cc  # Parallel loop → serial per-thread via Inverse()
│   └── parallel.cc        # ParallelOp, fragment containment proofs
├── transform/
│   ├── layout_inference.cc  # 3-phase BFS layout propagation
│   ├── layout_reducer.cc   # Reducer → fragment conversion
│   └── lower_tile_op.cc    # Fragment buffer shape rewriting
├── schedule/primitives/   # C++ schedule primitives (cache_read_at, etc.)
└── target/
    └── codegen_cuda.cc    # CUDA code generation
tests/                 # Test suites per operator type
examples/              # Example kernels (GEMM, FlashAttention, etc.)
3rdparty/tvm/          # Modified TVM submodule
```
