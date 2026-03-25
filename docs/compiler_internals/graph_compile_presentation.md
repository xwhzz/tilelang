# TileLang 图编译：从 PyTorch 到 CUDA Kernel

## 一、端到端流程

### 1.1 完整编译流水线

```
┌──────────────────────────────────────────────────────────────┐
│  @tilelang.jit(mode="graph")                                  │
│  def ffn(x, w1, w2):                                         │
│      return (F.silu(x @ w1) * (x @ w2)) @ w3                │
└──────────────┬───────────────────────────────────────────────┘
               │ ① Tracing: torch.export → Relax IR
               ▼
┌──────────────────────────────────────────────────────────────┐
│  Relax IRModule                                               │
│    main():                                                    │
│      lv0 = R.call_tir("matmul",  [x, w1])                   │
│      lv1 = R.call_tir("silu",    [lv0])                     │
│      lv2 = R.call_tir("matmul1", [x, w2])                   │
│      lv3 = R.call_tir("multiply",[lv1, lv2])                │
│      lv4 = R.call_tir("matmul2", [lv3, w3])                 │
└──────────────┬───────────────────────────────────────────────┘
               │ ② Relax Passes: FuseOps → FuseTIR
               ▼
┌──────────────────────────────────────────────────────────────┐
│  融合后的 TIR 函数（每个是一个独立 Kernel）                       │
│  • fused_matmul_silu(x, w1)           → Matmul + SiLU 融合   │
│  • fused_matmul_multiply(x, w2, lv1)  → Matmul + Multiply   │
│  • matmul2(lv3, w3)                   → 单独 Matmul          │
└──────────────┬───────────────────────────────────────────────┘
               │ ③ Schedule Rules: 按优先级匹配，变换为 tile-level IR
               ▼
┌──────────────────────────────────────────────────────────────┐
│  Scheduled TIR（T.copy, T.gemm, T.Parallel, pipeline ...）    │
└──────────────┬───────────────────────────────────────────────┘
               │ ④ Lower: LayoutInference → LowerTileOp → Codegen
               ▼
┌──────────────────────────────────────────────────────────────┐
│  CUDA Source → nvcc → CUBIN（每个融合 Kernel 一份）             │
└──────────────┬───────────────────────────────────────────────┘
               │ ⑤ GraphRunner: 预分配 Buffer，串联 Kernel 执行
               ▼
┌──────────────────────────────────────────────────────────────┐
│  运行时调度（Python 循环 / CUDA Graph / Native C Dispatch）    │
└──────────────────────────────────────────────────────────────┘
```

### 1.2 用户接口

```python
@tilelang.jit(mode="graph")
def swiglu_ffn(x, w_gate, w_up, w_down):
    gate = F.silu(x @ w_gate)
    up   = x @ w_up
    return (gate * up) @ w_down

# 第一次调用触发编译，后续调用直接执行
out = swiglu_ffn(x, w_gate, w_up, w_down)
```

支持两条路径：

| | `tilelang.jit(mode="graph")` | `tilelang.jit` (直接 DSL) |
|---|---|---|
| 用户写什么 | 普通 PyTorch 代码 | Tile 原语 (`T.copy`, `T.gemm`, ...) |
| 编译器做什么 | 自动拆图 → 匹配 Schedule → Lower → CUDA | Lower tile 原语 → CUDA |
| 适用场景 | 端到端自动优化 | 单 Kernel 极致调优 |

---

## 二、图编译流水线各阶段

### 2.1 阶段 ① Tracing：PyTorch → Relax IR

```python
# _build_relax_module() 内部流程：

# 1. torch.export 捕获计算图（静态追踪，无 Python 控制流）
exported = torch.export.export(wrapper_module, example_args, dynamic_shapes=...)

# 2. 自动发现 torch.library 注册的 custom op，创建 stub PrimFunc
custom_ops = _scan_custom_ops(exported.graph)
#    例：用户注册的 @torch.library.custom_op("my_lib::my_scale")
#    → 生成带 "torch_op" 属性的 stub TIR 函数，编译时跳过、运行时直接调用 PyTorch

# 3. 转为 Relax IR
mod = from_exported_program(exported, custom_convert_map=custom_ops)
```

**三种 Shape 处理策略**：

```python
# 策略 1: 静态形状（默认）—— 每种 shape 编译一次，缓存到 dict
@tilelang.jit(mode="graph")
def fn(a, b): return a + b

fn(torch.randn(128))   # 编译 shape=(128,)
fn(torch.randn(256))   # 编译 shape=(256,)，第二次缓存命中
fn(torch.randn(128))   # 缓存命中，不再编译

# 策略 2: 动态形状 —— 编译一次，符号化维度，运行任意 shape
@tilelang.jit(mode="graph", dynamic_dims={0: [0], 1: [0]})  # arg0 和 arg1 的 dim0 动态
def fn(a, b): return a + b

fn(torch.randn(128))   # 编译（带符号维度 s0）
fn(torch.randn(256))   # 复用同一 Kernel，不再编译
fn(torch.randn(1024))  # 复用

# 策略 3: Dynamo 回退 —— 自动处理数据依赖控制流
@tilelang.jit(mode="graph")
def fn(x):
    if x.sum() > 0:    # ← 数据依赖！torch.export 无法处理
        return x * 2
    return x * 0.5

# torch.export 失败 → 自动回退到 torch.compile + TileLang 后端
# torch.compile 在 graph break 处切分子图，每段子图走 TileLang 编译
```

**Dynamo 回退的完整流程**：

```
torch.compile(fn, backend=_tilelang_dynamo_backend)
    │
    ├── Dynamo 切分子图（在 graph break 处断开）
    │
    ├── 子图 1: x * 2.0
    │   └── from_fx → Relax → Schedule → tilelang.compile → mini GraphRunner
    │
    ├── 子图 2: x * 0.5
    │   └── from_fx → Relax → Schedule → tilelang.compile → mini GraphRunner
    │
    └── Python 控制流粘合各子图
```

### 2.2 阶段 ② Relax 变换 & 算子融合

```python
mod = relax.transform.LegalizeOps()(mod)         # 高层 Relax op → 底层 TIR 循环
mod = relax.transform.AnnotateTIROpPattern()(mod) # 标注算子模式（elemwise/broadcast/reduce）
mod = relax.transform.FoldConstant()(mod)         # 常量折叠
mod = relax.transform.FuseOps()(mod)              # 根据算子模式做融合决策
mod = relax.transform.FuseTIR()(mod)              # 将融合组合并为一个 TIR 函数
```

**融合规则**（基于 `OpPatternKind`）：

| 模式 | 含义 | 融合行为 |
|---|---|---|
| `kElemWise` | 逐元素 (relu, cast) | 可链式融合入前后任何算子 |
| `kBroadcast` | 广播 (add, multiply) | 同上 |
| `kInjective` | 单射 (reshape, transpose) | 可融合入 producer |
| `kOutReduction` | 归约输出 (matmul, conv) | 可作为 root，吸收后续 elemwise |
| `kOpaque` | 不透明 (custom op) | 不融合 |

**示例**：`matmul → silu → multiply` 融合后变成一个 TIR 函数 `fused_matmul_silu_multiply`。Matmul 是 root（kOutReduction），silu 和 multiply 是 elemwise 尾巴，被吸收进同一个 Kernel。

### 2.3 阶段 ③ Schedule Rules

见第三节详细展开。

### 2.4 阶段 ④ Lowering

Scheduled TIR 中的 tile 原语（T.copy, T.gemm, T.reduce, T.Parallel）经过三阶段 lowering 变成 CUDA 代码：

```
Phase 0  语义检查        验证并行循环、Fragment 访问合法性
Phase 1  Lower+Legalize  LayoutInference → LowerTileOp → LoopPartition
Phase 2  目标优化         软件流水线注入 → 向量化 → 存储优化 → Codegen
```

### 2.5 阶段 ⑤ GraphRunner 运行时

```python
class GraphRunner:
    def __init__(self, mod, kernels, calls, input_names, device):
        # 预分配所有中间 tensor——一次性分配，零运行时开销
        for call in calls:
            shape, dtype = _output_buffer_info(mod[call.gv_name])
            self._call_outputs.append(torch.empty(shape, device=device))

    def _run_kernels(self, *args):
        # 执行计划：按 Relax main 中的 call 顺序依次调度
        env = dict(zip(self.input_names, args))
        for idx, call in enumerate(self.calls):
            kernel = self.kernels[call.gv_name]
            inputs = [env[n] for n in call.arg_names]
            result = kernel(*inputs, self._call_outputs[idx])
            env[call.out_name] = result
        return env[self.calls[-1].out_name]
```

**执行计划提取**：从 Relax `main` 函数的 SeqExpr 中解析出有序的 `CallRecord` 列表：

```python
# Relax main 函数结构：
#   lv0 = call_tir("fused_matmul_silu", [x, w1])
#   lv1 = call_tir("fused_matmul_multiply", [x, w2, lv0])
#   lv2 = call_tir("matmul2", [lv1, w3])
#   return lv2
#
# 提取为：
calls = [
    CallRecord(out="lv0", gv="fused_matmul_silu",     args=["x", "w1"]),
    CallRecord(out="lv1", gv="fused_matmul_multiply",  args=["x", "w2", "lv0"]),
    CallRecord(out="lv2", gv="matmul2",                args=["lv1", "w3"]),
]
# GraphRunner 按此顺序串行调度，用 env dict 传递中间结果
```

**三种执行模式**：

| 模式 | 机制 | 调度开销 | 适用场景 |
|---|---|---|---|
| **Python 循环** | 逐个调用 `kernel(...)` | ~10 μs/kernel | 开发调试 |
| **CUDA Graph** | 录制后整体 replay | ~1 μs 总计 | 推理部署（固定 shape） |
| **Native C** | 编译为 C host 函数 | ~2 μs/kernel | 无 Python GIL 场景 |

```python
# CUDA Graph 模式
runner = swiglu_ffn._cache[sig]
runner.enable_cuda_graph(warmup_iters=3)

# 第一次调用：warmup + 录制
out = swiglu_ffn(x, w_gate, w_up, w_down)

# 后续调用：仅 copy 输入 → replay → 读输出（~1 μs）
out = swiglu_ffn(x, w_gate, w_up, w_down)
```

**Custom Kernel 注入**：用户可在任意环节用手写 Kernel 替换自动生成的：

```python
@tilelang.jit(mode="graph", custom_kernels={"matmul": my_hand_tuned_matmul})
def fn(x, w):
    return x @ w  # 此 matmul 使用用户提供的 Kernel，跳过自动 Schedule
```

---

## 三、Schedule 原语与模板规则

### 3.1 Tile-Level 编程模型

TileLang 的核心思想：在 thread-level 编程之上引入 **tile-level（协作式）操作层**。

```
┌───────────────────────────────────────────────────┐
│  Thread-Level（每个线程独立执行）                      │
│                                                     │
│  for i in range(N):           ← 普通 for 循环       │
│      local_buf[i] = ...       ← local scope (寄存器) │
├───────────────────────────────────────────────────┤
│  Tile-Level（所有线程协作完成）                        │
│                                                     │
│  for i in T.Parallel(N):      ← 工作自动分配给线程    │
│      fragment[i] = ...        ← local.fragment scope │
│                                                     │
│  T.copy(global → shared)      ← 协作搬运数据         │
│  T.gemm(A_sh, B_sh, C_frag)  ← 协作矩阵乘          │
│  T.reduce(in_frag, out_frag)  ← 协作归约             │
└───────────────────────────────────────────────────┘
```

**Schedule 原语的作用**：将 loop-level TIR（标量循环）变换为 tile-level TIR（协作 tile 操作）。编译器后端（LayoutInference + LowerTileOp）再自动将 tile 操作展开为 thread 级代码。

### 3.2 GPU 内存层次 & Buffer Scope

```
                            ┌──────────────┐
                            │   global     │  设备 DRAM (HBM)
                            │  Input/Output│  带宽: ~3 TB/s (H100)
                            └──────┬───────┘
                                   │ T.copy (TMA / SIMT)
                            ┌──────▼───────┐
                            │  shared.dyn  │  共享内存 / L1 Cache
                            │  Tile 暂存    │  容量: 96-228 KB/SM
                            └──────┬───────┘
                                   │ T.gemm / T.copy / T.reduce
                            ┌──────▼───────┐
                            │local.fragment│  寄存器（分布式）
                            │ 累加器/操作数  │  tile 视角的 shape
                            └──────┬───────┘
                                   │ LayoutInference + LoopPartition
                            ┌──────▼───────┐
                            │    local     │  寄存器（每线程私有）
                            │ 标量 / 循环变量│  thread 视角的 shape
                            └──────────────┘
```

**`local.fragment` 的关键特性**：
- shape 从 **block 的视角** 描述，如 `(128, 128)` 的累加器 tile
- 数据通过 **Fragment Layout** 物理分布到每个线程的寄存器
- 例：128×128 fragment，128 threads → 每 thread 持有 128 个寄存器值
- 编译器通过 `Fragment.Inverse()` 自动推导每个线程应处理哪些元素

### 3.3 TileSchedule 原语详解

#### 线程与网格控制

| 原语 | 作用 | 示例 |
|---|---|---|
| `launch_thread(root, N)` | 设定 Kernel 线程数 | `launch_thread(root, 256)` |
| `bind(loop, thread_axis)` | 绑定循环到 GPU 维度 | `bind(bx, "blockIdx.x")` |
| `parallel(loop)` | 标记为 T.Parallel | `parallel(inner)` → 线程间分工 |

#### 内存调度原语

**`cache_read_at(loop, block, buf_idx, scope)`** — 在指定循环层级插入数据预取

```
变换前:                              变换后:
for bx:                             for bx:
  for inner:                          A_frag = alloc("local.fragment")
    ... = A[bx*tile + inner]          T.copy(A[bx*tile : (bx+1)*tile], A_frag)
                                      for inner:
                                        ... = A_frag[inner]
```

**`cache_write_at(loop, block, buf_idx, scope)`** — 输出 buffer 暂存到快速内存

```
变换前:                              变换后:
for bx:                             for bx:
  for inner:                          C_frag = alloc("local.fragment")
    C[bx*tile + inner] = ...          for inner:
                                        C_frag[inner] = ...
                                      T.copy(C_frag, C[bx*tile : (bx+1)*tile])
```

特殊参数：
- `reduce_type="sum"` — 写回时插入跨线程归约（用于 GEMV）
- `reducer_replication="all"` — 每线程持有完整累加值，AllReduce 后写回

**`cache_reduce_at(loop, block, buf_idx, scope, init_value)`** — 三合一归约原语

等价于 `alloc + fill(init_value) + compute + reduce_writeback`：
```
for bx:
  accum = alloc("local.fragment")
  T.fill(accum, 0.0)              ← 初始化
  for k_outer:
    for k_inner in parallel:
      accum += input[...]          ← 本地累加
  T.reduce(accum, output[bx])     ← 跨线程归约 + 写回
```

#### Tile 操作原语

**`gemm_at(loop, block)`** — 替换 matmul 循环为 tile 级 T.gemm

```
变换前:                              变换后:
for i, j in parallel(M*N):         T.gemm(A_shared, B_shared, C_fragment,
  for k:                                   transpose_A=False, transpose_B=False)
    C[i,j] += A[i,k] * B[k,j]
```

T.gemm 在 lower 阶段映射到 WMMA (Ampere) 或 WGMMA (Hopper) 指令。

**`reduce_at(loop, block, ...)`** — 替换归约循环为 tile 级 T.reduce

```
变换前:                              变换后:
for i in parallel(N):              T.reduce(input_fragment, output_fragment,
  for k:                                    reduce_type="sum", dim=0)
    out[i] += in[i, k]
```

#### 优化控制

**`pipeline(loop, stages)`** — 标注软件流水线

```python
pipeline(k_outer, num_stages=2)
# 效果：load[k+1] 与 compute[k] 并行执行（双缓冲）
# Hopper: stages=4 配合 TMA async copy
# Ampere: stages=2 配合 cp.async
```

**`annotate_layout(block, buf_name, layout)`** — Swizzle 布局

```python
annotate_layout(root, "A_shared", make_swizzled_layout(128))
# XOR-based 索引重排，消除 shared memory bank conflict
# 128B / 64B / 32B 三种粒度，根据 element size 自动选择
```

### 3.4 Schedule 规则体系

#### 匹配优先级（first-match-wins）

```python
default_schedule_rules() = [
    Matmul(),           # ① 矩阵乘法（Tensor Core 加速）
    GEMV(),             # ② 矩阵-向量乘（reduction 并行化）
    GeneralReduction(), # ③ 通用归约（softmax, layernorm, RMSNorm ...）
    Transpose(),        # ④ 转置（shared memory 合并访存）
    ElementWise(),      # ⑤ 逐元素（fragment 缓存）
    Fallback(),         # ⑥ 兜底（最简并行化）
]
```

每条规则实现 `apply(func, target) → Schedule | None`。对 IRModule 中每个 TIR 函数，依次尝试，第一个返回非 None 者胜出。Fallback 永远匹配，保证所有函数都被调度。

#### 3.4.1 Matmul 规则

**匹配**：reduction block，2 read + 1 write buffer，dtype 支持 Tensor Core (fp16/bf16/int8/fp32)。

**变换过程**：

```
原始 TIR:                               Scheduled TIR:
                                        for by in bind("blockIdx.y"):          # ← M tiles
for i, j, k:                             for bx in bind("blockIdx.x"):        # ← N tiles
  C[i,j] += A[i,k] * B[k,j]               C_frag = cache_write_at(            # ← 累加器
                                                bx, "local.fragment")
                                            for ko in pipeline(K/Bk, stages):  # ← K 分块
                                              A_sh = cache_read_at(             # ← A tile → smem
                                                  ko, "shared.dyn")
                                              B_sh = cache_read_at(             # ← B tile → smem
                                                  ko, "shared.dyn")
                                              gemm_at(ko)                       # ← T.gemm
                                            # epilogue (cast/relu 融合到写回)
```

**详细步骤**：

1. `normalize_to_matmul()` — 标准化循环结构为 [batch, M, N, K]
2. 按配置 split：M→[M_o, M_i], N→[N_o, N_i], K→[K_o, K_i]
3. `cache_write_at(N_o, block, "local.fragment")` — fragment 累加器
4. `pipeline(K_o, stages=2~4)` — K 维度软件流水线
5. `cache_read_at(K_o, ..., "shared.dyn")` — A、B tile 搬到共享内存
6. `gemm_at(K_o, block)` — 内层循环替换为 T.gemm
7. Epilogue 处理：若后接 silu/relu/cast，通过 `reverse_compute_at` 融合到累加器写回中
8. `bind(M_o, "blockIdx.y") + bind(N_o, "blockIdx.x") + launch_thread(root, 128~256)`

**架构适配**：

| 架构 | Copy 机制 | Pipeline Stages | GEMM 指令 | 典型 Tile |
|---|---|---|---|---|
| Hopper (sm_90) | TMA (硬件异步) | 4 | WGMMA 64×64 | 128×128×32 |
| Ampere (sm_80) | cp.async (SIMT) | 2 | WMMA 16×16 | 128×128×32 |
| Turing (sm_75) | SIMT copy | 2 | WMMA 16×16 | 64×64×32 |

#### 3.4.2 GEMV 规则

**匹配**：inner-reduction 结构，一个 2D 矩阵 buffer + 一个 1D/2D 向量 buffer。

**变换过程**：

```
原始:                                Scheduled:
for i:                               for bx in bind("blockIdx.x"):       # ← spatial
  for k:                               for ko in range(K / block_k):     # ← reduction 外层
    y[i] += A[i,k] * x[k]                A_frag = cache_read_at(ko, A,   # ← 向量化 load (float4)
                                              "local.fragment")
                                          x_frag = cache_read_at(ko, x,
                                              "local.fragment")
                                          for ki in parallel(block_k):    # ← 线程分担 reduction
                                            local_accum += A_frag * x_frag
                                        y_frag = cache_write_at(bx, y,
                                            "local.fragment",
                                            reduce_type="sum",            # ← 跨线程 AllReduce
                                            reducer_replication="all")
```

**关键优化**：
- **向量化 Load**：`cache_read_at` 对矩阵 A 使用 float4 / half8 向量加载，最大化带宽利用
- **Streaming Cache**：编译参数 `-dlcm=cs` 开启 evict-first 策略，大矩阵 A 只经过 L2 不驻留，避免污染缓存（H100 上提升 ~20%）
- **AllReduce 写回**：`reducer_replication="all"` 使每线程持有完整累加结果，通过 warp shuffle 实现跨线程归约

#### 3.4.3 GeneralReduction 规则

**匹配**：包含 reduction block 的计算图——softmax、layernorm、RMSNorm 等。

**单步归约**（sum, max, min）：

```
原始:                                Scheduled:
for i:                               for bx in bind("blockIdx.x"):
  for k:                               accum = cache_reduce_at(bx,         # ← 三合一
    out[i] += in[i,k]                      "local.fragment", init=0.0)
                                        for ko in range(K / chunk):
                                          in_frag = cache_read_at(ko,
                                              "local.fragment")
                                          reduce_at(ko, reduce_type="sum",  # ← tile 级归约
                                              replace_loop_body=True)
                                        # cache_reduce_at 自动处理写回
```

**两步归约链**（softmax = max → subtract → exp → sum → divide）：

```
                    ┌─────────────────────────────────────┐
                    │         Softmax 计算图                │
                    │                                       │
                    │  Reduction 1: max_val = max(x, dim=1) │
                    │       ↓                               │
                    │  Bridge: exp_x = exp(x - max_val)     │ ← injective 变换
                    │       ↓                               │
                    │  Reduction 2: sum_val = sum(exp_x)    │
                    │       ↓                               │
                    │  Output: exp_x / sum_val              │
                    └─────────────────────────────────────┘

Schedule 策略：
1. 以 Output block 为 spatial anchor，bind(bx, "blockIdx.x")
2. compute_at(reduction1, bx) + compute_at(reduction2, bx)
3. Reduction 1 (max):
   - cache_reduce_at(bx, "local.fragment", init=-inf)
   - reduce_at(ko, reduce_type="max")
4. Bridge (exp): cache_read_at → fragment 暂存
5. Reduction 2 (sum):
   - cache_reduce_at(bx, "local.fragment", init=0.0)
   - reduce_at(ko, reduce_type="sum")
6. Output (divide): parallel 写回
```

**关键识别逻辑**：`_should_preserve_two_reduction_bridge()` 自动检测两个 reduction 之间是否有 injective 变换（如 exp），并保留该 bridge 以确保数据流正确。

#### 3.4.4 Transpose 规则

**匹配**：2D block，写入索引是读取索引的排列（`store[i,j] = load[j,i]`）。

**变换过程**：

```
原始:                                Scheduled:
for i in range(M):                   for bi in bind("blockIdx.x"):
  for j in range(N):                   for bj in bind("blockIdx.y"):
    B[j,i] = A[i,j]                     A_shared = cache_read_at(bj, A,
                                             "shared.dyn", disable_tma=True)
                                         annotate_layout(root, "A_shared",
                                             swizzle_layout)           # ← 消除 bank conflict
                                         for ii in parallel(tile_i):
                                           for jj in parallel(tile_j):
                                             B[...] = A_shared[...]
```

**为什么用 shared memory？**
- 直接转置：读合并但写不合并（或反之），DRAM 带宽浪费一半
- 经 shared memory 中转：读合并 → 存 shared → 取 shared → 写合并，两端都合并
- `disable_tma=True`：SIMT copy + Swizzle 比 TMA 更适合转置场景

#### 3.4.5 ElementWise 规则

**匹配**：单个 injective block，全部 Spatial 循环。

```
原始:                                Scheduled:
for i in range(M):                   for bx in bind("blockIdx.x"):
  for j in range(N):                   frag_A = cache_read_at(bx, A, "local.fragment")
    C[i,j] = relu(A[i,j] + B[i,j])    frag_B = cache_read_at(bx, B, "local.fragment")
                                        frag_C = cache_write_at(bx, C, "local.fragment")
                                        for inner in parallel(tile):
                                          frag_C[inner] = relu(frag_A[inner] + frag_B[inner])
```

**后缀积规则（Suffix-Product Rule）**：

多维 tensor 融合后分 tile 时，tile 边界可能跨越原始维度边界，导致 fragment 访问模式不可分析。

```
示例：shape = (3, 8, 4)，fused extent = 96

tile = 32:
  后缀检查: 4 % 32 ≠ 0, 8*4=32 % 32 = 0 ✓  → 可用 fragment
  （tile 对齐到最后两维的乘积边界）

tile = 48:
  后缀检查: 4 % 48 ≠ 0, 32 % 48 ≠ 0         → 不可用 fragment
  （tile 跨越维度边界，fragment 无法推导）
  → 退化为无 fragment 的简单并行化
```

#### 3.4.6 Fallback 规则

兜底：对所有剩余 TIR 函数，做最简并行化——fuse spatial loops → split → bind(blockIdx.x) + bind(threadIdx.x)。不使用 fragment，中间结果暂存到 shared memory，通过 `reverse_compute_at` 将 consumer 拉入同一 block。

### 3.5 规则总结对比

| 规则 | 匹配模式 | 核心原语 | 内存层次 | 关键优化 |
|---|---|---|---|---|
| **Matmul** | SSR + tensorizable | `gemm_at`, `pipeline` | shared.dyn + fragment | TMA/WMMA, Swizzle, 多 stage 流水线 |
| **GEMV** | 向量-矩阵乘 | `cache_read_at`, `parallel` | fragment | Streaming cache, 向量化 load, AllReduce |
| **GeneralReduction** | 含 reduction block | `cache_reduce_at`, `reduce_at` | fragment | 两步归约链识别, 跨线程归约 |
| **Transpose** | 转置访存 | `cache_read_at`, `annotate_layout` | shared.dyn | Swizzle 消除 bank conflict |
| **ElementWise** | 纯 Spatial | `cache_read_at`, `cache_write_at` | fragment | 后缀积规则 |
| **Fallback** | 通配 | `bind` | shared | 最简并行化 |

---

## 四、可观测性与调试

### 4.1 编译 Trace

```python
@tilelang.jit(mode="graph")
def my_fn(x, w):
    return F.silu(x @ w)

out = my_fn(x, w)
```

**查看编译摘要**：

```python
my_fn.show_trace()
```
```
Graph Compilation Trace
  path: export, arch: sm_90, dynamic: False
  timing: trace=1040ms, schedule=306ms, compile=16841ms, total=18190ms
  kernels (1):
    fused_matmul_silu: [Matmul] → float32[32, 1024] (5654ms)
```

**查看 Schedule 前后的 TIR 对比**：

```python
trace = my_fn.get_trace()
trace.show_tir("fused_matmul_silu")
```
```
--- Unscheduled TIR ---
@T.prim_func
def main(args_0: T.Buffer((32, 256), "float32"),
         args_1: T.Buffer((256, 1024), "float32"),
         T_multiply: T.Buffer((32, 1024), "float32")):
    matmul_buf = T.alloc_buffer((32, 1024))
    for i0, i1, k in T.grid(32, 1024, 256):       # ← 标量循环
        with T.block("matmul"):
            matmul_buf[i0, i1] += args_0[i0, k] * args_1[k, i1]
    for i0, i1 in T.grid(32, 1024):
        T_multiply[i0, i1] = matmul_buf[i0, i1] * T.sigmoid(matmul_buf[i0, i1])

--- Scheduled TIR ---
@T.prim_func
def main(...):
    T.func_attr({"tir.is_scheduled": True})
    for tx in T.thread_binding(128, thread="threadIdx.x"):
      for by in T.thread_binding(1, thread="blockIdx.y"):       # ← M 维
        for bx in T.thread_binding(16, thread="blockIdx.x"):    # ← N 维 (16 blocks)
          C_frag = T.alloc_buffer((32, 64), scope="local.fragment")
          for ko in T.serial(8, annotations={"num_stages": 2}): # ← K 维 (流水线)
            B_shared = T.alloc_buffer((32, 64), scope="shared.dyn")
            T.copy(args_1[ko*32, bx*64], B_shared)              # ← tile 搬运
            A_shared = T.alloc_buffer((32, 32), scope="shared.dyn")
            T.copy(args_0[0, ko*32], A_shared)
            for ij in T.parallel(2048):                          # ← tile 级 GEMM
              for k in range(32):
                C_frag[...] += A_shared[...] * B_shared[...]
          for i, j in T.parallel(...):                           # ← epilogue: silu 融合
            T_multiply[i, bx*64+j] = C_frag[i,j] * T.sigmoid(C_frag[i,j])
```

**查看生成的 CUDA 源码**：

```python
sources = my_fn.get_kernel_sources()
for name, cuda_src in sources.items():
    print(f"=== {name} ===")
    print(cuda_src[:500])
```
```
=== fused_matmul_silu ===
#include <tl_templates/cuda/gemm.h>
extern "C" __global__ void __launch_bounds__(256, 1)
fused_matmul_silu_kernel(float* T_multiply,
    __grid_constant__ const CUtensorMap args_0_desc,
    __grid_constant__ const CUtensorMap args_1_desc) {
  extern __shared__ __align__(1024) uchar buf_dyn_shmem[];
  float matmul_reindex_local_fragment[16];
  __shared__ __align__(16) uint64_t mbarrier_mem[4];
  ...
```

### 4.2 编译日志

```python
import logging
logging.getLogger("tilelang.jit.graph").setLevel(logging.INFO)
```

```
Graph compile: tracing via torch.export ...
Graph compile: tracing done (1040.1 ms).
Schedule rule matching:
  fused_matmul_silu                        → Matmul
  fused_matmul_multiply                    → Matmul
  matmul2                                  → Matmul
Graph compile: scheduling done (305.5 ms).
  fused_matmul_silu                        [Matmul] 5654 ms
  fused_matmul_multiply                    [Matmul] 5413 ms
  matmul2                                  [Matmul] 5773 ms
Graph compile: total 18190 ms (3 kernels).
```

### 4.3 Programmatic Trace Access

```python
trace = my_fn.get_trace()

# 编译路径和时间
trace.compilation_path   # "export" | "dynamo" | "export+dynamo_fallback"
trace.arch               # "sm_90"
trace.total_time_ms      # 18190.0

# 遍历每个 Kernel 的详细信息
for kt in trace.kernels:
    print(f"{kt.name}: rule={kt.schedule_rule}, "
          f"shape={kt.output_shape}, dtype={kt.output_dtype}, "
          f"compile={kt.compile_time_ms:.0f}ms")
    # 访问 TIR（惰性序列化，不访问不付出 str() 开销）
    # kt.unscheduled_tir, kt.scheduled_tir
```

---

## 五、性能实测

> 测试平台：NVIDIA H100 PCIe，对比基线为 `torch.compile`（PyTorch inductor / cuBLAS）。
> 带宽基准：H100 PCIe 理论 HBM 带宽 ~2 TB/s。

### 5.1 ElementWise（25 种 shape）

| Shape | Fragment | TileLang (GB/s) | torch.compile (GB/s) | Speedup |
|---|---|---|---|---|
| (64, 128, 512) | Yes | 1,500 | 1,488 | **1.008x** |
| (128, 128, 128) | Yes | 1,272 | 1,254 | **1.015x** |
| (1, 1024, 1024) | No | 1,008 | 971 | **1.038x** |
| (64, 128, 1024) | Yes | 1,637 | 1,609 | **1.017x** |
| (32, 256, 2048) | Yes | 1,750 | 1,726 | **1.014x** |
| (64, 128, 16384) | Yes | 1,898 | 1,869 | **1.016x** |
| (1, 100, 8192) | Yes | 918 | 848 | **1.082x** |
| (10, 10, 8192) | Yes | 908 | 856 | **1.061x** |
| (1, 4096, 4096) | No | 1,750 | 1,723 | **1.016x** |

**汇总**：25 shape 全部通过正确性验证，平均 speedup **0.977x**，其中 **15/25 shape 优于 torch.compile**。

**分析**：
- Fragment 对齐的大 shape（满足后缀积规则）：**1.01-1.08x**，fragment 缓存有效
- 非对齐 shape 或极小 shape：~0.80-0.96x，退化为无 fragment 模式，kernel launch 开销占比高
- 接近 H100 理论带宽上限（1,898 GB/s ≈ 95% HBM 峰值）

### 5.2 Reduction

**简单归约**（shape 1024×16×2048, fp32, reduce over dim=2）：

| | TileLang | torch.compile | Speedup |
|---|---|---|---|
| max-reduction | 0.091 ms (1,473 GB/s) | 0.080 ms (1,673 GB/s) | 0.88x |
| sum-reduction | 0.091 ms (1,473 GB/s) | 0.081 ms (1,659 GB/s) | 0.89x |

**GeneralReduction**（复合归约模式）：

| 模式 | TileLang (ms) | torch (ms) | Speedup | 说明 |
|---|---|---|---|---|
| sum + ReLU (3D) | 0.560 | 0.584 | **1.043x** | 归约 + 逐元素 epilogue |
| max + bias + ReLU | 0.560 | 0.584 | **1.042x** | 多输入 epilogue |
| keepdim + sigmoid | 0.560 | 0.585 | **1.044x** | keepdim 归约 |
| RMSNorm-like | 0.600 | 0.592 | 0.987x | 单步归约 + broadcast |
| LayerNorm-like | 0.598 | 0.590 | 0.988x | 两步归约链 |
| Softmax-like | 0.671 | 0.613 | 0.913x | max + sum(exp) |
| LogSumExp-like | 0.316 | 0.298 | 0.942x | 两步 + 标量输出 |

**分析**：
- 单步归约 + epilogue 融合：**1.04x**，归约与后处理融合在同一 kernel 中
- 两步归约链（softmax, layernorm）：0.91-0.99x，与高度优化的 PyTorch fused kernel 接近
- 简单归约略逊于 torch（0.88x），因为 torch.compile 的 reduction kernel 已针对单步归约深度优化

### 5.3 GEMV（矩阵-向量乘）

**fp32**：

| Shape (M×N) | TileLang (GB/s) | torch.compile (GB/s) | Speedup |
|---|---|---|---|
| 1024×1024 | 407 | 348 | **1.169x** |
| 4096×4096 | 1,243 | 1,360 | 0.914x |
| 4096×8192 | 1,472 | 1,627 | 0.905x |
| 8192×4096 | 1,474 | 1,498 | 0.984x |
| 8192×8192 | 1,695 | 1,752 | 0.967x |
| 16384×4096 | 1,695 | 1,580 | **1.073x** |
| 4096×16384 | 1,661 | 1,713 | 0.970x |

**fp16**（Streaming Cache 优化）：

| Shape (M×N) | TileLang (GB/s) | torch.compile (GB/s) | Speedup |
|---|---|---|---|
| 1024×1024 | 241 | 241 | 1.001x |
| 4096×4096 | 1,094 | 901 | **1.214x** |
| 4096×8192 | 1,241 | 1,155 | **1.075x** |
| 8192×4096 | 1,237 | 1,005 | **1.231x** |
| 8192×8192 | 1,472 | 1,251 | **1.177x** |
| 16384×4096 | 1,473 | 1,070 | **1.376x** |
| 4096×16384 | 1,471 | 1,368 | **1.075x** |

**分析**：
- **fp16 全面领先**，平均 **1.16x**，最高达 **1.38x**（16384×4096）
- 核心优势来自 **Streaming Cache（`-dlcm=cs`）**：大矩阵 A 采用 evict-first 策略，只经过 L2 不驻留 L1，避免缓存污染。向量 x 常驻 L1 缓存，重复读取零开销
- fp32 在中等 shape 略逊于 cuBLAS，但在小 shape（1024²）和大 M（16384×4096）场景下胜出
- fp16 下 cuBLAS 的 GEMV 实现未充分利用 L2 缓存层次，TileLang 的显式缓存策略更优

### 5.4 性能总结

| 规则 | 典型 Speedup vs torch.compile | 优势场景 |
|---|---|---|
| **ElementWise** | 0.98x (avg), 1.01-1.08x (large aligned) | Fragment 对齐的大 tensor |
| **Reduction** | 1.04x (with epilogue) | 归约 + epilogue 融合 |
| **GeneralReduction** | 0.91-1.04x | 单步归约 + epilogue |
| **GEMV fp16** | **1.16x (avg)**, max 1.38x | L2 streaming cache 优化 |
| **GEMV fp32** | 0.97x (avg), max 1.17x | 小 shape / 大 M |

---

## 六、设计总结

| 设计原则 | 实现方式 |
|---|---|
| **Tile 级抽象** | Schedule 原语操作 tile（block 级别），编译器自动映射到 thread 级 |
| **Layout 驱动** | Fragment Layout 统一了 WMMA / WGMMA / SIMT 的寄存器分布，3 阶段 BFS 自动推断 |
| **模板匹配** | 6 条优先级规则，first-match-wins，每种计算模式有专门优化模板 |
| **渐进回退** | torch.export → dynamo fallback → Fallback rule，任何 PyTorch 代码都能编译 |
| **可观测性** | 每阶段可追踪：规则匹配、TIR 前后对比、CUDA 源码、per-kernel 编译耗时 |
| **零开销运行时** | 预分配所有中间 buffer，支持 CUDA Graph 录制 replay（~1 μs 总开销） |
