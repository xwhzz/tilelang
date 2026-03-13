from tilelang import tvm
from tvm import te
import tilelang
from tilelang.schedule import Schedule

"""
Transpose Schedule Template
"""
M, N = 1024, 2048

a = te.placeholder((M, N), name="a")
c = te.compute((N, M), lambda i, j: a[j, i], name="c")
func = te.create_prim_func([a, c])

sch = Schedule(func)

main_block = sch.get_block("c")

i, j = sch.get_loops(main_block)

bx, ii = sch.split(i, factors=[None, 32])
by, jj = sch.split(j, factors=[None, 32])

sch.reorder(bx, by, ii, jj)

sch.cache_read_at(by, main_block, 0, "shared.dyn")
sch.cache_write_at(by, main_block, 0, "local.fragment")

sch.parallel(ii)
sch.parallel(jj)
sch.bind(bx, "blockIdx.x")
sch.bind(by, "blockIdx.y")
print(sch.mod)
root_block = sch.get_block("root")
sch.launch_thread(root_block, 256)
mod = sch.mod
print("=== Final scheduled mod ===")
print(mod)

mod = tvm.tir.transform.Simplify()(mod)

mod = tvm.tir.transform.ConvertBlocksToOpaque()(mod)

# mod = tvm.tir.transform.LowerOpaqueBlock()(mod)
mod = tilelang.transform.ReserveRootBlock()(mod)

print("=== After lowering passes ===")
print(mod)
kernel = tilelang.compile(mod["main"])

import torch

torch._dynamo.reset()  # 强制重置 Dynamo，确保重新触发编译
a = torch.randn((M, N)).cuda()

c = torch.empty((N, M)).cuda()
kernel(a, c)


@torch.compile()
def fntorch(a):
    return a.transpose(0, 1).contiguous()


ref_c = fntorch(a)

# ref_c = a.transpose(0, 1).contiguous()
print(c, ref_c, sep="\n")
torch.testing.assert_close(c, ref_c, rtol=1e-5, atol=1e-5)

from tilelang.profiler import do_bench

tilelang_time = do_bench(lambda: kernel(a, c), backend="cupti")
print(f"TileLang kernel time: {tilelang_time} ms")
# Compare with Torch kernel — .contiguous() forces an actual data copy
torch_time = do_bench(lambda: fntorch(a), backend="cupti")
print(f"Torch kernel time: {torch_time} ms")

print(f"Speedup: {torch_time / tilelang_time}x")
