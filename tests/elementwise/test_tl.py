from tilelang import tvm
from tvm import te
import tilelang
from tvm.tir.stmt_functor import ir_transform

a = te.placeholder((16, 1024, ), name="a")
b = te.placeholder((16, 1024, ), name="b")

c = te.compute(a.shape, lambda i, j: a[i, j] + b[i, j], name="c")
func = te.create_prim_func([a, b, c])

sch = tvm.tir.Schedule(func)

loops = sch.get_loops(sch.get_block("c"))

fused_loop = sch.fuse(*loops)

outer, inner = sch.split(fused_loop, factors=[None, 16384])

sch.bind(outer, "blockIdx.x")

main_block = sch.get_block("c")
read_0 = sch.cache_read(main_block, 0, "shared.dyn")
read_1 = sch.cache_read(main_block, 1, "shared.dyn")
write_0 = sch.cache_write(main_block, 0, "local.fragment")

sch.compute_at(read_0, inner)
sch.compute_at(read_1, inner)
sch.reverse_compute_at(write_0, inner)
sch.parallel(inner)

mod = sch.mod
mod = tvm.tir.transform.ConvertBlocksToOpaque()(mod)
mod = tvm.tir.transform.CompactBufferAllocation()(mod)


mod = tvm.tir.transform.LowerOpaqueBlock()(mod)
mod = tilelang.transform.FakeLaunchThread()(mod)

kernel = tilelang.compile(mod["main"])
print(kernel.get_kernel_source())

import torch
a = torch.randn(16, 1024).cuda()
b = torch.randn(16, 1024).cuda()
c = torch.randn(16, 1024).cuda()

kernel(a, b, c)

ref_c = a + b

torch.testing.assert_close(c, ref_c)

print("\033[92mTest passed!\033[0m")
