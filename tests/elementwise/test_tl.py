from tilelang import tvm
from tvm import te
import tilelang
from tvm.tir.stmt_functor import ir_transform
from tilelang.schedule import Schedule

row = 1024
col = 1024

a = te.placeholder((row, col, ), name="a")
b = te.placeholder((row, col, ), name="b")
scale = te.placeholder((row, col, ), name="scale")

c = te.compute(a.shape, lambda i, j: a[i, j] + b[i, j] * scale[i, j], name="c")
func = te.create_prim_func([a, b, scale, c])

sch = Schedule(func)

main_block = sch.get_block("c")

loops = sch.get_loops(main_block)

fused_loop = sch.fuse(*loops)

outer, inner = sch.split(fused_loop, factors=[None, 16384])

sch.bind(outer, "blockIdx.x")

# TODO: enable cache read/write after supporting dynamic shared memory allocation

# read_0 = sch.cache_read(main_block, 0, "shared.dyn")
# read_1 = sch.cache_read(main_block, 1, "shared.dyn")
# write_0 = sch.cache_write(main_block, 0, "local.fragment")

# sch.compute_at(read_0, inner)
# sch.compute_at(read_1, inner)
# sch.reverse_compute_at(write_0, inner)

sch.parallel(inner)
# TODO: extend launch_thread schedule primitives
sch.launch_thread(1024) 

mod = sch.mod

mod = tvm.tir.transform.ConvertBlocksToOpaque()(mod)
mod = tvm.tir.transform.CompactBufferAllocation()(mod)
mod = tvm.tir.transform.LowerOpaqueBlock()(mod)

kernel = tilelang.compile(mod["main"])

import torch
a = torch.randn(row, col).cuda()
b = torch.randn(row, col).cuda()
scale = torch.randn(row, col).cuda()
c = torch.empty(row, col).cuda()


kernel(a, b, scale, c)

ref_c = a + b * scale

torch.testing.assert_close(c, ref_c)

print("\033[92mTest passed!\033[0m")
