from tilelang import tvm
from tvm import te
import tilelang
from tvm.tir.stmt_functor import ir_transform
from tilelang.schedule import Schedule

shape = (6, 1024, 1024)

a = te.placeholder(shape, name="a")
b = te.placeholder(shape, name="b")
scale = te.placeholder(shape, name="scale")
c = te.compute(a.shape, lambda i, j, k: a[i, j, k] + b[i, j, k] * scale[i, j, k], name="c")
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

kernel = tilelang.compile(mod["main"], execution_backend="cython")

# print(kernel.get_host_source())
import torch
a = torch.randn(shape).cuda()
b = torch.randn(shape).cuda()
scale = torch.randn(shape).cuda()
c = torch.empty(shape).cuda()

kernel(a, b, scale, c)

ref_c = a + b * scale

torch.testing.assert_close(c, ref_c)

print("\033[92mTest passed!\033[0m")

from tilelang.profiler import do_bench

tilelang_time = do_bench(lambda: kernel(a, b, scale, c))
print(f"TileLang kernel time: {tilelang_time} ms")

# Compare with Torch kernel
torch_time = do_bench(lambda: a + b * scale)
print(f"Torch kernel time: {torch_time} ms")

print(f"Speedup: {torch_time / tilelang_time}x")

"""
Loading tilelang libs from dev root: /data/xwh/imp/tilelang/build
Test passed!
TileLang kernel time: 0.0636356920003891 ms
Torch kernel time: 0.08889956027269363 ms
Speedup: 1.397007834410759x
"""