from tvm import te
import tvm
import sys

sys.path.append("/home/xwh/project/imp/tvm/tilesch/python")

from tilesch.gpu import GEMV


a = te.placeholder(
    (
        1,
        16384,
    ),
    name="a",
)
b = te.placeholder(
    (
        1024,
        16384,
    ),
    name="b",
)
k = te.reduce_axis((0, 16384), name="k")
c = te.compute((1, 1024), lambda i, j: te.sum(a[i, k] * b[j, k], axis=k), name="c")
func = te.create_prim_func([a, b, c])
target = tvm.target.cuda()

sch = GEMV().apply(func, target, False)

print(sch.mod)
