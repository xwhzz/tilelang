"""Compile TIR functions from a Relax module using TileLang's compilation + cache."""

import logging

from tilelang import tvm as tvm
from tvm import tir, runtime
from tvm.target import Target

from tilelang.engine.phase import NormalizeScheduledIR

logger = logging.getLogger(__name__)


def compile_tir_functions(
    mod: tvm.IRModule,
    target: Target,
) -> dict[str, callable]:
    """Extract and compile all TIR functions from a Relax module.

    Uses ``tilelang.par_compile()`` for parallel kernel compilation
    with built-in kernel-level caching.

    Returns raw ``runtime.Executable`` objects (not JITKernel) to avoid
    per-call Python/DLPack adapter overhead — the TVM FFI accepts
    torch tensors directly at the C level.
    """
    names = []
    funcs = []
    for gvar, func in mod.functions.items():
        if not isinstance(func, tir.PrimFunc):
            continue

        name = gvar.name_hint
        if "global_symbol" not in func.attrs:
            func = func.with_attr("global_symbol", name)

        func_mod = tvm.IRModule({name: func})
        with tvm.transform.PassContext(opt_level=3), target:
            func_mod = NormalizeScheduledIR(func_mod)

        prim_func = list(func_mod.functions.values())[0]
        names.append(name)
        funcs.append(prim_func)

    if not funcs:
        return {}

    from tilelang.jit import par_compile
    jit_kernels = par_compile(funcs, target=target)

    # Extract raw Executable from each JITKernel to bypass
    # the Python adapter layer on every call.
    compiled = {}
    for name, kernel in zip(names, jit_kernels):
        exe = kernel.adapter.executable
        if exe is None:
            exe = runtime.Executable(kernel.artifact.rt_mod)
        compiled[name] = exe
        logger.debug("Compiled: %s", name)

    return compiled
