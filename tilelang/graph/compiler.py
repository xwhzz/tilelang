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

    # For static-shape kernels, extract raw Executable to bypass the
    # Python adapter layer.  For dynamic-shape kernels, keep the
    # JITKernel which resolves symbolic vars from tensor shapes.
    compiled = {}
    for name, kernel, func in zip(names, jit_kernels, funcs):
        has_dynamic = any(
            isinstance(s, tir.Var)
            for p in func.params if p in func.buffer_map
            for s in func.buffer_map[p].shape
        )
        if has_dynamic:
            compiled[name] = kernel
        else:
            exe = kernel.adapter.executable
            if exe is None:
                exe = runtime.Executable(kernel.artifact.rt_mod)
            compiled[name] = exe
        logger.debug("Compiled: %s%s", name, " (dynamic)" if has_dynamic else "")

    return compiled
