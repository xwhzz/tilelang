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

    Static-shape kernels use ``par_compile`` + raw ``Executable``.
    Dynamic-shape kernels use ``tilelang.lower()`` directly, producing
    ``Executable`` objects that accept symbolic shape parameters.
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

    logger.info("Compiling %d TIR functions: %s", len(funcs), names)

    from tilelang.jit import par_compile
    jit_kernels = par_compile(funcs, target=target)

    compiled = {}
    for name, kernel in zip(names, jit_kernels):
        exe = kernel.adapter.executable
        if exe is None:
            exe = runtime.Executable(kernel.artifact.rt_mod)
        compiled[name] = exe
        logger.debug("Compiled: %s", name)

    return compiled
