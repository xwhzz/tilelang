"""Compile TIR functions from a Relax module using TileLang's compilation + cache."""

import logging

import tilelang
from tilelang import tvm as tvm
from tvm import tir, runtime
from tvm.target import Target

from tilelang.engine.phase import NormalizeScheduledIR

logger = logging.getLogger(__name__)


def _has_dynamic_shapes(func: tir.PrimFunc) -> bool:
    """Check if a TIR function has symbolic (dynamic) buffer shapes."""
    for p in func.params:
        if p in func.buffer_map:
            for s in func.buffer_map[p].shape:
                if isinstance(s, tir.Var):
                    return True
    return False


def _has_simple_dynamic_shapes(func: tir.PrimFunc) -> bool:
    """Check if all symbolic dims are plain tir.Var (not complex expressions)."""
    for p in func.params:
        if p in func.buffer_map:
            for s in func.buffer_map[p].shape:
                if not isinstance(s, (int, tir.IntImm, tir.Var)):
                    return False
    return True


def compile_tir_functions(
    mod: tvm.IRModule,
    target: Target,
) -> dict[str, callable]:
    """Extract and compile all TIR functions from a Relax module.

    Static-shape kernels use ``par_compile`` + raw ``Executable``.
    Dynamic-shape kernels use ``tilelang.lower()`` directly, producing
    ``Executable`` objects that accept symbolic shape parameters.
    """
    static_names, static_funcs = [], []
    dynamic_names, dynamic_funcs = [], []

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

        if _has_dynamic_shapes(prim_func):
            dynamic_names.append(name)
            dynamic_funcs.append(prim_func)
        else:
            static_names.append(name)
            static_funcs.append(prim_func)

    compiled = {}

    # Static kernels: parallel compilation via par_compile
    if static_funcs:
        from tilelang.jit import par_compile
        jit_kernels = par_compile(static_funcs, target=target)
        for name, kernel in zip(static_names, jit_kernels):
            exe = kernel.adapter.executable
            if exe is None:
                exe = runtime.Executable(kernel.artifact.rt_mod)
            compiled[name] = exe
            logger.debug("Compiled: %s (static)", name)

    # Dynamic kernels: compile via tilelang.lower() which supports
    # symbolic shapes in grid dimensions.
    for name, func in zip(dynamic_names, dynamic_funcs):
        try:
            func_mod = tvm.IRModule({name: func})
            artifact = tilelang.lower(func_mod, target=target)
            compiled[name] = runtime.Executable(artifact.rt_mod)
            logger.debug("Compiled: %s (dynamic)", name)
        except Exception as e:
            logger.warning("Failed to compile dynamic kernel %s: %s", name, e)

    return compiled
