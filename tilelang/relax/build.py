"""Relax VM build helpers backed by TileLang lowering."""

from __future__ import annotations


from tilelang import tvm
from tvm import relax, tir
from tvm.runtime import Executable
from tvm.target import Target


def _extract_attrs(mod: tvm.IRModule):
    attrs = dict(mod.attrs) if mod.attrs else {}
    ext_libs = attrs.get("external_mods", [])
    constants = attrs.get("const_name_to_constant", {})
    return ext_libs, constants


def _has_functions(mod: tvm.IRModule | None) -> bool:
    return mod is not None and len(mod.functions) > 0


def _merge_irmodules(*mods: tvm.IRModule, attrs=None) -> tvm.IRModule:
    funcs = {}
    for mod in mods:
        if mod is None:
            continue
        funcs.update(mod.functions)
    return tvm.IRModule(funcs, attrs=attrs)


def _split_vm_entry_functions(tir_mod: tvm.IRModule) -> tuple[tvm.IRModule | None, tvm.IRModule | None]:
    vm_entry_funcs = {}
    kernel_funcs = {}
    for gvar, func in tir_mod.functions.items():
        if isinstance(func, tir.PrimFunc):
            symbol = func.attrs.get("global_symbol") if func.attrs else None
            if symbol is not None and str(symbol).startswith("__vmtir__"):
                vm_entry_funcs[gvar] = func
                continue
        kernel_funcs[gvar] = func

    vm_entry_mod = tvm.IRModule(vm_entry_funcs, attrs=tir_mod.attrs) if vm_entry_funcs else None
    kernel_mod = tvm.IRModule(kernel_funcs, attrs=tir_mod.attrs) if kernel_funcs else None
    return vm_entry_mod, kernel_mod


def _build_with_tilelang(
    tir_mod: tvm.IRModule,
    target: str | Target | None,
):
    from tilelang.engine.lower import (
        canon_target_host,
        device_codegen,
        get_device_call,
        get_host_call,
        host_codegen,
        is_cpu_device_backend,
    )
    from tilelang.engine.phase import LowerAndLegalize, OptimizeForTarget

    if target is None:
        raise ValueError("TileLang Relax VM build requires an explicit target.")
    if isinstance(target, str):
        target = tvm.target.Target(target)
    target_host = tvm.target.Target.canon_target(canon_target_host(target, None))
    target = tvm.target.Target(target, target_host)

    tir_mod = tir.transform.BindTarget(target)(tir_mod)
    vm_entry_mod, kernel_mod = _split_vm_entry_functions(tir_mod)

    kernel_host_mod = None
    device_rt_mod = None
    if _has_functions(kernel_mod):
        with target:
            kernel_mod = LowerAndLegalize(kernel_mod, target)
            kernel_mod = OptimizeForTarget(kernel_mod, target)

        is_device_c = is_cpu_device_backend(target)
        kernel_host_mod = tir.transform.Filter(get_host_call(is_device_c=is_device_c))(kernel_mod)
        kernel_device_mod = tir.transform.Filter(get_device_call(is_device_c=is_device_c))(kernel_mod)
        if _has_functions(kernel_device_mod):
            with target:
                device_rt_mod = device_codegen(kernel_device_mod, target)

    if _has_functions(vm_entry_mod):
        vm_entry_mod = tir.transform.MakePackedAPI()(vm_entry_mod)

    host_mod = _merge_irmodules(vm_entry_mod, kernel_host_mod, attrs=tir_mod.attrs)
    host_rt_mod = host_codegen(host_mod, target_host)
    if device_rt_mod is not None:
        host_rt_mod.import_module(device_rt_mod)
    return host_rt_mod


def default_tir_pipeline() -> tvm.transform.Pass:
    """Compatibility stub for callers that still pass a TIR pipeline.

    TileLang Relax VM build does not use TVM's generic TIR pipeline by default.
    The return value remains available for opt-out / fallback callers that want
    to route through `tvm.relax.build`.
    """

    @tvm.transform.module_pass(opt_level=0)
    def _identity(mod: tvm.ir.IRModule, _ctx: tvm.transform.PassContext) -> tvm.ir.IRModule:
        return mod

    return _identity


def build(
    mod: tvm.IRModule,
    target: str | Target | None = None,
    params: dict[str, list] | None = None,
    relax_pipeline: None | str | tvm.transform.Pass = "default_build",
    tir_pipeline: str | tvm.transform.Pass | None = None,
    exec_mode: str = "compiled",
    *,
    system_lib: bool | None = None,
    use_tilelang_tir: bool = True,
) -> Executable:
    """Build a Relax module to VM executable using TileLang TIR lowering.

    By default, this avoids TVM's generic TIR build pipeline entirely. Instead
    it runs Relax VM codegen, lowers the leftover TIR module through TileLang's
    own lowering/codegen stack, and links the resulting runtime module into the
    Relax VM executable.
    """

    if isinstance(target, str):
        target = tvm.target.Target(target)
    if not params:
        params = {}

    if relax_pipeline is not None:
        if isinstance(relax_pipeline, str):
            relax_pipeline = relax.get_pipeline(relax_pipeline)
        if target is None:
            mod = relax_pipeline(mod)
        else:
            with target:
                mod = relax_pipeline(mod)

    ext_libs, constants = _extract_attrs(mod)
    params.update(dict(constants))

    builder = relax.ExecBuilder()
    mod = relax.vm_build._vmcodegen(builder, mod, exec_mode)
    tir_mod = relax.vm_build._filter_tir(mod)

    if use_tilelang_tir and tir_mod is not None and len(tir_mod.functions) > 0:
        tilelang_rt_mod = _build_with_tilelang(tir_mod, target)
        ext_libs = list(ext_libs) + [tilelang_rt_mod]
        return relax.vm_build._vmlink(
            builder=builder,
            target=target,
            tir_mod=None,
            ext_libs=ext_libs,
            params=params,
            system_lib=system_lib,
        )

    if tir_pipeline is None:
        tir_pipeline = "default"
    return relax.vm_build._vmlink(
        builder=builder,
        target=target,
        tir_mod=tir_mod,
        tir_pipeline=tir_pipeline,
        ext_libs=ext_libs,
        params=params,
        system_lib=system_lib,
    )
