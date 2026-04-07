"""Build Relax VM executable with TileLang-compiled TIR kernels.

Alternative execution path to the Python/C wrapper codegen (codegen.py).
Builds a Relax VM that:
  - Manages memory natively (StaticPlanBlockMemory, no Python-side pools)
  - Has minimal dispatch overhead (C++ bytecode loop vs per-kernel Python dispatch)
  - Uses TileLang's CUDA codegen for all GPU kernels (proper half-precision headers)
"""

import logging

import torch
import tvm_ffi

from tilelang import tvm as tvm
from tvm import relax, tir, runtime
from tvm.runtime import from_dlpack as _tvm_from_dlpack
from tvm.target import Target

from tilelang.graph.pipeline import run_pipeline
from tilelang.engine.phase import NormalizeScheduledIR, LowerAndLegalize, OptimizeForTarget
from tilelang.engine.lower import (
    device_codegen,
    host_codegen,
    get_host_call,
    get_device_call,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Torch fallback dispatch: register torch ops as TVM packed functions
# ---------------------------------------------------------------------------

def _register_torch_fallbacks(fallback_calls: dict) -> list[str]:
    """Register torch fallback ops as TVM packed functions for the VM.

    The converter emits ``Call(ExternFunc("torch_fallback.<op>"), inputs)``
    for ops dispatched via ``extern_dispatch``.  The VM resolves these
    at runtime through the TVM FFI registry.  The fallback function
    returns the result directly (no pre-allocation or copy).

    Returns the list of registered function names.
    """
    registered = []
    for op_name, (op_fn, arg_template, kwargs) in fallback_calls.items():
        func_name = f"torch_fallback.{op_name}"
        wrapper = _make_fallback_wrapper(op_fn, arg_template, kwargs)
        tvm_ffi.register_global_func(func_name, wrapper, override=True)
        registered.append(func_name)
        logger.debug("Registered VM fallback: %s", func_name)
    return registered


def _make_fallback_wrapper(op_fn, arg_template, kwargs):
    """Generic fallback wrapper — calls torch op and returns result directly.

    Since fallback ops use regular ``Call`` (not ``call_dps_packed``),
    the callee allocates the output and returns it.  No pre-allocation
    or DtoD copy needed — the VM stores the returned tensor in a register.
    """
    def wrapper(*tvm_args):
        # Zero-copy DLPack conversion for all inputs
        torch_inputs = [torch.from_dlpack(t) for t in tvm_args]

        # Reconstruct positional args from template
        args = []
        tidx = 0
        for kind, val in arg_template:
            if kind is True:
                args.append(torch_inputs[tidx])
                tidx += 1
            elif kind == "list":
                lst = []
                for name in val:
                    if isinstance(name, str):
                        lst.append(torch_inputs[tidx])
                        tidx += 1
                    else:
                        lst.append(name)
                args.append(lst)
            else:
                args.append(val)

        # Reconstruct keyword args
        resolved_kw = {}
        for k, v in kwargs.items():
            if isinstance(v, tuple) and len(v) == 2 and v[0] == "__tensor__":
                resolved_kw[k] = torch_inputs[tidx]
                tidx += 1
            else:
                resolved_kw[k] = v

        # Call torch op — callee allocates output
        if isinstance(op_fn, str):
            result = getattr(args[0], op_fn)(*args[1:], **resolved_kw)
        else:
            result = op_fn(*args, **resolved_kw)

        # Return result to VM as TVM tensor
        if isinstance(result, torch.Tensor):
            return _tvm_from_dlpack(result.contiguous())
        return result

    return wrapper


# ---------------------------------------------------------------------------
# VM lowering passes (extends the Relax pipeline for VM execution)
# ---------------------------------------------------------------------------

def _apply_vm_lowering(mod: tvm.IRModule, target: Target) -> tvm.IRModule:
    """Apply VM-specific lowering passes after the main Relax pipeline.

    These passes transform the Relax IR into a form the VM codegen can consume:
      - KillAfterLastUse:    mark tensors freeable after last consumer
      - LowerRuntimeBuiltin: lower runtime builtins (print_format, etc.)
      - ComputePrimValue:    replace symbolic PrimValue exprs with call_tir to tiny PrimFuncs
      - VMShapeLower:        lower shape-related ops to VM-understood form
      - AttachGlobalSymbol:  attach global_symbol attr so VM can find TIR funcs by name
    """
    with target:
        seq = tvm.transform.Sequential([
            relax.transform.KillAfterLastUse(),
            relax.transform.LowerRuntimeBuiltin(),
            relax.transform.ComputePrimValue(),
            relax.transform.VMShapeLower(),
            relax.transform.AttachGlobalSymbol(),
        ])
        mod = seq(mod)
    return mod


# ---------------------------------------------------------------------------
# TIR compilation: split kernel vs host-only, compile each appropriately
# ---------------------------------------------------------------------------

def _is_host_target(func: tir.PrimFunc) -> bool:
    """Check if a PrimFunc is bound to a host (CPU) target."""
    target = func.attrs.get("target", None) if func.attrs else None
    if target is None:
        return False
    return str(target.kind) in ["llvm", "c"]


def _compile_kernels_tilelang(kernel_mod: tvm.IRModule, target: Target) -> runtime.Module:
    """Compile GPU kernel PrimFuncs through TileLang's full pipeline + codegen."""
    target_host_str = "llvm" if runtime.enabled("llvm") else "c"
    target_host = Target(target_host_str)
    full_target = Target(target, target_host)

    with tvm.transform.PassContext(opt_level=3), full_target:
        # Separate TileLang DSL kernels (already scheduled, skip NormalizeScheduledIR)
        # from schedule-rule kernels (need NormalizeScheduledIR).
        tl_funcs = {}
        sched_funcs = {}
        for gv, func in kernel_mod.functions.items():
            if isinstance(func, tir.PrimFunc) and func.attrs and \
               func.attrs.get("tir.is_tilelang_kernel", False):
                tl_funcs[gv] = func
            else:
                sched_funcs[gv] = func

        # Schedule-rule kernels: full pipeline
        if sched_funcs:
            sm = tvm.IRModule(sched_funcs)
            sm = NormalizeScheduledIR(sm)
            sm = LowerAndLegalize(sm, full_target)
            sm = OptimizeForTarget(sm, full_target)
        else:
            sm = tvm.IRModule({})

        # TileLang DSL kernels: skip NormalizeScheduledIR (already scheduled)
        if tl_funcs:
            tm = tvm.IRModule(tl_funcs)
            tm = LowerAndLegalize(tm, full_target)
            tm = OptimizeForTarget(tm, full_target)
        else:
            tm = tvm.IRModule({})

        # Merge
        km = tvm.IRModule({})
        for gv, func in sm.functions.items():
            km[gv] = func
        for gv, func in tm.functions.items():
            km[gv] = func

    # After OptimizeForTarget, functions are split into host wrappers + device kernels
    _is_host = get_host_call(False)
    _is_device = get_device_call(False)
    khost = tir.transform.Filter(_is_host)(km)
    kdevice = tir.transform.Filter(_is_device)(km)

    # TileLang codegen: proper CUTLASS headers, half-precision support
    device_rt = device_codegen(kdevice, full_target)
    host_rt = host_codegen(khost, target_host)
    host_rt.import_module(device_rt)
    return host_rt


def _compile_tir_for_vm(tir_mod: tvm.IRModule, target: Target) -> runtime.Module:
    """Compile all TIR functions for the VM.

    The TIR module contains two kinds of PrimFuncs after VM lowering:

    1. **GPU kernel functions** (from schedule rules / pattern rewrite)
       → Compiled via TileLang (NormalizeScheduledIR + LowerAndLegalize +
         OptimizeForTarget + TileLang CUDA codegen)

    2. **Host-only scalar functions** (from ComputePrimValue for symbolic shapes)
       → Compiled via standard TVM TIR pipeline (MakePackedAPI + LLVM codegen)

    BindTarget classifies them: host-only functions have ``tir.is_host_func``
    which BindTarget maps to a CPU target; kernel functions get the CUDA target.
    """
    target_host_str = "llvm" if runtime.enabled("llvm") else "c"
    target_host = Target(target_host_str)
    full_target = Target(target, target_host)

    # BindTarget classifies functions: kIsHostFunc → CPU target, others → CUDA
    tir_mod = tir.transform.BindTarget(full_target)(tir_mod)

    kernel_mod = tir.transform.Filter(lambda f: not _is_host_target(f))(tir_mod)
    host_only_mod = tir.transform.Filter(_is_host_target)(tir_mod)

    n_kernel = len(kernel_mod.get_global_vars())
    n_host = len(host_only_mod.get_global_vars())
    logger.info("TIR split: %d kernel, %d host-only functions", n_kernel, n_host)

    lib = None

    # GPU kernel functions → TileLang pipeline + codegen
    if n_kernel > 0:
        lib = _compile_kernels_tilelang(kernel_mod, target)

    # Host-only functions → standard TVM pipeline (needed for MakePackedAPI)
    if n_host > 0:
        host_lib = tvm.tir.build(host_only_mod, target=full_target, pipeline="default")
        if lib is not None:
            lib.import_module(host_lib)
        else:
            lib = host_lib

    return lib


# ---------------------------------------------------------------------------
# Top-level VM build
# ---------------------------------------------------------------------------

def build_vm_executable(
    relax_mod: tvm.IRModule,
    target: Target,
    fallback_calls: dict = None,
) -> relax.vm_build.VMExecutable:
    """Build a Relax VM executable from a Relax module.

    Pipeline::

        Relax optimization → VM lowering → bytecode generation
        → TileLang TIR compilation → VM linking → VMExecutable

    Parameters
    ----------
    fallback_calls : dict, optional
        Torch fallback ops from the converter.  Each entry is registered as
        a TVM packed function so the VM can dispatch to torch at runtime.
    """
    # Register torch fallback ops so the VM can resolve them
    if fallback_calls:
        _register_torch_fallbacks(fallback_calls)

    # Step 1: TileLang's Relax optimization pipeline
    mod = run_pipeline(relax_mod, target)

    # Step 2: VM-specific lowering passes
    mod = _apply_vm_lowering(mod, target)
    logger.info("VM lowering complete")

    # Step 3: VM bytecode generation (consumes Relax functions, leaves TIR)
    builder = relax.ExecBuilder()
    mod = tvm.get_global_func("relax.VMCodeGen")(builder, mod)
    logger.info("VM bytecode generated")

    # Step 4: Extract and compile TIR
    tir_funcs = {gv: f for gv, f in mod.functions.items() if isinstance(f, tir.PrimFunc)}
    lib = None
    if tir_funcs:
        tir_mod = tvm.IRModule(tir_funcs, attrs=mod.attrs)
        lib = _compile_tir_for_vm(tir_mod, target)

    # Step 5: Link into VM executable
    vm_exe = tvm.get_global_func("relax.VMLink")(
        builder, target, lib, [], {},
    )
    return relax.vm_build.VMExecutable(vm_exe)


# ---------------------------------------------------------------------------
# Torch-compatible VM runner
# ---------------------------------------------------------------------------

class VMRunner:
    """Wraps a Relax VM as a ``torch.compile``-compatible callable.

    Handles ``torch.Tensor`` ↔ ``tvm.nd.NDArray`` conversion via DLPack
    (zero-copy for contiguous tensors on the same device).

    Parameters
    ----------
    clone_output : bool
        If True (default), clone output tensors to prevent corruption
        when the VM reuses storage on the next call.  Set to False for
        benchmarking when outputs are consumed immediately.
    """

    def __init__(self, vm_exe, device, func_name="main", clone_output=True):
        self._vm = runtime.vm.VirtualMachine(vm_exe, device)
        self._func = self._vm[func_name]
        self._device = device
        self._clone = clone_output

    def __call__(self, *args):
        tvm_args = []
        for a in args:
            if isinstance(a, torch.Tensor):
                tvm_args.append(_tvm_from_dlpack(a))
            else:
                tvm_args.append(a)

        result = self._func(*tvm_args)
        return self._to_torch(result)

    def _to_torch(self, result):
        """Convert VM output to torch tensors."""
        clone = self._clone

        def _convert(v):
            if hasattr(v, "__dlpack__"):
                t = torch.from_dlpack(v)
                return t.clone() if clone else t
            return v

        # VM may return a single tensor, an Array/tuple, or a nested structure.
        # The FX graph's output is always a tuple — return matching structure.
        if hasattr(result, "__dlpack__"):
            return (_convert(result),)
        # tvm_ffi.container.Array or list/tuple
        if hasattr(result, "__len__") and hasattr(result, "__getitem__"):
            return tuple(_convert(result[i]) for i in range(len(result)))
        return result


def build_vm_runner(
    relax_mod: tvm.IRModule,
    target: Target,
    fallback_calls: dict = None,
    func_name: str = "main",
    clone_output: bool = True,
) -> VMRunner:
    """Build and return a torch-callable VM runner.

    Parameters
    ----------
    relax_mod : tvm.IRModule
        The Relax module (pre-pipeline).
    target : Target
        The compilation target (e.g. ``tvm.target.Target("cuda")``)
    fallback_calls : dict, optional
        Torch fallback ops to register for VM dispatch.
    func_name : str
        The Relax function to call (default ``"main"``).
    clone_output : bool
        Clone output tensors for safety (default True).  Set False
        for benchmarking when outputs are consumed before the next call.

    Returns
    -------
    VMRunner
        A callable that takes torch tensors and returns torch tensors.
    """
    device = tvm.cuda() if "cuda" in str(target) else tvm.cpu()
    vm_exe = build_vm_executable(relax_mod, target, fallback_calls)
    logger.info("VM runner built for %s on %s", func_name, device)
    return VMRunner(vm_exe, device, func_name, clone_output=clone_output)
