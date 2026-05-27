from . import _ffi_api


def FuseTIR():
    """Fuse TIR blocks into a single block.

    This pass is used to fuse TIR blocks into a single block, which can be used for
    code generation. It is used in the code generation pipeline after the TIR blocks
    are generated.

    Returns
    -------
    fuse_tir : tvm.transform.Pass
        The pass to fuse TIR blocks.
    """
    return _ffi_api.FuseTIR()


def FuseChainedGemm():
    """Detect chained GEMM call_tir calls in Relax dataflow and fuse them.

    Scans the Relax main function for patterns like::

        lv  = call_tir(matmul_gv,  [A, B], ...)
        lv2 = call_tir(matmul1_gv, [lv, C], ...)

    and merges the underlying TIR PrimFuncs so the intermediate tensor
    stays in shared memory instead of going through global memory.

    The fused function is marked ``tir.is_lowered`` so downstream
    lowering skips it.

    Returns
    -------
    fpass : tvm.transform.Pass
    """
    from .fuse_chained_gemm import FuseChainedGemmRelax

    return FuseChainedGemmRelax()
