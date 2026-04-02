
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