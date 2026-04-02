"""TileLang Relax optimization pipeline."""

import logging

from tilelang import tvm as tvm
from tvm import relax, dlight as dl
from tvm.target import Target

from tilelang.schedule.gpu import default_schedule_rules
from tilelang.graph.fuse_skip_reduction import FuseSkipReduction

logger = logging.getLogger(__name__)


def _try_pass(mod, transform, name):
    """Apply a pass, returning the original module on failure."""
    try:
        return transform(mod)
    except Exception:
        logger.debug("Optional pass %s failed, skipping", name, exc_info=True)
        return mod


def run_pipeline(mod: tvm.IRModule, target: Target) -> tvm.IRModule:
    """Apply the TileLang Relax compilation pipeline."""
    rules = default_schedule_rules()

    with target:
        mod = _try_pass(mod, relax.transform.FuseTransposeMatmul(), "FuseTransposeMatmul")
        mod = _try_pass(mod, relax.transform.CombineParallelMatmul(), "CombineParallelMatmul")
        mod = _try_pass(mod, relax.transform.ReorderPermuteDimsAfterConcat(),
                        "ReorderPermuteDimsAfterConcat")

        seq = tvm.transform.Sequential([
            relax.transform.LegalizeOps(),
            relax.transform.AnnotateTIROpPattern(),
            # FoldConstant is skipped — it uses emit_te which can't
            # handle symbolic shapes from dynamic dim propagation.
            # relax.transform.FoldConstant(),
            relax.transform.FuseOps(),
            FuseSkipReduction(),
            relax.transform.FuseTIR(),
            relax.transform.DeadCodeElimination(),
            dl.ApplyDefaultSchedule(*rules),
            relax.transform.RewriteDataflowReshape(),
            relax.transform.ToNonDataflow(),
            relax.transform.RemovePurityChecking(),
            relax.transform.CallTIRRewrite(),
        ])
        mod = seq(mod)

    return mod
