"""TileLang Relax optimization pipeline."""

import logging

from tilelang import tvm as tvm
from tvm import relax, dlight as dl
from tvm.target import Target

from tilelang.schedule.gpu import default_schedule_rules
from tilelang.graph.pattern_rewrite import PatternRewritePass
import tilelang.graph.patterns  # noqa: F401 — registers built-in patterns
from tilelang.relax import FuseTIR

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
            # Pattern replacement: runs BEFORE LegalizeOps on high-level
            # Relax ops. Matches registered graph patterns (RMSNorm, etc.)
            # and replaces with optimized call_tir implementations.
            PatternRewritePass(),
            relax.transform.LegalizeOps(),
            relax.transform.AnnotateTIROpPattern(),
            relax.transform.DeadCodeElimination(),
            relax.transform.FuseOps(),
            FuseTIR(),
            relax.transform.DeadCodeElimination(),
            dl.ApplyDefaultSchedule(*rules),
            relax.transform.RewriteDataflowReshape(),
            relax.transform.ToNonDataflow(),
            relax.transform.RemovePurityChecking(),
            relax.transform.CallTIRRewrite(),
            relax.transform.StaticPlanBlockMemory(),
            relax.transform.LowerAllocTensor(),
        ])
        mod = seq(mod)

    return mod
