"""TileLang Relax optimization pipeline."""

import logging

from tilelang import tvm as tvm
from tvm import relax, dlight as dl
from tvm.target import Target
from tilelang.schedule.templates import default_schedule_rules
from tilelang.graph.pattern_rewrite import PatternRewritePass
from tilelang.graph.patterns import DEFAULT_PATTERNS
from tilelang.graph.patterns.fused_rope import fuse_qk_rope_pass
from tilelang.graph.passes import eliminate_reshape_kernels, fold_zero_binops
from tilelang.relax import FuseTIR

logger = logging.getLogger(__name__)


def _try_pass(mod, transform, name):
    """Apply a pass, returning the original module on failure."""
    try:
        return transform(mod)
    except Exception:
        logger.debug("Optional pass %s failed, skipping", name, exc_info=True)
        return mod


def run_pipeline(mod: tvm.IRModule, target: Target,
                 use_cuda_graph: bool = False) -> tvm.IRModule:
    """Apply the TileLang Relax compilation pipeline."""
    rules = default_schedule_rules(target)

    with target:
        mod = _try_pass(mod, relax.transform.FuseTransposeMatmul(), "FuseTransposeMatmul")
        mod = _try_pass(mod, relax.transform.CombineParallelMatmul(), "CombineParallelMatmul")
        mod = _try_pass(mod, relax.transform.ReorderPermuteDimsAfterConcat(),
                        "ReorderPermuteDimsAfterConcat")

        mod = _try_pass(mod, relax.transform.CanonicalizeBindings(),
                        "CanonicalizeBindings")
        mod = _try_pass(mod, relax.transform.FoldConstant(), "FoldConstant")
        mod = _try_pass(mod, fold_zero_binops, "FoldZeroBinops")

        seq1 = tvm.transform.Sequential([
            PatternRewritePass(DEFAULT_PATTERNS),
            relax.transform.LegalizeOps(),
            relax.transform.AnnotateTIROpPattern(),
            relax.transform.DeadCodeElimination(),
            relax.transform.FuseOps(),
            relax.transform.DeadCodeElimination(),
            FuseTIR(),
            relax.transform.DeadCodeElimination(),
        ])
        mod = seq1(mod)

        # LegalizeOps + FuseTIR can introduce call_tir(add_fn, [zero, x])
        # from broadcast/shape legalisation, so re-run after.
        mod = _try_pass(mod, fold_zero_binops, "FoldZeroBinops_post_fuse")
        mod = _try_pass(mod, fuse_qk_rope_pass, "FuseQKRope")

        # eliminate_reshape_kernels needs unscheduled TIR with original loop
        # structure, so run it before ApplyDefaultSchedule.
        mod = _try_pass(mod, eliminate_reshape_kernels,
                        "EliminateReshapeKernels")

        lowering_passes = [
            dl.ApplyDefaultSchedule(*rules),
            relax.transform.RewriteDataflowReshape(),
            relax.transform.ToNonDataflow(),
            relax.transform.RemovePurityChecking(),
            relax.transform.CallTIRRewrite(),
            relax.transform.StaticPlanBlockMemory(),
        ]
        if use_cuda_graph:
            lowering_passes.append(relax.transform.RewriteCUDAGraph())
        lowering_passes.append(relax.transform.LowerAllocTensor())
        mod = tvm.transform.Sequential(lowering_passes)(mod)

    return mod
