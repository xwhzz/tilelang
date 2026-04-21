"""TileLang Relax optimization pipeline."""

import logging

from tilelang import tvm as tvm
from tvm import relax, dlight as dl, tir
from tvm.target import Target

from tilelang.schedule.gpu import default_schedule_rules
from tilelang.graph.pattern_rewrite import PatternRewritePass
import tilelang.graph.patterns  # noqa: F401 — registers built-in patterns
from tilelang.graph.patterns.fused_rope import fuse_qk_rope_pass
from tilelang.relax import FuseTIR

logger = logging.getLogger(__name__)

_has_reshape_pattern = tvm.ffi.get_global_func("relax.analysis.has_reshape_pattern")


@relax.expr_functor.mutator
class _ReshapeEliminator(relax.PyExprMutator):
    """Replace call_tir to reshape-pattern TIR with zero-copy R.reshape.

    Unlike TVM's RewriteDataflowReshape, this works on ALL bindings
    (not just DataflowVar), eliminating reshape GPU kernels that would
    otherwise waste a kernel launch + memory copy.
    """

    def __init__(self, mod):
        super().__init__(mod)
        self.mod = mod

    def visit_call_(self, call):
        call = self.visit_expr_post_order(call)
        call_tir_op = tvm.ir.Op.get("relax.call_tir")
        if call.op != call_tir_op:
            return call

        gv = call.args[0]
        if not isinstance(gv, relax.GlobalVar):
            return call
        fn = self.mod.functions.get(gv)
        if not isinstance(fn, tir.PrimFunc):
            return call

        arg_tuple = call.args[1]
        if not isinstance(arg_tuple, relax.Tuple):
            return call
        args = arg_tuple.fields
        if len(args) == 0:
            return call

        if not _has_reshape_pattern(fn):
            return call

        # Verify element count matches (exclude strided_slice etc.)
        inp = args[0]
        inp_sinfo = inp.struct_info_
        res_sinfo = call.struct_info_
        if (not isinstance(inp_sinfo, relax.TensorStructInfo)
                or not isinstance(res_sinfo, relax.TensorStructInfo)):
            return call
        if inp_sinfo.dtype != res_sinfo.dtype:
            return call
        if inp_sinfo.ndim < 0 or res_sinfo.ndim < 0:
            return call
        inp_shape = inp_sinfo.shape
        res_shape = res_sinfo.shape
        if inp_shape is None or res_shape is None:
            return call

        return relax.op.reshape(inp, res_shape)


def _eliminate_reshape_kernels(mod):
    """Module pass: replace reshape TIR kernels with zero-copy reshapes."""
    for gv, fn in mod.functions.items():
        if isinstance(fn, relax.Function):
            new_fn = _ReshapeEliminator(mod).visit_expr(fn)
            mod[gv] = new_fn
    # Dead-code elimination removes now-unused TIR reshape functions.
    return relax.transform.DeadCodeElimination()(mod)


def _is_const_zero(expr) -> bool:
    """True iff ``expr`` is a Relax Constant tensor whose values are all 0."""
    if not isinstance(expr, relax.Constant):
        return False
    try:
        arr = expr.data.numpy()
    except Exception:
        return False
    import numpy as np
    return bool(np.all(arr == 0))


def _resolve_to_constant(expr, var_to_value: dict) -> "relax.Constant | None":
    """Walk var bindings until we either find a Constant or give up."""
    seen = set()
    while isinstance(expr, relax.Var):
        if expr in seen:
            return None
        seen.add(expr)
        v = var_to_value.get(expr)
        if v is None:
            return None
        expr = v
    if isinstance(expr, relax.Constant):
        return expr
    return None


def _is_pure_add_primfunc(fn, gv_name: str) -> bool:
    """Conservative check: is `fn` exactly the topi-emitted add PrimFunc?

    We require the global var name to be exactly 'add' (LegalizeOps gives
    standalone Relax adds this name, and FuseOps doesn't fuse them any
    further if their inputs are external).  The body must have 3 buffer
    params with int64 dtype matching the cache_position add pattern.
    """
    if not isinstance(fn, tir.PrimFunc):
        return False
    if gv_name != "add":
        return False
    if len(fn.params) != 3:
        return False
    return True


@relax.expr_functor.mutator
class _ZeroBinopFolder(relax.PyExprMutator):
    """Algebraic simplifier: ``add(0, x) → x``, ``add(x, 0) → x``.

    Matches both ``relax.add`` (pre-LegalizeOps) and ``call_tir(add_fn,
    [a, b])`` (post-LegalizeOps) so the same pass works either side of
    legalization.
    """

    def __init__(self, mod, var_to_value: dict):
        super().__init__(mod)
        self.mod = mod
        self.var_to_value = var_to_value
        self.folded_count = 0

    def _fold(self, a, b):
        ac = _resolve_to_constant(a, self.var_to_value)
        if ac is not None and _is_const_zero(ac):
            self.folded_count += 1
            return b
        bc = _resolve_to_constant(b, self.var_to_value)
        if bc is not None and _is_const_zero(bc):
            self.folded_count += 1
            return a
        return None

    def visit_call_(self, call):
        call = self.visit_expr_post_order(call)
        if not isinstance(call, relax.Call):
            return call
        op_name = call.op.name if hasattr(call.op, "name") else None

        if op_name == "relax.add" and len(call.args) == 2:
            folded = self._fold(call.args[0], call.args[1])
            if folded is not None:
                return folded

        if (op_name in ("relax.call_tir", "relax.call_tir_inplace")
                and len(call.args) >= 2):
            gv = call.args[0]
            arg_tup = call.args[1]
            if (isinstance(gv, relax.GlobalVar)
                    and isinstance(arg_tup, relax.Tuple)
                    and len(arg_tup.fields) == 2
                    and _is_pure_add_primfunc(
                        self.mod.functions.get(gv), gv.name_hint)):
                folded = self._fold(arg_tup.fields[0], arg_tup.fields[1])
                if folded is not None:
                    return folded
        return call


def _fold_zero_binops(mod):
    """Module pass: fold ``add(0, x) → x`` across all Relax functions.

    Builds a var→value map per function so the folder can resolve vars
    that ultimately bind to a Constant (Relax SSA usually binds constants
    to vars before using them).
    """
    total_folded = 0
    for gv, fn in mod.functions.items():
        if not isinstance(fn, relax.Function):
            continue
        var_to_value: dict = {}
        if isinstance(fn.body, relax.SeqExpr):
            for block in fn.body.blocks:
                for b in block.bindings:
                    if isinstance(b, relax.VarBinding):
                        var_to_value[b.var] = b.value
        folder = _ZeroBinopFolder(mod, var_to_value)
        mod[gv] = folder.visit_expr(fn)
        total_folded += folder.folded_count
    if total_folded > 0:
        logger.debug("_fold_zero_binops folded %d ops", total_folded)
    return relax.transform.DeadCodeElimination()(mod)


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
    rules = default_schedule_rules()

    with target:
        mod = _try_pass(mod, relax.transform.FuseTransposeMatmul(), "FuseTransposeMatmul")
        mod = _try_pass(mod, relax.transform.CombineParallelMatmul(), "CombineParallelMatmul")
        mod = _try_pass(mod, relax.transform.ReorderPermuteDimsAfterConcat(),
                        "ReorderPermuteDimsAfterConcat")

        mod = _try_pass(mod, relax.transform.CanonicalizeBindings(),
                        "CanonicalizeBindings")
        mod = _try_pass(mod, relax.transform.FoldConstant(), "FoldConstant")
        mod = _try_pass(mod, _fold_zero_binops, "FoldZeroBinops")

        seq1 = tvm.transform.Sequential([
            PatternRewritePass(),
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
        mod = _try_pass(mod, _fold_zero_binops, "FoldZeroBinops_post_fuse")
        mod = _try_pass(mod, fuse_qk_rope_pass, "FuseQKRope")
        mod = relax.transform.DeadCodeElimination()(mod)

        # _eliminate_reshape_kernels needs unscheduled TIR with original
        # loop structure, so run it before ApplyDefaultSchedule.
        mod = _try_pass(mod, _eliminate_reshape_kernels,
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
