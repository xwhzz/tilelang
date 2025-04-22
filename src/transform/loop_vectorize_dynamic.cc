// Copyright (c) Tile-AI Corporation.
// Licensed under the MIT License.

/*!
 * \file loop_vectorize_dynamic.cc
 * \brief A tool to automatically vectorize a for loop with dynamic shape
 * \brief Reference to loop_vectorize.cc and vectorize_loop.cc
 */

#include <cstdint>
#include <tvm/arith/iter_affine_map.h>
#include <tvm/tir/builtin.h>
#include <tvm/tir/op.h>
#include <tvm/tir/stmt_functor.h>

#include <numeric>

#include "../layout/layout.h"
#include "../layout/utils.h"
#include "../op/builtin.h"
#include "arith/int_operator.h"
#include "arith/ir_visitor_with_analyzer.h"
#include "common/loop_vectorization_utils.h"

namespace tvm {
namespace tl {

using namespace tir;
using arith::IRMutatorWithAnalyzer;

struct VectorizePlanResult {
  int vector_size;
  bool dynamic;
  PrimExpr condition;
};

bool IndiceCanVectorizeDynamic(PrimExpr expr, Var var, PrimExpr iter_var_size,
                               int target_vectorized_size,
                               arith::Analyzer *analyzer) {
  ICHECK(target_vectorized_size >= 1);
  if (target_vectorized_size == 1)
    return true;
  if (!analyzer->CanProveEqual(FloorMod(iter_var_size, target_vectorized_size),
                               0))
    return false;
  Var v0("v0"), v1("v1");
  analyzer->Bind(v0, Range(0, target_vectorized_size));
  analyzer->Bind(v1, Range(0, FloorDiv(iter_var_size, target_vectorized_size)));
  PrimExpr expr_transformed = analyzer->Simplify(
      Substitute(expr, {{var, v0 + v1 * target_vectorized_size}}));

  Vectorizer vectorizer(v0, IntImm(v0->dtype, target_vectorized_size));
  PrimExpr expr_vectorized = vectorizer.VisitExpr(expr_transformed);
  auto ramp_node = expr_vectorized.as<RampNode>();
  if (!ramp_node) {
    // Broadcast value
    if (expr_vectorized.dtype().lanes() == 1)
      return true;
    else
      return false;
  } else {
    return is_one(ramp_node->stride);
  }
}

class VectorizePlannerDynamic : public arith::IRVisitorWithAnalyzer {
public:
  VectorizePlannerDynamic(int dynamic_alignment,
                          bool disable_dynamic_tail_split)
      : dynamic_alignment_(dynamic_alignment),
        disable_dynamic_tail_split_(disable_dynamic_tail_split),
        vector_load_bits_max_(128) {
    if (disable_dynamic_tail_split_) {
      vector_size_ = dynamic_alignment_;
    } else {
      vector_size_ = vector_load_bits_max_;
    }
  }

  int Plan(const For &node) {
    this->operator()(node);
    // Always Enable vectorization
    // if (!has_nonlocal_memory_access_) return 1;
    return vector_size_;
  }

  bool GetDynamic() { return dynamic_; }

  PrimExpr GetCondition() { return condition_; }

private:
  void VisitStmt_(const ForNode *node) final {
    inner_for_ = node;
    iter_map_.Set(node->loop_var, Range(node->min, node->extent));
    arith::IRVisitorWithAnalyzer::VisitStmt_(node);
  }

  void VisitExpr_(const BufferLoadNode *node) final {
    if (node->buffer.scope() == "shared" || node->buffer.scope() == "global" ||
        node->buffer.scope() == "shared.dyn")
      has_nonlocal_memory_access_ = true;
    if (node->buffer->shape.size() == 1) {
      // TODO(lei): This should be improved as
      // constant buffer that tl hack to use as local register.
      auto boundary_check = node->buffer->shape[0].as<IntImmNode>();
      if (boundary_check && boundary_check->value == 1) {
        return arith::IRVisitorWithAnalyzer::VisitExpr_(node);
      }
    }
    UpdateVectorSize(node->indices, node->buffer);
    return arith::IRVisitorWithAnalyzer::VisitExpr_(node);
  }

  void VisitStmt_(const BufferStoreNode *node) final {
    if (node->buffer.scope() == "shared" || node->buffer.scope() == "global" ||
        node->buffer.scope() == "shared.dyn")
      has_nonlocal_memory_access_ = true;
    UpdateVectorSize(node->indices, node->buffer);
    return arith::IRVisitorWithAnalyzer::VisitStmt_(node);
  }

  void VisitStmt_(const IfThenElseNode *node) final {
    CheckConditionVectorized(node->condition);
    return arith::IRVisitorWithAnalyzer::VisitStmt_(node);
  }

  void VisitExpr_(const CallNode *node) final {
    if (node->op == builtin::if_then_else()) {
      CheckConditionVectorized(node->args[0]);
    } else if (node->op == builtin::call_extern()) {
      // do not vectorize extern calls
      vector_size_ = 1;
    }
    return arith::IRVisitorWithAnalyzer::VisitExpr_(node);
  }

  void CheckConditionVectorized(const PrimExpr &cond) {
    // TODO: may perform some checks here
  }

  void UpdateVectorSize(const Array<PrimExpr> indices, const Buffer &buffer) {
    if (!inner_for_)
      return;
    auto extent_ptr = inner_for_->extent.as<IntImmNode>();
    if (!extent_ptr)
      return;

    const DataType &access_type = buffer->dtype;
    // i // 2, i % 8 can also be vectorized as factor 16
    int max_vector_size = vector_load_bits_max_ / access_type.bits();
    if (access_type.is_e4m3_float8() or access_type.is_e5m2_float8()) {
      max_vector_size = 1; // [temporarily] do not vectorize float8
    }
    // so we should disable this GCD optimization
    max_vector_size = arith::ZeroAwareGCD(max_vector_size, extent_ptr->value);

    auto last_dim = buffer->shape.back();
    auto mod_set = analyzer_.modular_set(last_dim);
    // when dynamic shape like [m, k]: coeff=1, base=0, GCD will block
    // conditionally tail vectorize
    if (buffer->shape.back().as<IntImmNode>()) {
      max_vector_size = arith::ZeroAwareGCD(max_vector_size, mod_set->coeff);

      auto gcd_base = arith::ZeroAwareGCD(max_vector_size, mod_set->base);
      // If gcd_base is equal to the last dimension,
      // we should analyze the second-to-last dimension
      // in relation to the last dimension.
      if (gcd_base < Downcast<IntImm>(last_dim)->value) {
        max_vector_size = gcd_base;
      }

      vector_size_ = arith::ZeroAwareGCD(max_vector_size, vector_size_);

      PrimExpr elem_offset = 0;
      PrimExpr stride = 1;
      for (int i = indices.size() - 1; i >= 0; --i) {
        elem_offset = elem_offset + indices[i] * stride;
        stride = stride * buffer->shape[i];
      }
      while (!IndiceCanVectorizeDynamic(elem_offset, inner_for_->loop_var,
                                        inner_for_->extent, vector_size_,
                                        &analyzer_)) {
        vector_size_ /= 2;
      }
    } else {
      // dynamic shape load: get the vectorization condition
      dynamic_ = true;
      if (!disable_dynamic_tail_split_ &&
          vector_size_ >= vector_load_bits_max_ / buffer->dtype.bits()) {
        vector_size_ = vector_load_bits_max_ / buffer->dtype.bits();
      }
      PrimExpr offset = buffer.OffsetOf(indices).back();
      // condition for alignment, maybe useless
      condition_ = (FloorMod(offset, vector_size_) == 0);
    }
  }

  // Use dynamic alignment from pass config
  int vector_load_bits_max_;
  int dynamic_alignment_;
  bool disable_dynamic_tail_split_;

  int vector_size_;

  const ForNode *inner_for_;
  Map<Var, Range> iter_map_;
  bool has_nonlocal_memory_access_ = false;
  // conditionally vectorize
  bool dynamic_ = false;
  PrimExpr condition_;
};

class VectorizedBodyMutator : public StmtExprMutator {
public:
  VectorizedBodyMutator(Var inner_var, int vector_size,
                        std::vector<PrimExpr> conditions)
      : inner_var_(inner_var), vector_size_(vector_size),
        conditions_(conditions) {}

private:
  PrimExpr VisitExpr_(const CallNode *op) final {
    if (op->op.same_as(builtin::if_then_else())) {
      // TODO: Currently not ramp, but only reserve the "then" part (because
      // conditions are move outside this vectorized loop)
      PrimExpr ifexpr = op->args[0];
      PrimExpr thenexpr = op->args[1];
      bool flag = false;
      for (auto &cond : conditions_) {
        if (ifexpr.get() == cond.get()) {
          flag = true;
        }
      }
      if (flag) {
        return thenexpr;
      } else {
        return GetRef<PrimExpr>(op);
      }
    } else {
      return GetRef<PrimExpr>(op);
    }
  }

  Var inner_var_;
  int vector_size_;
  std::vector<PrimExpr> conditions_;
};

class VectorizedConditionExtracter : public StmtExprVisitor {
public:
  VectorizedConditionExtracter() = default;
  std::vector<PrimExpr> GetConditions(Stmt body) {
    this->VisitStmt(body);
    return conditions_;
  }

private:
  void VisitExpr_(const CallNode *op) final {
    if (op->op.same_as(builtin::if_then_else())) {
      PrimExpr cond = op->args[0];
      conditions_.emplace_back(cond);
    }
    StmtExprVisitor::VisitExpr_(op);
  }

  void VisitStmt_(const IfThenElseNode *node) final {
    conditions_.emplace_back(node->condition);
    StmtExprVisitor::VisitStmt_(node);
  }

  std::vector<PrimExpr> conditions_;
};

class NestedLoopChecker : public StmtExprVisitor {
public:
  NestedLoopChecker() : loop_num_(0) {}
  int GetNestLoopNum(Stmt body) {
    this->VisitStmt(body);
    return loop_num_;
  }

private:
  void VisitStmt_(const ForNode *node) final {
    loop_num_++;
    StmtExprVisitor::VisitStmt_(node);
  }
  int loop_num_;
};

// Modify every subexpression in the condition
class VectorizedConditionMutator : public StmtExprMutator {
public:
  VectorizedConditionMutator(Var inner_var, int extent)
      : inner_var_(inner_var), vector_size_(extent) {}

private:
  PrimExpr VisitExpr_(const GENode *node) final {
    PrimExpr lhs = StmtExprMutator::VisitExpr(node->a);
    PrimExpr rhs = StmtExprMutator::VisitExpr(node->b);
    auto span = node->span;
    Map<Var, PrimExpr> vmap_lhs, vmap_rhs;
    vmap_lhs.Set(inner_var_, 0);
    PrimExpr lhs_bound = Substitute(lhs, vmap_lhs);
    vmap_rhs.Set(inner_var_, vector_size_ - 1);
    PrimExpr rhs_bound = Substitute(rhs, vmap_rhs);
    return GE(lhs_bound, rhs_bound, span);
  }

  PrimExpr VisitExpr_(const GTNode *node) final {
    PrimExpr lhs = StmtExprMutator::VisitExpr(node->a);
    PrimExpr rhs = StmtExprMutator::VisitExpr(node->b);
    auto span = node->span;
    Map<Var, PrimExpr> vmap_lhs, vmap_rhs;
    vmap_lhs.Set(inner_var_, 0);
    PrimExpr lhs_bound = Substitute(lhs, vmap_lhs);
    vmap_rhs.Set(inner_var_, vector_size_ - 1);
    PrimExpr rhs_bound = Substitute(rhs, vmap_rhs);
    return GT(lhs_bound, rhs_bound, span);
  }

  PrimExpr VisitExpr_(const LENode *node) final {
    PrimExpr lhs = StmtExprMutator::VisitExpr(node->a);
    PrimExpr rhs = StmtExprMutator::VisitExpr(node->b);
    auto span = node->span;
    Map<Var, PrimExpr> vmap_lhs, vmap_rhs;
    vmap_lhs.Set(inner_var_, vector_size_ - 1);
    PrimExpr lhs_bound = Substitute(lhs, vmap_lhs);
    vmap_rhs.Set(inner_var_, 0);
    PrimExpr rhs_bound = Substitute(rhs, vmap_rhs);
    return LE(lhs_bound, rhs_bound, span);
  }

  PrimExpr VisitExpr_(const LTNode *node) final {
    PrimExpr lhs = StmtExprMutator::VisitExpr(node->a);
    PrimExpr rhs = StmtExprMutator::VisitExpr(node->b);
    auto span = node->span;
    Map<Var, PrimExpr> vmap_lhs, vmap_rhs;
    vmap_lhs.Set(inner_var_, vector_size_ - 1);
    PrimExpr lhs_bound = Substitute(lhs, vmap_lhs);
    vmap_rhs.Set(inner_var_, 0);
    PrimExpr rhs_bound = Substitute(rhs, vmap_rhs);
    return LT(lhs_bound, rhs_bound, span);
  }

  Var inner_var_;
  int vector_size_;
};

class VectorizeRewriterDynamic : public StmtExprMutator {
public:
  VectorizeRewriterDynamic(VectorizePlanResult plan,
                           bool disable_dynamic_tail_split)
      : vector_size_(plan.vector_size), condition_(plan.condition),
        dynamic_(plan.dynamic),
        disable_dynamic_tail_split_(disable_dynamic_tail_split) {}

private:
  Stmt VisitStmt_(const ForNode *node) final {
    // Get pass config `tl.disable_dynamic_tail_split`
    tvm::transform::PassContext ctxt = tvm::transform::PassContext::Current();
    Optional<Bool> opt_disable_dynamic_tail_split =
        ctxt->GetConfig(kDisableDynamicTailSplit, Optional<Bool>());
    bool disable_dynamic_tail_split =
        opt_disable_dynamic_tail_split.value_or(Bool(false));

    inner_for_ = node;
    auto ret = StmtExprMutator::VisitStmt_(node);
    if (inner_for_ != node) {
      return ret;
    }
    For fnode = ret.as<For>().value();
    auto old_var = fnode->loop_var;
    if (!fnode->extent.as<IntImmNode>()) {
      return ret;
    }
    int extent = Downcast<IntImm>(fnode->extent)->value;

    if (!dynamic_) {
      return fnode;
    }

    if (!disable_dynamic_tail_split) {
      // To handle the fact that cp.async only support address aligned with
      // access size
      vector_size_ = 1;
    }

    ICHECK(extent % vector_size_ == 0)
        << "extent: " << extent << " vector_size_: " << vector_size_;
    ICHECK(is_zero(fnode->min));
    Var inner_var = Var("vec");
    Var outer_var = Var(old_var->name_hint);
    Map<Var, PrimExpr> vmap;
    vmap.Set(fnode->loop_var, outer_var * vector_size_ + inner_var);
    Stmt body = Substitute(fnode->body, vmap);

    VectorizedConditionExtracter extracter;
    std::vector<PrimExpr> conditions = extracter.GetConditions(body);

    VectorizedConditionMutator condition_mutator(inner_var, vector_size_);

    // Adaptively set vectorized variable to the min/max value of the extent
    PrimExpr condition_bound;
    if (conditions.size() > 0) {
      condition_bound = condition_mutator(conditions[0]);
      for (int i = 1; i < conditions.size(); ++i) {
        condition_bound = condition_bound && condition_mutator(conditions[i]);
      }
    }

    if (!disable_dynamic_tail_split) {
      // If dynamic_tail_split is true, we will vectorize the loop with
      // if-then-else conditions modify body in the vectorized loop
      VectorizedBodyMutator mutator(inner_var, vector_size_, conditions);
      Stmt vectorize_body = mutator(body);

      // add condition ifthenelse here
      For vectorize_for =
          For(inner_var, 0, vector_size_, ForKind::kVectorized, vectorize_body);
      For serial_for = For(inner_var, 0, vector_size_, ForKind::kSerial, body);
      if (conditions.size() > 0) {
        body = IfThenElse(condition_bound, vectorize_for, serial_for);
      } else {
        body = vectorize_for;
      }
      body = For(outer_var, 0, extent / vector_size_, fnode->kind, body,
                 fnode->thread_binding, fnode->annotations, fnode->span);
      return body;
    } else {
      // If dynamic_tail_split is false, we will directly vectorize the loop
      // without dynamic tail split and if_then_else, which may lead to error
      VectorizedBodyMutator mutator(inner_var, vector_size_, conditions);
      Stmt vectorize_body = mutator(body);

      For vectorize_for =
          For(inner_var, 0, vector_size_, ForKind::kVectorized, vectorize_body);
      body =
          For(outer_var, 0, extent / vector_size_, fnode->kind, vectorize_for,
              fnode->thread_binding, fnode->annotations, fnode->span);
      return body;
    }
  }

  const ForNode *inner_for_;
  int vector_size_;
  const PrimExpr condition_;
  const bool dynamic_;
  const bool disable_dynamic_tail_split_;
};

VectorizePlanResult
GetVectorizePlanResultDynamic(const For &loop, int dynamic_alignment,
                              bool disable_dynamic_tail_split) {
  VectorizePlannerDynamic planner(dynamic_alignment,
                                  disable_dynamic_tail_split);
  int vector_size = planner.Plan(loop);
  bool dynamic = planner.GetDynamic();
  PrimExpr condition = planner.GetCondition();
  return {vector_size, dynamic, condition};
}

class LoopVectorizerDynamic : public IRMutatorWithAnalyzer {
public:
  static Stmt Substitute(Stmt stmt, bool disable_dynamic_tail_split,
                         int dynamic_alignment) {
    arith::Analyzer analyzer;
    LoopVectorizerDynamic substituter(&analyzer, disable_dynamic_tail_split,
                                      dynamic_alignment);
    stmt = substituter.VisitStmt(stmt);
    return stmt;
  }

private:
  LoopVectorizerDynamic(arith::Analyzer *analyzer,
                        bool disable_dynamic_tail_split, int dynamic_alignment)
      : arith::IRMutatorWithAnalyzer(analyzer),
        disable_dynamic_tail_split_(disable_dynamic_tail_split),
        dynamic_alignment_(dynamic_alignment) {}

  Stmt VisitStmt_(const ForNode *op) final {
    For for_node = Downcast<For>(IRMutatorWithAnalyzer::VisitStmt_(op));
    VectorizePlanResult res{vector_load_bits_max_, false, 0};
    res = GetVectorizePlanResultDynamic(for_node, dynamic_alignment_,
                                        disable_dynamic_tail_split_);
    NestedLoopChecker checker;
    int nest_num = checker.GetNestLoopNum(for_node);
    if (nest_num > 1 ||
        for_node->kind == ForKind::kVectorized) { // only rewrite the innermost
                                                  // non-vectorized loop
      return for_node;
    }
    int vectorize_hint = res.vector_size;
    auto rewriter = VectorizeRewriterDynamic(res, disable_dynamic_tail_split_);
    return Downcast<For>(rewriter(for_node));
  }

  const int vector_load_bits_max_ = 128;
  int dynamic_alignment_;
  bool disable_dynamic_tail_split_;
};

class VectorizeSkipperDynamic : public StmtMutator {
public:
  Stmt VisitStmt_(const ForNode *op) final {
    Stmt stmt = StmtMutator::VisitStmt_(op);
    op = stmt.as<ForNode>();
    if (op->kind == ForKind::kVectorized) {
      return For(op->loop_var, op->min, op->extent, ForKind::kSerial, op->body);
    } else {
      return stmt;
    }
  }
};

tvm::transform::Pass LoopVectorizeDynamic() {
  using namespace tir::transform;
  auto pass_func = [=](PrimFunc f, IRModule m, PassContext ctx) {
    bool disable_dynamic_tail_split =
        ctx->GetConfig<Bool>(kDisableDynamicTailSplit, Bool(true)).value();
    int dynamic_alignment =
        (int)(ctx->GetConfig<Integer>(kDynamicAlignment, Integer(8))
                  .value_or(Integer(8))
                  ->value);
    // Ensure tl.dynamic_alignment is a power of 2
    if (disable_dynamic_tail_split &&
        ((dynamic_alignment & (dynamic_alignment - 1)) != 0)) {
      LOG(FATAL) << "tl.dynamic_alignment must be a power of 2, but got "
                 << dynamic_alignment;
    }
    auto *n = f.CopyOnWrite();
    n->body = LoopVectorizerDynamic::Substitute(
        std::move(n->body), disable_dynamic_tail_split, dynamic_alignment);
    return f;
  };
  return CreatePrimFuncPass(pass_func, 0, "tl.LoopVectorizeDynamic", {});
}

// Register the pass globally so it can be used in the compilation pipeline
TVM_REGISTER_GLOBAL("tl.transform.LoopVectorizeDynamic")
    .set_body_typed(LoopVectorizeDynamic);

} // namespace tl
} // namespace tvm
