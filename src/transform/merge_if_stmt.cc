/*!
 * \file if_stmt_binding.cc
 * \brief Merge the If Stmt in SeqStmt
 */

#include <tvm/tir/analysis.h>
#include <tvm/tir/builtin.h>
#include <tvm/tir/op.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>

#include "../op/builtin.h"

namespace tvm {
namespace tl {

using namespace tir;

class MergeIfStmtRewriter : public StmtExprMutator {
public:
  static PrimFunc Substitute(PrimFunc &f) {
    auto rewriter = MergeIfStmtRewriter();
    f.CopyOnWrite()->body = rewriter(f->body);
    return f;
  }

private:
  MergeIfStmtRewriter() = default;

  Stmt VisitStmt_(const SeqStmtNode *op) final {
    Array<Stmt> new_seq;

    PrimExpr current_condition;
    Array<Stmt> current_if_bodies;

    auto insert_attr = [&](const IfThenElse &if_stmt) -> Stmt {
      Stmt stmt;
      if (thread_extent_.defined()) {
        stmt = AttrStmt(make_zero(DataType::Int(32)), "shuffle_and_elect",
                        thread_extent_, if_stmt);
        thread_extent_ = PrimExpr();
      } else {
        stmt = if_stmt;
      }
      return stmt;
    };

    for (const Stmt &stmt : op->seq) {
      Stmt new_stmt = this->VisitStmt(stmt);
      if (const IfThenElseNode *if_node = new_stmt.as<IfThenElseNode>()) {
        if (!if_node->else_case.defined()) {
          if (current_condition.defined() &&
              StructuralEqual()(current_condition, if_node->condition)) {
            current_if_bodies.push_back(if_node->then_case);
            continue;
          } else {
            if (!current_if_bodies.empty()) {
              auto if_stmt = IfThenElse(current_condition,
                                        current_if_bodies.size() == 1
                                            ? current_if_bodies[0]
                                            : SeqStmt(current_if_bodies),
                                        Stmt());
              new_seq.push_back(insert_attr(if_stmt));
              current_if_bodies.clear();
            }

            current_condition = if_node->condition;
            current_if_bodies.push_back(if_node->then_case);
            continue;
          }
        }
      }

      if (!current_if_bodies.empty()) {
        auto if_stmt = IfThenElse(current_condition,
                                  current_if_bodies.size() == 1
                                      ? current_if_bodies[0]
                                      : SeqStmt(current_if_bodies),
                                  Stmt());
        new_seq.push_back(insert_attr(if_stmt));
        current_condition = PrimExpr();
        current_if_bodies.clear();
      }

      new_seq.push_back(new_stmt);
    }

    if (!current_if_bodies.empty()) {
      auto if_stmt =
          IfThenElse(current_condition,
                     current_if_bodies.size() == 1 ? current_if_bodies[0]
                                                   : SeqStmt(current_if_bodies),
                     Stmt());
      new_seq.push_back(insert_attr(if_stmt));
    }

    return new_seq.size() == 1 ? new_seq[0] : SeqStmt(new_seq);
  }

  Stmt VisitStmt_(const AttrStmtNode *op) final {
    if (op->attr_key == "shuffle_and_elect") {
      if (thread_extent_.defined() &&
          StructuralEqual()(thread_extent_, op->value)) {
        return StmtExprMutator::VisitStmt(op->body);
      }
      thread_extent_ = op->value;
      // if (!thread_extent_.defined()) {
      return StmtExprMutator::VisitStmt(op->body);
      // }
    }
    return StmtExprMutator::VisitStmt_(op);
  }

  PrimExpr thread_extent_;
};

using namespace tir::transform;
tvm::transform::Pass MergeIfStmt() {
  auto pass_func = [=](PrimFunc f, IRModule m, PassContext ctx) {
    return MergeIfStmtRewriter::Substitute(f);
  };
  return CreatePrimFuncPass(pass_func, 0, "tl.MergeIfStmt", {});
}

TVM_REGISTER_GLOBAL("tl.transform.MergeIfStmt").set_body_typed(MergeIfStmt);

} // namespace tl
} // namespace tvm
