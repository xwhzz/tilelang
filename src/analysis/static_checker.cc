/*!
 * \file warp_specialized_rewriter.h
 * \brief tools for warp-specialized-related analysis and transformation
 */

#include "arith/ir_visitor_with_analyzer.h"
#include <tvm/ffi/reflection/registry.h>
#include <tvm/tir/analysis.h>
#include <tvm/tir/builtin.h>
#include <tvm/tir/op.h>


namespace tvm {
namespace tl {

using namespace tir;
using arith::IRVisitorWithAnalyzer;

class StaticChecker : public IRVisitorWithAnalyzer {
public:

  static bool Check(const Stmt &stmt) {
    StaticChecker checker;
    checker(stmt);
    return checker.is_static_;
  }

  StaticChecker() = default;

private:
  bool check_buffer_is_static(const tir::Buffer &bf) {
    for (const auto &dim : bf->shape) {
      if (!dim.as<IntImmNode>()) {
        return false;
      }
    }
    return true;
  }

  void VisitStmt_(const BlockRealizeNode *op) final {
    const BlockNode *block = op->block.as<BlockNode>();
    for (const auto &bf : block->alloc_buffers) {
      if (!check_buffer_is_static(bf)) {
        is_static_ = false;
        return;
      }
    }
    for (const auto &match_buffer : block->match_buffers) {
      const tir::Buffer &bf = match_buffer->buffer;
      if (!check_buffer_is_static(bf)) {
        is_static_ = false;
        return;
      }
    }
    IRVisitorWithAnalyzer::VisitStmt_(op);
  }

  void VisitStmt_(const ForNode *op) final {
    auto extent = analyzer_.Simplify(op->extent);

    if (!extent.as<IntImmNode>()) {
      is_static_ = false;
      return;
    }
    IRVisitorWithAnalyzer::VisitStmt_(op);
  }

  void VisitStmt_(const IfThenElseNode *op) final {
    if (!analyzer_.CanProve(op->condition)) {
        is_static_ = false;
        return;
    }
    IRVisitorWithAnalyzer::VisitStmt_(op);
  }

  bool is_static_ = true;
};

bool CheckStatic(const tir::PrimFunc& func) {
  return StaticChecker::Check(func->body);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.analysis.CheckStatic", CheckStatic);
}

} // namespace tl
} // namespace tvm
