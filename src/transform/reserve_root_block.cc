/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership. The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \file lower_opaque_block.cc
 */

#include <tvm/ffi/reflection/registry.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>

#include "tir/transforms/ir_utils.h"

namespace tvm {
namespace tl {
using namespace tir;

/*!
 * \brief Remove Block to ensure that the TIR can not be scheduled again.
 */
class RootBlockReserver : public StmtExprMutator {
 public:
  static Stmt Rewrite(Stmt body) {
    RootBlockReserver reserver;
    reserver.storage_align_ = CollectStorageAlignAnnotation(body);
    return reserver(std::move(body));
  }

 private:
  Stmt VisitStmt_(const BlockRealizeNode* op) final {
    // We have convert blocks into opaque blocks in previous passes.
    ICHECK(op->iter_values.empty()) << "Non-opaque blocks are not allowed in FlattenBuffer. Please "
                                       "call pass ConvertBlocksToOpaque before.";
    // Step 1. Visit the body
    block_level_++;
    Block new_block = Downcast<Block>(this->VisitStmt(op->block));
    block_level_--;
    PrimExpr predicate = this->VisitExpr(op->predicate);
    // Step 2. Transform the `predicate` to if-then-else
    Stmt body = new_block->body;
    if (!is_one(predicate) && block_level_ != 0) {
      body = IfThenElse(predicate, std::move(body));
    }

    for (size_t i = 0; i < new_block->alloc_buffers.size(); ++i) {
      allocated_buffers_.insert(new_block->alloc_buffers[i]);
    }

    // Step 4. Handle annotations, block annotations are not preserved by default.
    std::vector<std::pair<std::string, PrimExpr>> pragma_attrs;
    HandleAnnotations(new_block->annotations, &pragma_attrs, /*is_block=*/true);
    for (auto it = pragma_attrs.rbegin(); it != pragma_attrs.rend(); ++it) {
      body = AttrStmt(Integer(0), it->first, it->second, std::move(body));
    }

    if (block_level_ == 0) {
      // 最外层的我们不想让其也lower
      auto p_block = new_block.CopyOnWrite();
      p_block->name_hint = "tilelang_root";
      // 遍历 alloc_buffers_，把它们添加到 new_block 的 alloc_buffers 中
      p_block->alloc_buffers = ffi::Array<Buffer>(allocated_buffers_.begin(), allocated_buffers_.end());
      p_block->body = std::move(body);
      Stmt block_realize = BlockRealize(ffi::Array<PrimExpr>(), std::move(predicate), std::move(new_block));
      
      for (auto it = thread_bindings_.rbegin(); it != thread_bindings_.rend(); ++it) {
        block_realize = MakeLaunchThread(std::get<0>(*it), std::get<1>(*it), std::get<2>(*it),
                                        std::get<3>(*it), std::move(block_realize));
      }
      Block root_block = Block(ffi::Array<IterVar>(), {}, {}, {}, block_realize);
      return BlockRealize(ffi::Array<PrimExpr>(), const_true(), std::move(root_block));
    }

    return body;
  }

  Stmt VisitStmt_(const ForNode* op) final {
    // Step 1. Update unit loop info.
    PrimExpr min = this->VisitExpr(op->min);
    PrimExpr extent = this->VisitExpr(op->extent);
    if (is_one(extent) && op->annotations.empty()) {
      // handling unit loop
      unit_loop_vars_[op->loop_var] = min;
    }

    // Step 2. Visit recursively
    Stmt body = this->VisitStmt(op->body);

    // Step 3. Handle annotations
    std::vector<std::pair<std::string, PrimExpr>> pragma_attrs;
    ffi::Map<ffi::String, ffi::Any> new_annotations =
        HandleAnnotations(op->annotations, &pragma_attrs, /*is_block=*/false);
    // Step 4. Create new For loop accordingly
    if (op->kind == ForKind::kThreadBinding) {
      // Case 1. Thread binding
      ICHECK(op->thread_binding.defined());
      ffi::String thread_tag = op->thread_binding.value()->thread_tag;
      // body = MakeLaunchThread(min, extent, op->loop_var, thread_tag, body);
      thread_bindings_.emplace_back(min, extent, op->loop_var, thread_tag);
    } else if (is_one(extent) && op->annotations.empty() &&
               !op->annotations.count(::tvm::tir::attr::irregular_loop_mark)) {
      // Case 2. Unit loop
      return body;
    } else {
      // Case 3. An ordinary loop
      body = For(op->loop_var, std::move(min), std::move(extent), op->kind, std::move(body),
                 std::nullopt, new_annotations, op->step);
    }
    // Step 5. Insert nested attrs
    for (auto it = pragma_attrs.rbegin(); it != pragma_attrs.rend(); ++it) {
      body = AttrStmt(op->loop_var, it->first, it->second, std::move(body));
    }
    return body;
  }

  PrimExpr VisitExpr_(const VarNode* op) final {
    Var var = ffi::GetRef<Var>(op);
    auto it = unit_loop_vars_.find(var);
    if (it == unit_loop_vars_.end()) {
      return var;

    } else {
      PrimExpr expr = it->second;
      if (expr.dtype() != var.dtype()) {
        expr = tvm::cast(var.dtype(), std::move(expr));
      }
      return expr;
    }
  }

  static Stmt MakeLaunchThread(PrimExpr min, PrimExpr extent, Var var, ffi::String thread_tag,
                               Stmt body) {
    IterVar iter_var(/*dom=*/Range::FromMinExtent(min, extent),
                     /*var=*/std::move(var),
                     /*iter_type=*/IterVarType::kThreadIndex,
                     /*thread_tag=*/thread_tag);
    ffi::String attr_key = (thread_tag == "vthread" || thread_tag == "vthread.x" ||
                            thread_tag == "vthread.y" || thread_tag == "vthread.z")
                               ? ::tvm::tir::attr::virtual_thread
                               : ::tvm::tir::attr::thread_extent;
    return AttrStmt(/*node=*/std::move(iter_var),
                    /*attr_key=*/std::move(attr_key),
                    /*value=*/std::move(extent),
                    /*body=*/std::move(body));
  }

  /*! \brief Convert attr value from annotation map into PrimExpr. */
  PrimExpr ConvertAttrValue(const ffi::String& key, const Any& obj) {
    if (obj == nullptr) {
      return PrimExpr();
    } else if (auto expr = obj.try_cast<PrimExpr>()) {
      return expr.value();
    } else if (auto str = obj.try_cast<ffi::String>()) {
      return std::move(StringImm(str.value()));
    } else {
      LOG(FATAL) << "Illegal attribute of key " << key << ", value type " << obj.GetTypeKey()
                 << " not supported";
      return PrimExpr();
    }
  }

  /*!
   * \brief Helper to handle annotation dict.
   * (1) if the attr key is prefixed by `pragma_`, move to ordered kv list. They
   * are lowered to `AttrStmt` by legacy TE schedule convention.
   * (2) the non-pragma loop annotations are preserved
   * (3) the non-pragma block annotations are dropped
   * \return New annotation dict with preserved keys. Also update pragma attr pairs ordered by key.
   */
  ffi::Map<ffi::String, ffi::Any> HandleAnnotations(
      const ffi::Map<ffi::String, ffi::Any>& annotations,
      std::vector<std::pair<std::string, PrimExpr>>* pragma_attrs, bool is_block) {
    ffi::Map<ffi::String, ffi::Any> preserved_annotations;
    pragma_attrs->clear();
    for (const auto& kv : annotations) {
      const ffi::String& key = kv.first;
      if (::tvm::tir::attr::IsPragmaKey(key)) {
        pragma_attrs->emplace_back(key, ConvertAttrValue(key, kv.second));
      } else if (!is_block) {
        // the loop annotation is preserved
        preserved_annotations.Set(key, kv.second);
      }
    }
    std::sort(pragma_attrs->begin(), pragma_attrs->end(),
              [](const auto& p1, const auto& p2) { return p1.first < p2.first; });
    return preserved_annotations;
  }

  /*! \brief Record the loop_var and loop start value of unit loops, whose extent is one. */
  std::unordered_map<Var, PrimExpr> unit_loop_vars_;

  /*! \brief Attr keys to preserve into loop annotations. */
  std::unordered_set<std::string> preserved_annotations_;

  /*! \brief The map from buffer var to its storage alignment information. */
  std::unordered_map<Var, StorageAlignAnnotation> storage_align_;

  std::unordered_set<Buffer, ObjectPtrHash, ObjectPtrEqual> allocated_buffers_;

  std::vector<std::tuple<PrimExpr, PrimExpr, Var, ffi::String>> thread_bindings_;
  int block_level_ = 0;
};

PrimFunc ReserveRootBlock(PrimFunc f) {
  auto fptr = f.CopyOnWrite();
  fptr->body = RootBlockReserver::Rewrite(std::move(fptr->body));
  return f;
}
using namespace tir::transform;

namespace transform {
Pass ReserveRootBlock() {
  auto pass_func = [=](PrimFunc f, IRModule m, PassContext ctx) {
    return ::tvm::tl::ReserveRootBlock(std::move(f));
  };
  return CreatePrimFuncPass(pass_func, 0, "tl.ReserveRootBlock", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.transform.ReserveRootBlock", ReserveRootBlock);
}
}  // namespace transform

}  // namespace tir
}  // namespace tvm
