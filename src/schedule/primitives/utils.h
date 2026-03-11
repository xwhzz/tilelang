/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
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
 * \file primitives/utils.h
 * \brief Shared utilities for tilelang schedule primitives.
 */

#ifndef TVM_TL_SCHEDULE_PRIMITIVES_UTILS_H_
#define TVM_TL_SCHEDULE_PRIMITIVES_UTILS_H_

#include <tvm/tir/op.h>
#include <tvm/tir/stmt_functor.h>

#include "tir/schedule/analysis.h"

#include "../../op/region.h"

namespace tvm {
namespace tl {

using namespace tir;

// ---------------------------------------------------------------------------
// MakeRegionCall: construct a tl.region() Call that encodes a BufferRegion as
// a PrimExpr for passing as an argument to tl.tileop.{copy,fill,reduce}.
// ---------------------------------------------------------------------------
static inline PrimExpr MakeRegionCall(const Buffer& buf,
                                      const ffi::Array<Range>& ranges,
                                      int access_mask) {
  ffi::Array<PrimExpr> args;
  ffi::Array<PrimExpr> min_indices;
  for (const auto& range : ranges) {
    min_indices.push_back(range->min);
  }
  args.push_back(BufferLoad(buf, min_indices));
  args.push_back(IntImm(DataType::Int(32), access_mask));
  for (const auto& range : ranges) {
    args.push_back(range->extent);
  }
  return Call(DataType::Handle(), RegionOp::Get(), args);
}

// ---------------------------------------------------------------------------
// LoopDomainOfSRefTreePathSkipBlocks: variant of TVM's loop-domain collector
// that keeps traversing through intermediate Block/BlockRealize wrappers.
//
// TileLang primitives often wrap loop bodies in opaque allocation blocks.
// Subsequent primitives still need to relax loop vars under the target loop,
// but TVM's default helper stops at the first non-For ancestor.  That makes
// later cache/reduce primitives think there are no inner loops, collapsing
// multi-element tiles to scalars.
// ---------------------------------------------------------------------------
static inline ffi::Map<Var, Range> LoopDomainOfSRefTreePathSkipBlocks(
    const StmtSRef& low_inclusive, const ffi::Optional<StmtSRef>& high_exclusive,
    const runtime::StorageScope& extra_relax_scope) {
  ffi::Map<Var, Range> result;
  const StmtSRefNode* p = low_inclusive.get();
  const StmtSRefNode* limit = static_cast<const StmtSRefNode*>(high_exclusive.get());
  for (; p != limit; p = p->parent) {
    if (const ForNode* loop = p->StmtAs<ForNode>()) {
      result.Set(loop->loop_var, Range::FromMinExtent(loop->min, loop->extent));
    }
  }
  if (extra_relax_scope.rank != runtime::StorageRank::kGlobal) {
    for (; p; p = p->parent) {
      if (const ForNode* loop = p->StmtAs<ForNode>()) {
        if (loop->kind == ForKind::kThreadBinding) {
          const ffi::String& thread_tag = loop->thread_binding.value()->thread_tag;
          if (CanRelaxStorageUnderThread(
                  extra_relax_scope, runtime::ThreadScope::Create(thread_tag))) {
            result.Set(loop->loop_var, Range::FromMinExtent(loop->min, loop->extent));
          }
        }
      }
    }
  }
  return result;
}

// ---------------------------------------------------------------------------
// LoopReplacer: rewrite a specific ForNode in the AST with a new one.
// ---------------------------------------------------------------------------
class LoopReplacer : public StmtMutator {
 public:
  LoopReplacer(const ForNode* old_loop, For new_loop)
      : old_loop_(old_loop), new_loop_(std::move(new_loop)), found_(false) {}

  Stmt VisitStmt(const Stmt& stmt) final {
    return found_ ? stmt : StmtMutator::VisitStmt(stmt);
  }
  Stmt VisitStmt_(const ForNode* loop) final {
    if (loop == old_loop_) {
      found_ = true;
      return new_loop_;
    }
    return StmtMutator::VisitStmt_(loop);
  }

 private:
  const ForNode* old_loop_;
  For new_loop_;
  bool found_;
};

}  // namespace tl
}  // namespace tvm

#endif  // TVM_TL_SCHEDULE_PRIMITIVES_UTILS_H_
