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
