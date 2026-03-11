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
 * \file reduce_at.cc
 * \brief Implements the ReduceAt schedule primitive for tilelang.
 *
 * Given a source block (whose write buffer is the reduction source),
 * a destination block (or the same block), a loop level, reduction type,
 * and dimension, this primitive inserts a `tl.tileop.reduce` statement.
 *
 * The reduce operation reads from the source buffer's region and writes
 * to the destination buffer's region, performing the specified reduction
 * (sum, max, min, abssum, absmax) along the given dimension.
 *
 * The primitive supports two insertion modes:
 * - Append mode (default): append reduce statement to the end of loop body.
 * - Replace mode: replace the loop body with only the reduce statement.
 *
 * This is essential for:
 * - General reduction templates (softmax, layernorm, etc.)
 * - Cross-thread reductions within a tile
 * - Multi-stage reduction pipelines
 */

#include <tvm/arith/analyzer.h>
#include <tvm/arith/int_set.h>
#include <tvm/tir/op.h>
#include <tvm/tir/stmt_functor.h>

#include "tir/schedule/analysis.h"
#include "tir/schedule/transform.h"
#include "tir/schedule/utils.h"

#include "utils.h"

namespace tvm {
namespace tl {

using namespace tir;
using support::NDIntSet;

// ---------------------------------------------------------------------------
// BlockRealizeReplacer: replace the first BlockRealize that wraps `target_`
// with `replacement_`, preserving other statements in the loop body.
// ---------------------------------------------------------------------------
class BlockRealizeReplacer : public StmtMutator {
 public:
  BlockRealizeReplacer(const BlockNode* target, Stmt replacement)
      : target_(target), replacement_(std::move(replacement)), replaced_(false) {}

  Stmt VisitStmt(const Stmt& stmt) final {
    return replaced_ ? stmt : StmtMutator::VisitStmt(stmt);
  }

  Stmt VisitStmt_(const BlockRealizeNode* op) final {
    if (op->block.get() == target_) {
      replaced_ = true;
      return replacement_;
    }
    return StmtMutator::VisitStmt_(op);
  }

  bool replaced() const { return replaced_; }

 private:
  const BlockNode* target_;
  Stmt replacement_;
  bool replaced_;
};

// ---------------------------------------------------------------------------
// DirectBlockLoopReplacer: replace the first For whose body directly contains
// the target BlockRealize with `replacement_`.
// This is used to collapse an inner serial reduction loop while preserving
// sibling statements in the outer loop body (e.g. cached T.copy).
// ---------------------------------------------------------------------------
class DirectBlockLoopReplacer : public StmtMutator {
 public:
  DirectBlockLoopReplacer(const BlockNode* target, Stmt replacement)
      : target_(target), replacement_(std::move(replacement)), replaced_(false) {}

  Stmt VisitStmt(const Stmt& stmt) final {
    return replaced_ ? stmt : StmtMutator::VisitStmt(stmt);
  }

  Stmt VisitStmt_(const ForNode* op) final {
    if (BodyDirectlyContainsTarget(op->body, target_)) {
      replaced_ = true;
      return replacement_;
    }
    return StmtMutator::VisitStmt_(op);
  }

  bool replaced() const { return replaced_; }

 private:
  static bool BodyDirectlyContainsTarget(const Stmt& body,
                                         const BlockNode* target) {
    if (const auto* realize = body.as<BlockRealizeNode>()) {
      return realize->block.get() == target;
    }
    if (const auto* seq = body.as<SeqStmtNode>()) {
      for (const Stmt& s : seq->seq) {
        if (const auto* realize = s.as<BlockRealizeNode>()) {
          if (realize->block.get() == target) return true;
        }
      }
    }
    return false;
  }

  const BlockNode* target_;
  Stmt replacement_;
  bool replaced_;
};

// ---------------------------------------------------------------------------
// Helper: Compute the relaxed access region of a buffer within a loop.
// ---------------------------------------------------------------------------
static ffi::Array<Range> ComputeRelaxedRegion(
    ScheduleState self, const StmtSRef& loop_sref,
    const StmtSRef& block_sref, const Buffer& buf,
    BufferIndexType buffer_type) {
  const BlockNode* block = TVM_SREF_TO_BLOCK(block_sref);

  BlockRealize realize = GetBlockRealize(self, block_sref);
  ffi::Map<Var, PrimExpr> bindings = GetBindings(realize);

  runtime::StorageScope scope = runtime::StorageScope::Create("local");
  ffi::Map<Var, arith::IntSet> var_dom = arith::AsIntSet(LoopDomainOfSRefTreePathSkipBlocks(
      ffi::GetRef<StmtSRef>(self->stmt2ref.at(block)->parent),
      loop_sref, scope));

  const auto& regions = (buffer_type == BufferIndexType::kRead)
                            ? block->reads : block->writes;

  std::vector<NDIntSet> relaxed_regions;
  for (const BufferRegion& buffer_region : regions) {
    if (buffer_region->buffer.same_as(buf)) {
      ffi::Array<arith::IntSet> relaxed =
          arith::EvalSet(Substitute(buffer_region->region, bindings), var_dom);
      relaxed_regions.push_back({relaxed.begin(), relaxed.end()});
    }
  }
  ICHECK(!relaxed_regions.empty())
      << "ValueError: buffer " << buf->name
      << " is not accessed in the specified block";

  NDIntSet unified = support::NDIntSetUnion(relaxed_regions);
  int ndim = static_cast<int>(unified.size());

  arith::Analyzer analyzer;
  ffi::Array<Range> result;
  result.reserve(ndim);
  for (int d = 0; d < ndim; ++d) {
    PrimExpr mn = analyzer.Simplify(unified[d].min());
    PrimExpr mx = analyzer.Simplify(unified[d].max());
    PrimExpr extent = analyzer.Simplify(mx - mn + 1);
    result.push_back(Range::FromMinExtent(mn, extent));
  }
  return result;
}

// ---------------------------------------------------------------------------
// ReduceAt: main entry point
//
// Inserts a tl.tileop.reduce statement at the end of the specified loop's
// body.  The source is a block's read buffer and the destination is the
// block's write buffer.
// ---------------------------------------------------------------------------
static void ReduceAt(ScheduleState self, const StmtSRef& loop_sref,
                     const StmtSRef& block_sref,
                     int read_buffer_index, int write_buffer_index,
                     const ffi::String& reduce_type, int dim, bool clear,
                     bool replace_loop_body) {
  // ---- Step 1: Obtain source and destination buffers -----------------------
  const BlockNode* block = TVM_SREF_TO_BLOCK(block_sref);
  Block block_ref = ffi::GetRef<Block>(block);

  Buffer src = GetNthAccessBuffer(self, block_ref, read_buffer_index,
                                  BufferIndexType::kRead);
  Buffer dst = GetNthAccessBuffer(self, block_ref, write_buffer_index,
                                  BufferIndexType::kWrite);

  const ForNode* loop = TVM_SREF_TO_FOR(loop_sref);

  // ---- Step 2: Compute the relaxed regions --------------------------------
  ffi::Array<Range> src_region =
      ComputeRelaxedRegion(self, loop_sref, block_sref, src,
                           BufferIndexType::kRead);
  ffi::Array<Range> dst_region =
      ComputeRelaxedRegion(self, loop_sref, block_sref, dst,
                           BufferIndexType::kWrite);

  // ---- Step 3: Build the T.reduce call ------------------------------------
  PrimExpr src_region_arg = MakeRegionCall(src, src_region, /*access_mask=*/1);
  PrimExpr dst_region_arg = MakeRegionCall(dst, dst_region, /*access_mask=*/2);

  Stmt reduce_stmt = Evaluate(
      Call(DataType::Handle(), Op::Get("tl.tileop.reduce"),
           {src_region_arg, dst_region_arg,
            StringImm(reduce_type),
            IntImm(DataType::Int(32), dim),
            Bool(clear)}));

  // ---- Step 4: Update loop body -------------------------------------------
  ObjectPtr<ForNode> new_loop_node = ffi::make_object<ForNode>(*loop);
  if (replace_loop_body) {
    // Try to collapse the innermost loop that directly wraps the block.
    DirectBlockLoopReplacer loop_replacer(block, reduce_stmt);
    Stmt replaced_body = loop_replacer(loop->body);
    if (loop_replacer.replaced()) {
      new_loop_node->body = replaced_body;
    } else {
      // Fallback: replace only the block realize.
      BlockRealizeReplacer block_replacer(block, reduce_stmt);
      replaced_body = block_replacer(loop->body);
      // Final fallback: replace the whole loop body.
      new_loop_node->body = block_replacer.replaced() ? replaced_body
                                                      : reduce_stmt;
    }
  } else {
    ffi::Array<Stmt> subtrees = AsArray(loop->body);
    subtrees.push_back(reduce_stmt);
    new_loop_node->body = subtrees.size() == 1 ? subtrees[0] : SeqStmt(subtrees);
  }
  For new_loop(new_loop_node);

  // ---- Step 5: Replace in the scope root block ----------------------------
  StmtSRef scope_root_sref =
      GetScopeRoot(self, loop_sref, /*require_stage_pipeline=*/false);
  const BlockNode* scope_block = TVM_SREF_TO_BLOCK(scope_root_sref);

  ffi::Map<Block, Block> block_sref_reuse;
  Block new_scope_block = Downcast<Block>(
      LoopReplacer(loop, new_loop)(ffi::GetRef<Block>(scope_block)));

  block_sref_reuse.Set(ffi::GetRef<Block>(scope_block), new_scope_block);
  self->Replace(scope_root_sref, new_scope_block, block_sref_reuse);
}

// ---------------------------------------------------------------------------
// FFI Registration
// ---------------------------------------------------------------------------
TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def(
      "tl.schedule.ScheduleReduceAt",
      [](Schedule self, const LoopRV& loop_rv, const BlockRV& block_rv,
         int read_buffer_index, int write_buffer_index,
         const ffi::String& reduce_type, int dim, bool clear,
         bool replace_loop_body) {
        ReduceAt(self->state(), self->GetSRef(loop_rv),
                 self->GetSRef(block_rv), read_buffer_index,
                 write_buffer_index, reduce_type, dim, clear,
                 replace_loop_body);
      });
}

}  // namespace tl
}  // namespace tvm
