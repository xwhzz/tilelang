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
 * \file fill_at.cc
 * \brief Implements the FillAt schedule primitive for tilelang.
 *
 * Given a block and its write buffer index, a loop where the fill should
 * reside, and a fill value, this primitive:
 *
 * 1. Analyzes the buffer write region within one iteration of the
 *    specified loop (by relaxing over all inner loop variables).
 * 2. Emits a T.fill (tl.tileop.fill) statement that initializes the
 *    accessed region of the buffer to the given value.
 * 3. Inserts the fill statement at the beginning of the loop body.
 *
 * This is essential for reduction patterns where an accumulator buffer
 * must be initialized before the reduction loop.
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
// FillAt: main entry point
// ---------------------------------------------------------------------------
static void FillAt(ScheduleState self, const StmtSRef& loop_sref,
                   const StmtSRef& block_sref, int write_buffer_index,
                   double value) {
  // ---- Step 1: Obtain the write buffer and loop ----------------------------
  const BlockNode* block = TVM_SREF_TO_BLOCK(block_sref);
  Block block_ref = ffi::GetRef<Block>(block);
  Buffer buf = GetNthAccessBuffer(self, block_ref, write_buffer_index,
                                  BufferIndexType::kWrite);

  const ForNode* loop = TVM_SREF_TO_FOR(loop_sref);

  // ---- Step 2: Gather inner-loop domains and block bindings ----------------
  BlockRealize realize = GetBlockRealize(self, block_sref);
  ffi::Map<Var, PrimExpr> bindings = GetBindings(realize);

  runtime::StorageScope scope = runtime::StorageScope::Create("local");
  ffi::Map<Var, arith::IntSet> var_dom = arith::AsIntSet(LoopDomainOfSRefTreePath(
      /*low_inclusive=*/ffi::GetRef<StmtSRef>(self->stmt2ref.at(block)->parent),
      /*high_exclusive=*/loop_sref,
      /*extra_relax_scope=*/scope));

  // ---- Step 3: Relax the buffer write region over the inner loops ----------
  std::vector<NDIntSet> relaxed_regions;
  for (const BufferRegion& buffer_region : block->writes) {
    if (buffer_region->buffer.same_as(buf)) {
      ffi::Array<arith::IntSet> relaxed =
          arith::EvalSet(Substitute(buffer_region->region, bindings), var_dom);
      relaxed_regions.push_back({relaxed.begin(), relaxed.end()});
    }
  }
  ICHECK(!relaxed_regions.empty())
      << "ValueError: buffer " << buf->name
      << " is not written in the specified block";

  NDIntSet unified = support::NDIntSetUnion(relaxed_regions);
  int ndim = static_cast<int>(unified.size());

  arith::Analyzer analyzer;
  ffi::Array<Range> fill_region;
  fill_region.reserve(ndim);

  for (int d = 0; d < ndim; ++d) {
    PrimExpr mn = analyzer.Simplify(unified[d].min());
    PrimExpr mx = analyzer.Simplify(unified[d].max());
    PrimExpr extent = analyzer.Simplify(mx - mn + 1);
    fill_region.push_back(Range::FromMinExtent(mn, extent));
  }

  // ---- Step 4: Build the T.fill call ---------------------------------------
  PrimExpr region_arg = MakeRegionCall(buf, fill_region, /*access_mask=*/2);
  PrimExpr fill_value = make_const(buf->dtype, value);
  Stmt fill_stmt = Evaluate(
      Call(DataType::Handle(), Op::Get("tl.tileop.fill"),
           {region_arg, fill_value}));

  // ---- Step 5: Insert the fill at the beginning of the loop body -----------
  ffi::Array<Stmt> subtrees = AsArray(loop->body);
  subtrees.insert(subtrees.begin(), fill_stmt);

  ObjectPtr<ForNode> new_loop_node = ffi::make_object<ForNode>(*loop);
  new_loop_node->body = subtrees.size() == 1 ? subtrees[0] : SeqStmt(subtrees);
  For new_loop(new_loop_node);

  // ---- Step 6: Replace in the scope root block -----------------------------
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
      "tl.schedule.ScheduleFillAt",
      [](Schedule self, const LoopRV& loop_rv, const BlockRV& block_rv,
         int write_buffer_index, double value) {
        FillAt(self->state(), self->GetSRef(loop_rv),
               self->GetSRef(block_rv), write_buffer_index, value);
      });
}

}  // namespace tl
}  // namespace tvm
