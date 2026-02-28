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
#include <tvm/tir/op.h>
#include <tvm/tir/schedule/schedule.h>
#include <tvm/tir/schedule/state.h>
#include <tvm/tir/stmt_functor.h>

#include "tir/schedule/utils.h"

namespace tvm {
namespace tl {
using namespace tir;

// ---------------------------------------------------------------------------
// LaunchThread:  wrap a block's body in a For(kThreadBinding) loop.
//
// Uses the sref-tree–aware Replace() method so that the schedule state
// (stmt2ref, block_info, …) stays consistent for subsequent primitives.
// ---------------------------------------------------------------------------
static void LaunchThread(ScheduleState self, const StmtSRef& block_sref,
                         int num_threads) {
  const BlockNode* block = TVM_SREF_TO_BLOCK(block_sref);

  // Build the thread-bound loop wrapping the block body.
  Var tx("tx");
  IterVar thread_iter(Range(nullptr), Var("threadIdx.x"),
                      kThreadIndex, "threadIdx.x");
  Stmt new_body = For(tx, 0, num_threads, ForKind::kThreadBinding,
                      block->body, thread_iter, {}, std::nullopt);

  // Copy the block, replace its body with the new loop.
  ObjectPtr<BlockNode> new_block_node = ffi::make_object<BlockNode>(*block);
  new_block_node->body = std::move(new_body);
  Block new_block(new_block_node);

  // Tell Replace to keep the sref for this block.
  ffi::Map<Block, Block> block_sref_reuse;
  block_sref_reuse.Set(ffi::GetRef<Block>(block), new_block);
  self->Replace(block_sref, new_block, block_sref_reuse);
}

// ---------------------------------------------------------------------------
// FFI Registration
// ---------------------------------------------------------------------------
TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def(
      "tl.schedule.ScheduleLaunchThread",
      [](Schedule self, const BlockRV& block_rv, int num_threads) {
        LaunchThread(self->state(), self->GetSRef(block_rv), num_threads);
      });
}

}  // namespace tl
}  // namespace tvm
