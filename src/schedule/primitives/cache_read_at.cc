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
 * \file cache_read_at.cc
 * \brief Implements the CacheReadAt / CacheWriteAt schedule primitive for
 *        tilelang.  Given a consumer block, a buffer index, a loop where the
 *        cache should reside, and a target storage scope, this primitive:
 *
 *        1. Analyzes the buffer access region within one iteration of the
 *           specified loop (by relaxing over all inner loop variables).
 *        2. Creates a compact cache buffer whose shape equals the per-iteration
 *           access extents.
 *        3. Emits a T.copy (tl.tileop.copy) statement that transfers the
 *           accessed region from the original buffer into the cache.
 *        4. Rewrites all references to the original buffer inside the loop body
 *           so they use the cache buffer with shifted indices.
 *        5. Allocates the cache buffer locally inside the loop via an opaque
 *           Block wrapper, keeping it scoped to one iteration.
 */

#include <tvm/arith/analyzer.h>
#include <tvm/arith/int_set.h>
#include <tvm/tir/op.h>
#include <tvm/tir/stmt_functor.h>

// TVM internal headers (included via ${TVM_SOURCE}/src in the include path)
#include "tir/schedule/analysis.h"
#include "tir/schedule/transform.h"
#include "tir/schedule/utils.h"

// TileLang headers
#include "../../op/region.h"

#include <string>
#include <vector>

namespace tvm {
namespace tl {

using namespace tir;
using support::NDIntSet;

// ---------------------------------------------------------------------------
// Helper: construct a tl.region() Call that encodes a BufferRegion as a
// PrimExpr so it can be passed as an argument to tl.tileop.copy.
// ---------------------------------------------------------------------------
static PrimExpr MakeRegionCall(const Buffer& buf, const ffi::Array<Range>& ranges,
                               int access_mask) {
  ffi::Array<PrimExpr> args;
  // arg 0: BufferLoad whose indices are the per-axis minima
  ffi::Array<PrimExpr> min_indices;
  for (const auto& range : ranges) {
    min_indices.push_back(range->min);
  }
  args.push_back(BufferLoad(buf, min_indices));
  // arg 1: access mask (1=read, 2=write)
  args.push_back(IntImm(DataType::Int(32), access_mask));
  // args 2+i: per-axis extents
  for (const auto& range : ranges) {
    args.push_back(range->extent);
  }
  return Call(DataType::Handle(), RegionOp::Get(), args);
}

// Detect whether the loop body already contains tl.tileop.region calls that
// reference `buf`. In that case, squeezing dimensions would require rewriting
// region extents as well; keep full rank to preserve consistency.
static bool HasTileRegionCallOnBuffer(const Stmt& body, const Buffer& buf) {
  bool found = false;
  PostOrderVisit(body, [&found, &buf](const ObjectRef& obj) {
    if (found) return;
    const auto* call = obj.as<CallNode>();
    if (call == nullptr) return;
    const auto* op = call->op.as<OpNode>();
    if (op == nullptr || op->name != "tl.tileop.region" || call->args.empty()) {
      return;
    }
    const auto* load = call->args[0].as<BufferLoadNode>();
    if (load != nullptr && load->buffer.same_as(buf)) {
      found = true;
    }
  });
  return found;
}

// Build an in-place square transform:
//   for ...:
//     dst[idx] = dst[idx] * dst[idx]
static Stmt BuildSquareInplaceStmt(const Buffer& dst,
                                   const ffi::Array<PrimExpr>& shape) {
  int ndim = static_cast<int>(shape.size());
  std::vector<Var> iters;
  iters.reserve(ndim);

  ffi::Array<PrimExpr> indices;
  indices.reserve(ndim);
  for (int d = 0; d < ndim; ++d) {
    Var it("ax" + std::to_string(d), DataType::Int(32));
    iters.push_back(it);
    indices.push_back(it);
  }

  PrimExpr val = BufferLoad(dst, indices);
  Stmt body = BufferStore(dst, val * val, indices);
  for (int d = ndim - 1; d >= 0; --d) {
    body = For(
        iters[d],
        /*min=*/IntImm(DataType::Int(32), 0),
        /*extent=*/shape[d],
        ForKind::kSerial,
        body);
  }
  return body;
}

// ---------------------------------------------------------------------------
// Visitor that replaces all accesses to `src_` with accesses to `dst_`,
// shifting the indices by subtracting the per-axis region minimums.
// ---------------------------------------------------------------------------
class CacheBufferReplacer : public StmtExprMutator {
 public:
  // kept_dims: indices of original dimensions that are kept (not squeezed).
  CacheBufferReplacer(const Buffer& src, const Buffer& dst,
                      const ffi::Array<PrimExpr>& region_mins,
                      const std::vector<int>& kept_dims,
                      ffi::Map<Block, Block>* block_sref_reuse,
                      const BlockNode* target_block = nullptr)
      : src_(src),
        dst_(dst),
        region_mins_(region_mins),
        kept_dims_(kept_dims),
        block_sref_reuse_(block_sref_reuse),
        target_block_(target_block),
        block_only_(target_block != nullptr),
        in_target_scope_(false) {}

 private:
  bool ShouldRewriteAccess() const {
    return !block_only_ || in_target_scope_;
  }

  // Build squeezed indices: for each kept dim, compute (original_idx - min).
  ffi::Array<PrimExpr> SqueezedIndices(const ffi::Array<PrimExpr>& indices) {
    ffi::Array<PrimExpr> new_indices;
    for (int d : kept_dims_) {
      new_indices.push_back(indices[d] - region_mins_[d]);
    }
    return new_indices;
  }

  PrimExpr VisitExpr_(const BufferLoadNode* _load) final {
    BufferLoad load = Downcast<BufferLoad>(StmtExprMutator::VisitExpr_(_load));
    if (ShouldRewriteAccess() && load->buffer.same_as(src_)) {
      ObjectPtr<BufferLoadNode> n = ffi::make_object<BufferLoadNode>(*load.get());
      n->buffer = dst_;
      n->indices = SqueezedIndices(n->indices);
      return BufferLoad(n);
    }
    return load;
  }

  Stmt VisitStmt_(const BufferStoreNode* _store) final {
    BufferStore store = Downcast<BufferStore>(StmtExprMutator::VisitStmt_(_store));
    if (ShouldRewriteAccess() && store->buffer.same_as(src_)) {
      ObjectPtr<BufferStoreNode> n = ffi::make_object<BufferStoreNode>(*store.get());
      n->buffer = dst_;
      n->indices = SqueezedIndices(n->indices);
      return BufferStore(n);
    }
    return store;
  }

  Stmt VisitStmt_(const BlockNode* _block) final {
    bool prev_in_target_scope = in_target_scope_;
    if (block_only_ && _block == target_block_) {
      in_target_scope_ = true;
    }
    Block old_block = ffi::GetRef<Block>(_block);
    Block block = Downcast<Block>(StmtExprMutator::VisitStmt_(_block));
    ObjectPtr<BlockNode> n = ffi::make_object<BlockNode>(*block.get());
    if (ShouldRewriteAccess()) {
      n->reads = ReplaceBufferWithShift(n->reads);
      n->writes = ReplaceBufferWithShift(n->writes);
    }
    Block new_block(n);
    block_sref_reuse_->Set(old_block, new_block);
    in_target_scope_ = prev_in_target_scope;
    return new_block;
  }

  ffi::Array<BufferRegion> ReplaceBufferWithShift(
      const ffi::Array<BufferRegion>& regions) {
    ffi::Array<BufferRegion> result;
    for (const auto& region : regions) {
      if (region->buffer.same_as(src_)) {
        ffi::Array<Range> new_ranges;
        for (int d : kept_dims_) {
          new_ranges.push_back(Range::FromMinExtent(
              region->region[d]->min - region_mins_[d],
              region->region[d]->extent));
        }
        result.push_back(BufferRegion(dst_, new_ranges));
      } else {
        result.push_back(region);
      }
    }
    return result;
  }

  const Buffer& src_;
  const Buffer& dst_;
  const ffi::Array<PrimExpr>& region_mins_;
  const std::vector<int>& kept_dims_;
  ffi::Map<Block, Block>* block_sref_reuse_;
  const BlockNode* target_block_;
  bool block_only_;
  bool in_target_scope_;
};

// ---------------------------------------------------------------------------
// Simple rewriter that replaces a specific ForNode in the AST with a new one.
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

// ---------------------------------------------------------------------------
// CacheReadAt:  main entry point
// ---------------------------------------------------------------------------
static void CacheReadAt(ScheduleState self, const StmtSRef& loop_sref,
                        const StmtSRef& block_sref, int read_buffer_index,
                        const ffi::String& storage_scope,
                        const ffi::String& transform) {
  // ---- Step 1: Obtain source buffer and loop --------------------------------
  const BlockNode* block = TVM_SREF_TO_BLOCK(block_sref);
  Block block_ref = ffi::GetRef<Block>(block);
  Buffer src =
      GetNthAccessBuffer(self, block_ref, read_buffer_index, BufferIndexType::kRead);

  const ForNode* loop = TVM_SREF_TO_FOR(loop_sref);

  // ---- Step 2: Gather inner-loop domains and block bindings -----------------
  BlockRealize realize = GetBlockRealize(self, block_sref);
  ffi::Map<Var, PrimExpr> bindings = GetBindings(realize);

  runtime::StorageScope scope = runtime::StorageScope::Create(storage_scope);
  ffi::Map<Var, arith::IntSet> var_dom = arith::AsIntSet(LoopDomainOfSRefTreePath(
      /*low_inclusive=*/ffi::GetRef<StmtSRef>(self->stmt2ref.at(block)->parent),
      /*high_exclusive=*/loop_sref,
      /*extra_relax_scope=*/scope));

  // ---- Step 3: Relax the buffer read region over the inner loops ------------
  std::vector<NDIntSet> relaxed_regions;
  for (const BufferRegion& buffer_region : block->reads) {
    if (buffer_region->buffer.same_as(src)) {
      ffi::Array<arith::IntSet> relaxed =
          arith::EvalSet(Substitute(buffer_region->region, bindings), var_dom);
      relaxed_regions.push_back({relaxed.begin(), relaxed.end()});
    }
  }
  ICHECK(!relaxed_regions.empty())
      << "ValueError: buffer " << src->name
      << " is not read in the specified block";

  // Union all relaxed regions
  NDIntSet unified = support::NDIntSetUnion(relaxed_regions);
  int ndim = static_cast<int>(unified.size());

  arith::Analyzer analyzer;
  ffi::Array<Range> cache_region;
  ffi::Array<PrimExpr> cache_shape;
  ffi::Array<PrimExpr> region_mins;
  cache_region.reserve(ndim);
  cache_shape.reserve(ndim);
  region_mins.reserve(ndim);

  for (int d = 0; d < ndim; ++d) {
    PrimExpr mn = analyzer.Simplify(unified[d].min());
    PrimExpr mx = analyzer.Simplify(unified[d].max());
    PrimExpr extent = analyzer.Simplify(mx - mn + 1);
    cache_region.push_back(Range::FromMinExtent(mn, extent));
    cache_shape.push_back(extent);
    region_mins.push_back(mn);
  }

  // ---- Step 4: Create the cache buffer --------------------------------------
  std::vector<int> kept_dims;
  ffi::Array<PrimExpr> squeezed_shape;
  bool keep_full_rank = HasTileRegionCallOnBuffer(loop->body, src);
  if (keep_full_rank) {
    kept_dims.reserve(ndim);
    squeezed_shape.reserve(ndim);
    for (int d = 0; d < ndim; ++d) {
      kept_dims.push_back(d);
      squeezed_shape.push_back(cache_shape[d]);
    }
  } else {
    // Default behavior: squeeze unit extents for compact cache allocation.
    for (int d = 0; d < ndim; ++d) {
      if (const auto* imm = cache_shape[d].as<IntImmNode>()) {
        if (imm->value == 1) continue;
      }
      kept_dims.push_back(d);
      squeezed_shape.push_back(cache_shape[d]);
    }
    // If all dims are 1, keep the last one to avoid 0-d buffer.
    if (kept_dims.empty()) {
      kept_dims.push_back(ndim - 1);
      squeezed_shape.push_back(cache_shape[ndim - 1]);
    }
  }

  Buffer dst = WithScope(src, storage_scope);
  {
    auto* w = dst.CopyOnWrite();
    w->shape = squeezed_shape;
    // Produce a readable name: e.g. a_shared_dyn  or  a_local_fragment
    std::string scope_suffix = storage_scope;
    // Replace '.' with '_' for valid identifier
    for (auto& c : scope_suffix) {
      if (c == '.') c = '_';
    }
    w->name = src->name + "_" + scope_suffix;
    // Strides can be empty (row-major) for the compact buffer.
    w->strides = {};
  }

  // ---- Step 5: Build the T.copy call ----------------------------------------
  // Source region: src[cache_region]  (read, full ndim)
  // Destination  : dst[0 : ext, ...]  (write, squeezed dims only)
  ffi::Array<Range> dst_ranges;
  dst_ranges.reserve(squeezed_shape.size());
  for (size_t i = 0; i < squeezed_shape.size(); ++i) {
    dst_ranges.push_back(
        Range::FromMinExtent(IntImm(DataType::Int(32), 0), squeezed_shape[i]));
  }

  PrimExpr src_region_arg = MakeRegionCall(src, cache_region, /*access_mask=*/1);
  PrimExpr dst_region_arg = MakeRegionCall(dst, dst_ranges, /*access_mask=*/2);

  Stmt copy_stmt = Evaluate(
      Call(DataType::Handle(), Op::Get("tl.tileop.copy"),
           {src_region_arg, dst_region_arg}));

  std::string transform_mode = transform;
  ICHECK(transform_mode.empty() || transform_mode == "square")
      << "ValueError: unsupported cache_read_at transform `" << transform_mode
      << "`, expected one of {\"\", \"square\"}";
  bool do_square_transform = transform_mode == "square";
  Stmt square_stmt;
  if (do_square_transform) {
    square_stmt = BuildSquareInplaceStmt(dst, squeezed_shape);
  }

  // ---- Step 6: Rewrite buffer references in the loop body -------------------
  ffi::Array<Stmt> subtrees = AsArray(loop->body);
  ffi::Map<Block, Block> block_sref_reuse;
  CacheBufferReplacer replacer(
      src, dst, region_mins, kept_dims, &block_sref_reuse,
      /*target_block=*/do_square_transform ? block : nullptr);
  for (int i = 0; i < static_cast<int>(subtrees.size()); ++i) {
    Stmt old_stmt = subtrees[i];
    subtrees.Set(i, Stmt(nullptr));
    subtrees.Set(i, replacer(std::move(old_stmt)));
  }

  // ---- Step 7: Insert copy + allocation into the loop body ------------------
  // Prepend the copy statement before the existing subtrees.
  subtrees.insert(subtrees.begin(), copy_stmt);
  if (do_square_transform) {
    subtrees.insert(subtrees.begin() + 1, square_stmt);
  }

  // Wrap everything in an opaque Block that owns the cache buffer allocation.
  Block alloc_block(
      /*iter_vars=*/{},
      /*reads=*/{},
      /*writes=*/{},
      /*name_hint=*/"",
      /*body=*/subtrees.size() == 1 ? subtrees[0] : SeqStmt(subtrees),
      /*init=*/std::nullopt,
      /*alloc_buffers=*/{dst});
  BlockRealize alloc_realize(
      /*values=*/{},
      /*predicate=*/const_true(),
      alloc_block);

  // Build the new For loop with the wrapper block as the body.
  ObjectPtr<ForNode> new_loop_node = ffi::make_object<ForNode>(*loop);
  new_loop_node->body = std::move(alloc_realize);
  For new_loop(new_loop_node);

  // ---- Step 8: Replace in the scope root block ------------------------------
  StmtSRef scope_root_sref =
      GetScopeRoot(self, loop_sref, /*require_stage_pipeline=*/false);
  const BlockNode* scope_block = TVM_SREF_TO_BLOCK(scope_root_sref);

  Block new_scope_block = Downcast<Block>(
      LoopReplacer(loop, new_loop)(ffi::GetRef<Block>(scope_block)));

  block_sref_reuse.Set(ffi::GetRef<Block>(scope_block), new_scope_block);
  self->Replace(scope_root_sref, new_scope_block, block_sref_reuse);
}

// ---------------------------------------------------------------------------
// CacheWriteAt:  mirror of CacheReadAt for write buffers
// ---------------------------------------------------------------------------
static void CacheWriteAt(ScheduleState self, const StmtSRef& loop_sref,
                         const StmtSRef& block_sref, int write_buffer_index,
                         const ffi::String& storage_scope, bool write_back) {
  // ---- Step 1: Obtain destination buffer and loop ---------------------------
  const BlockNode* block = TVM_SREF_TO_BLOCK(block_sref);
  Block block_ref = ffi::GetRef<Block>(block);
  Buffer src =
      GetNthAccessBuffer(self, block_ref, write_buffer_index, BufferIndexType::kWrite);

  const ForNode* loop = TVM_SREF_TO_FOR(loop_sref);

  // ---- Step 2: Gather inner-loop domains and block bindings -----------------
  BlockRealize realize = GetBlockRealize(self, block_sref);
  ffi::Map<Var, PrimExpr> bindings = GetBindings(realize);

  runtime::StorageScope scope = runtime::StorageScope::Create(storage_scope);
  ffi::Map<Var, arith::IntSet> var_dom = arith::AsIntSet(LoopDomainOfSRefTreePath(
      /*low_inclusive=*/ffi::GetRef<StmtSRef>(self->stmt2ref.at(block)->parent),
      /*high_exclusive=*/loop_sref,
      /*extra_relax_scope=*/scope));

  // ---- Step 3: Relax the buffer write region over the inner loops -----------
  std::vector<NDIntSet> relaxed_regions;
  for (const BufferRegion& buffer_region : block->writes) {
    if (buffer_region->buffer.same_as(src)) {
      ffi::Array<arith::IntSet> relaxed =
          arith::EvalSet(Substitute(buffer_region->region, bindings), var_dom);
      relaxed_regions.push_back({relaxed.begin(), relaxed.end()});
    }
  }
  ICHECK(!relaxed_regions.empty())
      << "ValueError: buffer " << src->name
      << " is not written in the specified block";

  NDIntSet unified = support::NDIntSetUnion(relaxed_regions);
  int ndim = static_cast<int>(unified.size());

  arith::Analyzer analyzer;
  ffi::Array<Range> cache_region;
  ffi::Array<PrimExpr> cache_shape;
  ffi::Array<PrimExpr> region_mins;
  cache_region.reserve(ndim);
  cache_shape.reserve(ndim);
  region_mins.reserve(ndim);

  for (int d = 0; d < ndim; ++d) {
    PrimExpr mn = analyzer.Simplify(unified[d].min());
    PrimExpr mx = analyzer.Simplify(unified[d].max());
    PrimExpr extent = analyzer.Simplify(mx - mn + 1);
    cache_region.push_back(Range::FromMinExtent(mn, extent));
    cache_shape.push_back(extent);
    region_mins.push_back(mn);
  }

  // ---- Step 4: Create the cache buffer --------------------------------------
  std::vector<int> kept_dims;
  ffi::Array<PrimExpr> squeezed_shape;
  bool keep_full_rank = HasTileRegionCallOnBuffer(loop->body, src);
  if (keep_full_rank) {
    kept_dims.reserve(ndim);
    squeezed_shape.reserve(ndim);
    for (int d = 0; d < ndim; ++d) {
      kept_dims.push_back(d);
      squeezed_shape.push_back(cache_shape[d]);
    }
  } else {
    // Default behavior: squeeze unit extents for compact cache allocation.
    for (int d = 0; d < ndim; ++d) {
      if (const auto* imm = cache_shape[d].as<IntImmNode>()) {
        if (imm->value == 1) continue;
      }
      kept_dims.push_back(d);
      squeezed_shape.push_back(cache_shape[d]);
    }
    if (kept_dims.empty()) {
      kept_dims.push_back(ndim - 1);
      squeezed_shape.push_back(cache_shape[ndim - 1]);
    }
  }

  Buffer dst = WithScope(src, storage_scope);
  {
    auto* w = dst.CopyOnWrite();
    w->shape = squeezed_shape;
    std::string scope_suffix = storage_scope;
    for (auto& c : scope_suffix) {
      if (c == '.') c = '_';
    }
    w->name = src->name + "_" + scope_suffix;
    w->strides = {};
  }

  // ---- Step 5: Build optional T.copy call (write-back: cache → original) ---
  ffi::Array<Range> dst_ranges;
  dst_ranges.reserve(squeezed_shape.size());
  for (size_t i = 0; i < squeezed_shape.size(); ++i) {
    dst_ranges.push_back(
        Range::FromMinExtent(IntImm(DataType::Int(32), 0), squeezed_shape[i]));
  }

  Stmt copy_stmt;
  if (write_back) {
    // For write-back: source is the cache, destination is the original buffer
    PrimExpr cache_region_arg = MakeRegionCall(dst, dst_ranges, /*access_mask=*/1);
    PrimExpr orig_region_arg = MakeRegionCall(src, cache_region, /*access_mask=*/2);
    copy_stmt = Evaluate(
        Call(DataType::Handle(), Op::Get("tl.tileop.copy"),
             {cache_region_arg, orig_region_arg}));
  }

  // ---- Step 6: Rewrite buffer references in the loop body -------------------
  ffi::Array<Stmt> subtrees = AsArray(loop->body);
  ffi::Map<Block, Block> block_sref_reuse;
  CacheBufferReplacer replacer(src, dst, region_mins, kept_dims, &block_sref_reuse);
  for (int i = 0; i < static_cast<int>(subtrees.size()); ++i) {
    Stmt old_stmt = subtrees[i];
    subtrees.Set(i, Stmt(nullptr));
    subtrees.Set(i, replacer(std::move(old_stmt)));
  }

  // ---- Step 7: Append the optional write-back copy --------------------------
  if (write_back) {
    subtrees.push_back(copy_stmt);
  }

  Block alloc_block(
      /*iter_vars=*/{},
      /*reads=*/{},
      /*writes=*/{},
      /*name_hint=*/"",
      /*body=*/subtrees.size() == 1 ? subtrees[0] : SeqStmt(subtrees),
      /*init=*/std::nullopt,
      /*alloc_buffers=*/{dst});
  BlockRealize alloc_realize(
      /*values=*/{},
      /*predicate=*/const_true(),
      alloc_block);

  ObjectPtr<ForNode> new_loop_node = ffi::make_object<ForNode>(*loop);
  new_loop_node->body = std::move(alloc_realize);
  For new_loop(new_loop_node);

  // ---- Step 8: Replace in the scope root block ------------------------------
  StmtSRef scope_root_sref =
      GetScopeRoot(self, loop_sref, /*require_stage_pipeline=*/false);
  const BlockNode* scope_block = TVM_SREF_TO_BLOCK(scope_root_sref);

  Block new_scope_block = Downcast<Block>(
      LoopReplacer(loop, new_loop)(ffi::GetRef<Block>(scope_block)));

  if (!write_back) {
    // Bridge/intermediate use-case: all consumers are rewritten to the cache
    // buffer, so the original tensor allocation can be removed from the
    // surrounding scope block.
    ObjectPtr<BlockNode> scope_ptr = ffi::make_object<BlockNode>(*new_scope_block.get());
    ffi::Array<Buffer> kept_alloc_buffers;
    kept_alloc_buffers.reserve(scope_ptr->alloc_buffers.size());
    for (const Buffer& buf : scope_ptr->alloc_buffers) {
      if (!buf.same_as(src)) {
        kept_alloc_buffers.push_back(buf);
      }
    }
    scope_ptr->alloc_buffers = std::move(kept_alloc_buffers);
    new_scope_block = Block(scope_ptr);
  }

  block_sref_reuse.Set(ffi::GetRef<Block>(scope_block), new_scope_block);
  self->Replace(scope_root_sref, new_scope_block, block_sref_reuse);
}

// ---------------------------------------------------------------------------
// FFI Registration
// ---------------------------------------------------------------------------
TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;

  refl::GlobalDef().def(
      "tl.schedule.ScheduleCacheReadAt",
      [](Schedule self, const LoopRV& loop_rv, const BlockRV& block_rv,
         int read_buffer_index, const ffi::String& storage_scope,
         const ffi::String& transform) {
        CacheReadAt(self->state(), self->GetSRef(loop_rv),
                    self->GetSRef(block_rv), read_buffer_index,
                    storage_scope, transform);
      });

  refl::GlobalDef().def(
      "tl.schedule.ScheduleCacheWriteAt",
      [](Schedule self, const LoopRV& loop_rv, const BlockRV& block_rv,
         int write_buffer_index, const ffi::String& storage_scope,
         bool write_back) {
        CacheWriteAt(self->state(), self->GetSRef(loop_rv),
                     self->GetSRef(block_rv), write_buffer_index,
                     storage_scope, write_back);
      });
}

}  // namespace tl
}  // namespace tvm
