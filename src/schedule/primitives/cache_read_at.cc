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
#include "../../transform/layout_reducer.h"
#include "utils.h"

#include <string>
#include <vector>

namespace tvm {
namespace tl {

using namespace tir;
using support::NDIntSet;

// Detect whether the loop body already contains tl.tileop.region calls that
// reference `buf`. In that case, squeezing dimensions would require rewriting
// region extents as well; keep full rank to preserve consistency.
static bool HasTileRegionCallOnBuffer(const Stmt &body, const Buffer &buf) {
  bool found = false;
  PostOrderVisit(body, [&found, &buf](const ObjectRef &obj) {
    if (found)
      return;
    const auto *call = obj.as<CallNode>();
    if (call == nullptr)
      return;
    const auto *op = call->op.as<OpNode>();
    if (op == nullptr || op->name != "tl.tileop.region" || call->args.empty()) {
      return;
    }
    const auto *load = call->args[0].as<BufferLoadNode>();
    if (load != nullptr && load->buffer.same_as(buf)) {
      found = true;
    }
  });
  return found;
}

// Build an in-place square transform:
//   for ...:
//     dst[idx] = dst[idx] * dst[idx]
static Stmt BuildSquareInplaceStmt(const Buffer &dst,
                                   const ffi::Array<PrimExpr> &shape) {
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
    body = For(iters[d],
               /*min=*/IntImm(DataType::Int(32), 0),
               /*extent=*/shape[d], ForKind::kSerial, body);
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
  CacheBufferReplacer(const Buffer &src, const Buffer &dst,
                      const ffi::Array<PrimExpr> &region_mins,
                      const std::vector<int> &kept_dims,
                      ffi::Map<Block, Block> *block_sref_reuse,
                      const BlockNode *target_block = nullptr)
      : src_(src), dst_(dst), region_mins_(region_mins), kept_dims_(kept_dims),
        block_sref_reuse_(block_sref_reuse), target_block_(target_block),
        block_only_(target_block != nullptr), in_target_scope_(false) {}

private:
  bool ShouldRewriteAccess() const { return !block_only_ || in_target_scope_; }

  bool ValueUsesDstBuffer(const PrimExpr &expr) const {
    bool found = false;
    PostOrderVisit(expr, [this, &found](const ObjectRef &obj) {
      if (found) {
        return;
      }
      if (const auto *load = obj.as<BufferLoadNode>()) {
        found = load->buffer.same_as(dst_);
      }
    });
    return found;
  }

  // Build squeezed indices: for each kept dim, compute (original_idx - min).
  ffi::Array<PrimExpr> SqueezedIndices(const ffi::Array<PrimExpr> &indices) {
    ffi::Array<PrimExpr> new_indices;
    for (int d : kept_dims_) {
      new_indices.push_back(indices[d] - region_mins_[d]);
    }
    return new_indices;
  }

  PrimExpr VisitExpr_(const BufferLoadNode *_load) final {
    BufferLoad load = Downcast<BufferLoad>(StmtExprMutator::VisitExpr_(_load));
    if (ShouldRewriteAccess() && load->buffer.same_as(src_)) {
      ObjectPtr<BufferLoadNode> n =
          ffi::make_object<BufferLoadNode>(*load.get());
      n->buffer = dst_;
      n->indices = SqueezedIndices(n->indices);
      return BufferLoad(n);
    }
    return load;
  }

  Stmt VisitStmt_(const BufferStoreNode *_store) final {
    BufferStore store =
        Downcast<BufferStore>(StmtExprMutator::VisitStmt_(_store));
    if (ShouldRewriteAccess() && store->buffer.same_as(src_)) {
      ObjectPtr<BufferStoreNode> n =
          ffi::make_object<BufferStoreNode>(*store.get());
      n->buffer = dst_;
      n->indices = SqueezedIndices(n->indices);
      if (!ValueUsesDstBuffer(n->value) && n->value.dtype() != dst_->dtype) {
        arith::Analyzer analyzer;
        n->value = analyzer.Simplify(Cast(dst_->dtype, n->value));
      }
      return BufferStore(n);
    }
    return store;
  }

  Stmt VisitStmt_(const BlockNode *_block) final {
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

  ffi::Array<BufferRegion>
  ReplaceBufferWithShift(const ffi::Array<BufferRegion> &regions) {
    ffi::Array<BufferRegion> result;
    for (const auto &region : regions) {
      if (region->buffer.same_as(src_)) {
        ffi::Array<Range> new_ranges;
        for (int d : kept_dims_) {
          new_ranges.push_back(
              Range::FromMinExtent(region->region[d]->min - region_mins_[d],
                                   region->region[d]->extent));
        }
        result.push_back(BufferRegion(dst_, new_ranges));
      } else {
        result.push_back(region);
      }
    }
    return result;
  }

  const Buffer &src_;
  const Buffer &dst_;
  const ffi::Array<PrimExpr> &region_mins_;
  const std::vector<int> &kept_dims_;
  ffi::Map<Block, Block> *block_sref_reuse_;
  const BlockNode *target_block_;
  bool block_only_;
  bool in_target_scope_;
};

static ffi::Array<Range>
ComputeRelaxedRegion(ScheduleState self, const StmtSRef &loop_sref,
                     const StmtSRef &block_sref, const Buffer &buf,
                     BufferIndexType buffer_type,
                     const runtime::StorageScope &extra_relax_scope) {
  const BlockNode *block = TVM_SREF_TO_BLOCK(block_sref);
  BlockRealize realize = GetBlockRealize(self, block_sref);
  ffi::Map<Var, PrimExpr> bindings = GetBindings(realize);

  ffi::Map<Var, arith::IntSet> var_dom =
      arith::AsIntSet(LoopDomainOfSRefTreePathSkipBlocks(
          ffi::GetRef<StmtSRef>(self->stmt2ref.at(block)->parent), loop_sref,
          extra_relax_scope));

  const auto &regions =
      (buffer_type == BufferIndexType::kRead) ? block->reads : block->writes;

  std::vector<NDIntSet> relaxed_regions;
  for (const BufferRegion &buffer_region : regions) {
    if (!buffer_region->buffer.same_as(buf)) {
      continue;
    }
    ffi::Array<arith::IntSet> relaxed =
        arith::EvalSet(Substitute(buffer_region->region, bindings), var_dom);
    relaxed_regions.push_back({relaxed.begin(), relaxed.end()});
  }
  ICHECK(!relaxed_regions.empty()) << "ValueError: buffer " << buf->name
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

static bool ContainsTargetBlock(const Stmt &stmt, const BlockNode *target) {
  bool found = false;
  PostOrderVisit(stmt, [&found, target](const ObjectRef &obj) {
    if (found) {
      return;
    }
    if (const auto *realize = obj.as<BlockRealizeNode>()) {
      if (realize->block.get() == target) {
        found = true;
      }
    }
  });
  return found;
}

class ComputeNestReplacer : public StmtMutator {
public:
  ComputeNestReplacer(const BlockNode *target, Stmt replacement)
      : target_(target), replacement_(std::move(replacement)),
        replaced_(false) {}

  Stmt VisitStmt(const Stmt &stmt) final {
    if (replaced_ || !stmt.defined()) {
      return stmt;
    }
    return StmtMutator::VisitStmt(stmt);
  }

  Stmt VisitStmt_(const SeqStmtNode *op) final {
    ffi::Array<Stmt> new_seq;
    new_seq.reserve(op->seq.size());
    for (const Stmt &stmt : op->seq) {
      if (!replaced_ && ContainsTargetBlock(stmt, target_)) {
        Stmt new_stmt = StmtMutator::VisitStmt(stmt);
        if (!replaced_ && ContainsTargetBlock(new_stmt, target_)) {
          new_seq.push_back(replacement_);
          replaced_ = true;
        } else {
          new_seq.push_back(new_stmt);
        }
      } else {
        new_seq.push_back(stmt);
      }
    }
    return SeqStmt(new_seq);
  }

  bool replaced() const { return replaced_; }

private:
  const BlockNode *target_;
  Stmt replacement_;
  bool replaced_;
};

static int GetConstInt(const PrimExpr &expr, const char *what) {
  if (const auto *imm = expr.as<IntImmNode>()) {
    return static_cast<int>(imm->value);
  }
  LOG(FATAL) << "ValueError: gemm_at expects a static " << what << ", but got "
             << expr;
  return 0;
}

static PrimExpr GetMatrixStride(const Buffer &buf) {
  if (buf->strides.size() >= 2) {
    return buf->strides[buf->strides.size() - 2];
  }
  ICHECK_GE(buf->shape.size(), 2U)
      << "ValueError: gemm_at expects at least a 2D buffer, but got "
      << buf->name;
  return buf->shape[buf->shape.size() - 1];
}

// ---------------------------------------------------------------------------
// CacheReadAt:  main entry point
// ---------------------------------------------------------------------------
static void CacheReadAt(ScheduleState self, const StmtSRef &loop_sref,
                        const StmtSRef &block_sref, int read_buffer_index,
                        const ffi::String &storage_scope,
                        const ffi::String &transform,
                        const ffi::String &cache_dtype,
                        bool disable_tma = false) {
  // ---- Step 1: Obtain source buffer and loop --------------------------------
  const BlockNode *block = TVM_SREF_TO_BLOCK(block_sref);
  Block block_ref = ffi::GetRef<Block>(block);
  Buffer src = GetNthAccessBuffer(self, block_ref, read_buffer_index,
                                  BufferIndexType::kRead);

  const ForNode *loop = TVM_SREF_TO_FOR(loop_sref);

  // ---- Step 2: Gather inner-loop domains and block bindings -----------------
  BlockRealize realize = GetBlockRealize(self, block_sref);
  ffi::Map<Var, PrimExpr> bindings = GetBindings(realize);

  runtime::StorageScope scope = runtime::StorageScope::Create(storage_scope);
  ffi::Map<Var, arith::IntSet> var_dom =
      arith::AsIntSet(LoopDomainOfSRefTreePathSkipBlocks(
          /*low_inclusive=*/ffi::GetRef<StmtSRef>(
              self->stmt2ref.at(block)->parent),
          /*high_exclusive=*/loop_sref,
          /*extra_relax_scope=*/scope));

  // ---- Step 3: Relax the buffer read region over the inner loops ------------
  std::vector<NDIntSet> relaxed_regions;
  for (const BufferRegion &buffer_region : block->reads) {
    if (buffer_region->buffer.same_as(src)) {
      ffi::Array<arith::IntSet> relaxed =
          arith::EvalSet(Substitute(buffer_region->region, bindings), var_dom);
      relaxed_regions.push_back({relaxed.begin(), relaxed.end()});
    }
  }
  ICHECK(!relaxed_regions.empty()) << "ValueError: buffer " << src->name
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
      if (const auto *imm = cache_shape[d].as<IntImmNode>()) {
        if (imm->value == 1)
          continue;
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
    auto *w = dst.CopyOnWrite();
    w->shape = squeezed_shape;
    if (!cache_dtype.empty()) {
      w->dtype = DataType(ffi::StringToDLDataType(cache_dtype));
    }
    // Produce a readable name: e.g. a_shared_dyn  or  a_local_fragment
    std::string scope_suffix = storage_scope;
    // Replace '.' with '_' for valid identifier
    for (auto &c : scope_suffix) {
      if (c == '.')
        c = '_';
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

  PrimExpr src_region_arg =
      MakeRegionCall(src, cache_region, /*access_mask=*/1);
  PrimExpr dst_region_arg = MakeRegionCall(dst, dst_ranges, /*access_mask=*/2);

  ffi::Map<ffi::String, ObjectRef> copy_annotations;
  if (disable_tma) {
    copy_annotations.Set("disable_tma", Bool(true));
  }
  Stmt copy_stmt = Evaluate(Call(DataType::Handle(), Op::Get("tl.tileop.copy"),
                                 {src_region_arg, dst_region_arg},
                                 copy_annotations));

  std::string transform_mode = transform;
  ICHECK(transform_mode.empty() || transform_mode == "square")
      << "ValueError: unsupported cache_read_at transform `" << transform_mode
      << "`, expected one of {\"\", \"square\"}";
  bool do_square_transform = transform_mode == "square";
  Stmt square_stmt;
  if (do_square_transform) {
    square_stmt = BuildSquareInplaceStmt(dst, squeezed_shape);
  }

  // ---- Step 6: Rewrite buffer references in the target consumer block -------
  ffi::Array<Stmt> subtrees = AsArray(loop->body);
  ffi::Map<Block, Block> block_sref_reuse;
  CacheBufferReplacer replacer(src, dst, region_mins, kept_dims,
                               &block_sref_reuse,
                               /*target_block=*/block);
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
      /*predicate=*/const_true(), alloc_block);

  // Build the new For loop with the wrapper block as the body.
  ObjectPtr<ForNode> new_loop_node = ffi::make_object<ForNode>(*loop);
  new_loop_node->body = std::move(alloc_realize);
  For new_loop(new_loop_node);

  // ---- Step 8: Replace in the scope root block ------------------------------
  StmtSRef scope_root_sref =
      GetScopeRoot(self, loop_sref, /*require_stage_pipeline=*/false);
  const BlockNode *scope_block = TVM_SREF_TO_BLOCK(scope_root_sref);

  Block new_scope_block = Downcast<Block>(
      LoopReplacer(loop, new_loop)(ffi::GetRef<Block>(scope_block)));

  block_sref_reuse.Set(ffi::GetRef<Block>(scope_block), new_scope_block);
  self->Replace(scope_root_sref, new_scope_block, block_sref_reuse);
}

// ---------------------------------------------------------------------------
// CacheWriteAt:  mirror of CacheReadAt for write buffers
// ---------------------------------------------------------------------------
static void CacheWriteAt(ScheduleState self, const StmtSRef &loop_sref,
                         const StmtSRef &block_sref, int write_buffer_index,
                         const ffi::String &storage_scope, bool write_back,
                         const ffi::String &reduce_type,
                         const ffi::String &reducer_replication,
                         const ffi::String &cache_dtype) {
  // ---- Step 1: Obtain destination buffer and loop ---------------------------
  const BlockNode *block = TVM_SREF_TO_BLOCK(block_sref);
  Block block_ref = ffi::GetRef<Block>(block);
  Buffer src = GetNthAccessBuffer(self, block_ref, write_buffer_index,
                                  BufferIndexType::kWrite);

  const ForNode *loop = TVM_SREF_TO_FOR(loop_sref);

  // ---- Step 2: Gather inner-loop domains and block bindings -----------------
  BlockRealize realize = GetBlockRealize(self, block_sref);
  ffi::Map<Var, PrimExpr> bindings = GetBindings(realize);

  runtime::StorageScope scope = runtime::StorageScope::Create(storage_scope);
  ffi::Map<Var, arith::IntSet> var_dom =
      arith::AsIntSet(LoopDomainOfSRefTreePathSkipBlocks(
          /*low_inclusive=*/ffi::GetRef<StmtSRef>(
              self->stmt2ref.at(block)->parent),
          /*high_exclusive=*/loop_sref,
          /*extra_relax_scope=*/scope));

  // ---- Step 3: Relax the buffer write region over the inner loops -----------
  std::vector<NDIntSet> relaxed_regions;
  for (const BufferRegion &buffer_region : block->writes) {
    if (buffer_region->buffer.same_as(src)) {
      ffi::Array<arith::IntSet> relaxed =
          arith::EvalSet(Substitute(buffer_region->region, bindings), var_dom);
      relaxed_regions.push_back({relaxed.begin(), relaxed.end()});
    }
  }
  ICHECK(!relaxed_regions.empty()) << "ValueError: buffer " << src->name
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
      if (const auto *imm = cache_shape[d].as<IntImmNode>()) {
        if (imm->value == 1)
          continue;
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
    auto *w = dst.CopyOnWrite();
    w->shape = squeezed_shape;
    if (!cache_dtype.empty()) {
      w->dtype = DataType(ffi::StringToDLDataType(cache_dtype));
    }
    std::string scope_suffix = storage_scope;
    for (auto &c : scope_suffix) {
      if (c == '.')
        c = '_';
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
    PrimExpr cache_region_arg =
        MakeRegionCall(dst, dst_ranges, /*access_mask=*/1);
    PrimExpr orig_region_arg =
        MakeRegionCall(src, cache_region, /*access_mask=*/2);
    copy_stmt = Evaluate(Call(DataType::Handle(), Op::Get("tl.tileop.copy"),
                              {cache_region_arg, orig_region_arg}));
  }

  bool use_reducer = !reduce_type.empty();
  Stmt fill_stmt;
  Stmt finalize_stmt;
  if (use_reducer) {
    ICHECK(reduce_type == "sum")
        << "ValueError: reducer-backed cache_write_at currently only supports "
           "`sum` reductions";
    ICHECK(reducer_replication == "all" || reducer_replication == "none")
        << "ValueError: unsupported reducer replication `"
        << reducer_replication << "`, expected one of {\"all\", \"none\"}";
    PrimExpr reducer_region_arg =
        MakeRegionCall(dst, dst_ranges, /*access_mask=*/2);
    fill_stmt =
        Evaluate(Call(DataType::Handle(), Op::Get("tl.tileop.fill"),
                      {reducer_region_arg, make_const(dst->dtype, 0.0)}));
    finalize_stmt =
        Evaluate(Call(DataType::Handle(), Op::Get("tl.tileop.finalize_reducer"),
                      {reducer_region_arg}));
  }

  // ---- Step 6: Rewrite buffer references in the loop body -------------------
  ffi::Array<Stmt> subtrees = AsArray(loop->body);
  ffi::Map<Block, Block> block_sref_reuse;
  CacheBufferReplacer replacer(src, dst, region_mins, kept_dims,
                               &block_sref_reuse);
  for (int i = 0; i < static_cast<int>(subtrees.size()); ++i) {
    Stmt old_stmt = subtrees[i];
    subtrees.Set(i, Stmt(nullptr));
    subtrees.Set(i, replacer(std::move(old_stmt)));
  }

  // ---- Step 7: Append the optional write-back copy --------------------------
  if (use_reducer) {
    subtrees.insert(subtrees.begin(), fill_stmt);
  }
  if (use_reducer) {
    subtrees.push_back(finalize_stmt);
  }
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
  if (use_reducer) {
    auto *block_ptr = alloc_block.CopyOnWrite();
    ffi::Map<Var, ffi::Map<ffi::String, ffi::String>> reducer_info;
    ffi::Map<ffi::String, ffi::String> reducer_meta;
    reducer_meta.Set("op", reduce_type);
    reducer_meta.Set("rep", reducer_replication);
    reducer_info.Set(dst->data, reducer_meta);
    block_ptr->annotations.Set(tvm::tl::attr::kReducerInfo, reducer_info);
  }
  BlockRealize alloc_realize(
      /*values=*/{},
      /*predicate=*/const_true(), alloc_block);

  ObjectPtr<ForNode> new_loop_node = ffi::make_object<ForNode>(*loop);
  new_loop_node->body = std::move(alloc_realize);
  For new_loop(new_loop_node);

  // ---- Step 8: Replace in the scope root block ------------------------------
  StmtSRef scope_root_sref =
      GetScopeRoot(self, loop_sref, /*require_stage_pipeline=*/false);
  const BlockNode *scope_block = TVM_SREF_TO_BLOCK(scope_root_sref);

  Block new_scope_block = Downcast<Block>(
      LoopReplacer(loop, new_loop)(ffi::GetRef<Block>(scope_block)));

  if (!write_back) {
    // Bridge/intermediate use-case: all consumers are rewritten to the cache
    // buffer, so the original tensor allocation can be removed from the
    // surrounding scope block.
    ObjectPtr<BlockNode> scope_ptr =
        ffi::make_object<BlockNode>(*new_scope_block.get());
    ffi::Array<Buffer> kept_alloc_buffers;
    kept_alloc_buffers.reserve(scope_ptr->alloc_buffers.size());
    for (const Buffer &buf : scope_ptr->alloc_buffers) {
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

static void GemmAt(ScheduleState self, const StmtSRef &loop_sref,
                   const StmtSRef &block_sref, bool transpose_a,
                   bool transpose_b, bool clear_accum, int policy_type,
                   bool use_py) {
  const BlockNode *block = TVM_SREF_TO_BLOCK(block_sref);
  Block block_ref = ffi::GetRef<Block>(block);
  const ForNode *loop = TVM_SREF_TO_FOR(loop_sref);

  ICHECK_EQ(block->reads.size(), 2U)
      << "ValueError: gemm_at expects a matmul-like block with exactly two "
         "reads";
  ICHECK_EQ(block->writes.size(), 1U)
      << "ValueError: gemm_at expects a matmul-like block with exactly one "
         "write";

  Buffer a = GetNthAccessBuffer(self, block_ref, 0, BufferIndexType::kRead);
  Buffer b = GetNthAccessBuffer(self, block_ref, 1, BufferIndexType::kRead);
  Buffer c = GetNthAccessBuffer(self, block_ref, 0, BufferIndexType::kWrite);

  ffi::Array<Range> a_region = ComputeRelaxedRegion(
      self, loop_sref, block_sref, a, BufferIndexType::kRead,
      runtime::StorageScope::Create(a.scope()));
  ffi::Array<Range> b_region = ComputeRelaxedRegion(
      self, loop_sref, block_sref, b, BufferIndexType::kRead,
      runtime::StorageScope::Create(b.scope()));
  ffi::Array<Range> c_region = ComputeRelaxedRegion(
      self, loop_sref, block_sref, c, BufferIndexType::kWrite,
      runtime::StorageScope::Create(c.scope()));

  ICHECK_EQ(a_region.size(), 2U)
      << "ValueError: gemm_at currently expects a 2D lhs tile, but got rank "
      << a_region.size();
  ICHECK_EQ(b_region.size(), 2U)
      << "ValueError: gemm_at currently expects a 2D rhs tile, but got rank "
      << b_region.size();
  ICHECK_EQ(c_region.size(), 2U) << "ValueError: gemm_at currently expects a "
                                    "2D accumulator tile, but got rank "
                                 << c_region.size();

  int m = GetConstInt(c_region[0]->extent, "M extent");
  int n = GetConstInt(c_region[1]->extent, "N extent");
  int k = GetConstInt(transpose_a ? a_region[0]->extent : a_region[1]->extent,
                      "K extent");

  PrimExpr a_region_arg = MakeRegionCall(a, a_region, /*access_mask=*/1);
  PrimExpr b_region_arg = MakeRegionCall(b, b_region, /*access_mask=*/1);
  PrimExpr c_region_arg = MakeRegionCall(c, c_region, /*access_mask=*/3);

  PrimExpr stride_a = GetMatrixStride(a);
  PrimExpr stride_b = GetMatrixStride(b);
  PrimExpr offset_a = a_region[a_region.size() - 1]->min;
  PrimExpr offset_b = b_region[b_region.size() - 1]->min;
  PrimExpr c_row = c_region[0]->min;
  PrimExpr c_col = c_region[1]->min;

  int stride_a_value = GetConstInt(stride_a, "lhs stride");
  int stride_b_value = GetConstInt(stride_b, "rhs stride");
  int offset_a_value = GetConstInt(offset_a, "lhs offset");
  int offset_b_value = GetConstInt(offset_b, "rhs offset");
  int c_row_value = GetConstInt(c_row, "accumulator row offset");
  int c_col_value = GetConstInt(c_col, "accumulator column offset");

  const char *op_name = use_py ? "tl.tileop.gemm_py" : "tl.tileop.gemm";
  Stmt gemm_stmt = Evaluate(Call(DataType::Handle(), Op::Get(op_name),
                                 {
                                     a_region_arg,
                                     b_region_arg,
                                     c_region_arg,
                                     Bool(transpose_a),
                                     Bool(transpose_b),
                                     IntImm(DataType::Int(32), m),
                                     IntImm(DataType::Int(32), n),
                                     IntImm(DataType::Int(32), k),
                                     IntImm(DataType::Int(32), policy_type),
                                     Bool(clear_accum),
                                     IntImm(DataType::Int(32), stride_a_value),
                                     IntImm(DataType::Int(32), stride_b_value),
                                     IntImm(DataType::Int(32), offset_a_value),
                                     IntImm(DataType::Int(32), offset_b_value),
                                     IntImm(DataType::Int(32), 1),
                                     IntImm(DataType::Int(32), 0),
                                     make_zero(DataType::UInt(32)),
                                     IntImm(DataType::Int(32), c_row_value),
                                     IntImm(DataType::Int(32), c_col_value),
                                 }));

  ComputeNestReplacer replacer(block, gemm_stmt);
  ObjectPtr<ForNode> new_loop_node = ffi::make_object<ForNode>(*loop);
  new_loop_node->body = replacer(loop->body);
  ICHECK(replacer.replaced())
      << "ValueError: gemm_at failed to isolate the tiled compute nest";
  For new_loop(new_loop_node);

  StmtSRef scope_root_sref =
      GetScopeRoot(self, loop_sref, /*require_stage_pipeline=*/false);
  const BlockNode *scope_block = TVM_SREF_TO_BLOCK(scope_root_sref);

  ffi::Map<Block, Block> block_sref_reuse;
  Block new_scope_block = Downcast<Block>(
      LoopReplacer(loop, new_loop)(ffi::GetRef<Block>(scope_block)));
  block_sref_reuse.Set(ffi::GetRef<Block>(scope_block), new_scope_block);
  self->Replace(scope_root_sref, new_scope_block, block_sref_reuse);
}

static void CopyAt(ScheduleState self, const StmtSRef &loop_sref,
                   const StmtSRef &block_sref, int read_buffer_index,
                   int write_buffer_index) {
  const BlockNode *block = TVM_SREF_TO_BLOCK(block_sref);
  Block block_ref = ffi::GetRef<Block>(block);
  const ForNode *loop = TVM_SREF_TO_FOR(loop_sref);

  Buffer src = GetNthAccessBuffer(self, block_ref, read_buffer_index,
                                  BufferIndexType::kRead);
  Buffer dst = GetNthAccessBuffer(self, block_ref, write_buffer_index,
                                  BufferIndexType::kWrite);

  ffi::Array<Range> src_region = ComputeRelaxedRegion(
      self, loop_sref, block_sref, src, BufferIndexType::kRead,
      runtime::StorageScope::Create(src.scope()));
  ffi::Array<Range> dst_region = ComputeRelaxedRegion(
      self, loop_sref, block_sref, dst, BufferIndexType::kWrite,
      runtime::StorageScope::Create(dst.scope()));

  Stmt copy_stmt =
      Evaluate(Call(DataType::Handle(), Op::Get("tl.tileop.copy"),
                    {
                        MakeRegionCall(src, src_region, /*access_mask=*/1),
                        MakeRegionCall(dst, dst_region, /*access_mask=*/2),
                    }));

  ComputeNestReplacer replacer(block, copy_stmt);
  ObjectPtr<ForNode> new_loop_node = ffi::make_object<ForNode>(*loop);
  new_loop_node->body = replacer(loop->body);
  ICHECK(replacer.replaced())
      << "ValueError: copy_at failed to isolate the tiled compute nest";
  For new_loop(new_loop_node);

  StmtSRef scope_root_sref =
      GetScopeRoot(self, loop_sref, /*require_stage_pipeline=*/false);
  const BlockNode *scope_block = TVM_SREF_TO_BLOCK(scope_root_sref);

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
      "tl.schedule.ScheduleCacheReadAt",
      [](Schedule self, const LoopRV &loop_rv, const BlockRV &block_rv,
         int read_buffer_index, const ffi::String &storage_scope,
         const ffi::String &transform, const ffi::String &cache_dtype,
         bool disable_tma) {
        CacheReadAt(self->state(), self->GetSRef(loop_rv),
                    self->GetSRef(block_rv), read_buffer_index, storage_scope,
                    transform, cache_dtype, disable_tma);
      });

  refl::GlobalDef().def(
      "tl.schedule.ScheduleCacheWriteAt",
      [](Schedule self, const LoopRV &loop_rv, const BlockRV &block_rv,
         int write_buffer_index, const ffi::String &storage_scope,
         bool write_back, const ffi::String &reduce_type,
         const ffi::String &reducer_replication,
         const ffi::String &cache_dtype) {
        CacheWriteAt(self->state(), self->GetSRef(loop_rv),
                     self->GetSRef(block_rv), write_buffer_index, storage_scope,
                     write_back, reduce_type, reducer_replication,
                     cache_dtype);
      });

  refl::GlobalDef().def(
      "tl.schedule.ScheduleGemmAt",
      [](Schedule self, const LoopRV &loop_rv, const BlockRV &block_rv,
         bool transpose_a, bool transpose_b, bool clear_accum, int policy_type,
         bool use_py) {
        GemmAt(self->state(), self->GetSRef(loop_rv), self->GetSRef(block_rv),
               transpose_a, transpose_b, clear_accum, policy_type, use_py);
      });

  refl::GlobalDef().def(
      "tl.schedule.ScheduleCopyAt",
      [](Schedule self, const LoopRV &loop_rv, const BlockRV &block_rv,
         int read_buffer_index, int write_buffer_index) {
        CopyAt(self->state(), self->GetSRef(loop_rv), self->GetSRef(block_rv),
               read_buffer_index, write_buffer_index);
      });
}

} // namespace tl
} // namespace tvm
