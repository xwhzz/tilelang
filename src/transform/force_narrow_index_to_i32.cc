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
 * \file force_narrow_index_to_i32.cc
 * \brief Force narrow down indexing expressions and integer buffers to int32 dtype.
 * \note This pass is not used in default cases.
 */

#include <tvm/ffi/reflection/registry.h>
#include <tvm/tir/data_type_rewriter.h>
#include <tvm/tir/op.h>
#include <tvm/tir/transform.h>

namespace tvm {
namespace tl {

using namespace tir;

class Int32DTypeNarrower : public IndexDataTypeNormalizer {
 public:
  static PrimFunc RewriteDataType(PrimFunc func) {
    // Int64 data buffers (e.g. position IDs, token IDs) are valid and
    // must not be rejected.  Only index expressions are narrowed.

    Int32DTypeNarrower narrower;
    return narrower.Rewrite(func);
  }

 private:
  Int32DTypeNarrower()
      : IndexDataTypeNormalizer(DataType::Int(32)) {}

  PrimExpr VisitExpr_(const IntImmNode* op) final {
    // ignore the enabled condition and always rewrite i64
    if (op->dtype == DataType::Int(64)) {
      ICHECK_LE(op->value, Downcast<Integer>(max_value(target_data_type_))->value);
      return IntImm(DataType::Int(32), op->value);
    }
    return ffi::GetRef<IntImm>(op);
  }

  Stmt VisitStmt_(const BlockNode* block) final {
    return Downcast<Block>(IndexDataTypeNormalizer::VisitStmt_(block));
  }

};

PrimFunc ForceNarrowIndexToInt32(PrimFunc func) {
  return Int32DTypeNarrower::RewriteDataType(func);
}

using namespace tir::transform;

namespace transform {

Pass ForceNarrowIndexToInt32() {
  auto pass_func = [](PrimFunc f, IRModule m, PassContext ctx) {
    return tl::ForceNarrowIndexToInt32(f);
  };
  return CreatePrimFuncPass(pass_func, 0, "tl.NarrowDataType", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.transform.ForceNarrowIndexToInt32", ForceNarrowIndexToInt32);
}

}  // namespace transform
}  // namespace tir
}  // namespace tvm
