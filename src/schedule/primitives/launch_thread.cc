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
#include <tvm/arith/analyzer.h>
#include <tvm/arith/int_set.h>
#include <tvm/arith/iter_affine_map.h>
#include <tvm/node/serialization.h>
#include <tvm/tir/analysis.h>
#include <tvm/tir/function.h>
#include <tvm/tir/op.h>
#include <tvm/tir/schedule/instruction.h>
#include <tvm/tir/schedule/schedule.h>
#include <tvm/tir/schedule/state.h>
#include <tvm/tir/schedule/trace.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/utils.h>

#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>

namespace tvm {
namespace tl {
using namespace tir;

class ThreadLauncher : public StmtExprMutator {
public:
  static PrimFunc Substitute(PrimFunc &f, int num_threads) {
    auto rewriter = ThreadLauncher(num_threads);
    f.CopyOnWrite()->body = rewriter(f->body);
    return f;
  }

private:
  ThreadLauncher(int num_threads) : num_threads_(num_threads) {}

  Stmt VisitStmt_(const BlockNode *op) final {
    auto new_var = Var("tx");
    IterVar thread_iter(Range(nullptr), Var("threadIdx.x"), kThreadIndex, "threadIdx.x");
    Stmt body = For(new_var, 0, num_threads_,
                     ForKind::kThreadBinding, op->body,
                     thread_iter, {}, std::nullopt);
    return Block(ffi::Array<IterVar>(), {}, {}, {}, std::move(body));
  }

  int num_threads_;
};

void LaunchThread(ScheduleState self, int num_threads) {
  auto mod = self->mod;
  for (const auto &[gvar, base_func] : mod->functions) {
      if (auto opt = base_func.as<tir::PrimFunc>()) {
        tir::PrimFunc func = opt.value();
        func = ThreadLauncher::Substitute(func, num_threads);
        mod.CopyOnWrite()->Update(gvar, func);
        break;
      }
  }

  self->mod = mod;
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.schedule.ScheduleLaunchThread",
                        [](Schedule self, int num_threads) {
                          return LaunchThread(self->state(), num_threads);
                        });
}
}  // namespace tir
}  // namespace tvm
