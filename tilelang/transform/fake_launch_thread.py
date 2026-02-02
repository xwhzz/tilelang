from tvm.tir import BufferStore, For, AttrStmt, ForKind, Var, PrimFunc, BufferLoad, Buffer, IntImm, IterVar
from tvm.tir.stmt_functor import ir_transform, post_order_visit
from tvm.tir.transform import prim_func_pass


def FakeLaunchThread():
    def pass_fn(func: PrimFunc, mod, ctx):
        launch_thread_num = 256

        def pre_visit(statement):
            """
            Pre-order visitor that tracks thread bindings and tile operation depth.
            """
            pass
            # if isinstance(statement, AttrStmt) and statement.attr_key == "thread_extent":
            #     iv = IterVar((0, launch_thread_num), Var(f"tx", "int32"), 1, "threadIdx.x")
            #     return AttrStmt(
            #         iv,
            #         statement.attr_key,
            #         launch_thread_num,
            #         statement
            #     )

        def post_visit(statement):
            """
            Post-order visitor that applies transformations and updates counters.
            """
            if isinstance(statement, AttrStmt) and statement.attr_key == "thread_extent":
                iv = IterVar((0, launch_thread_num), Var(f"tx", "int32"), 1, "threadIdx.x")
                new_body = AttrStmt(
                    iv,
                    statement.attr_key,
                    launch_thread_num,
                    statement.body
                )
                return AttrStmt(
                    statement.node,
                    statement.attr_key,
                    statement.value,
                    new_body
                )

        new_body = ir_transform(func.body, pre_visit, post_visit)

        return func.with_body(new_body)

    return prim_func_pass(pass_fn, opt_level=0)
