from . import _ffi_api
from tvm.tir import Schedule as TVMSchedule
from tvm.tir.schedule import LoopRV, BlockRV

class Schedule(TVMSchedule):
    
    def launch_thread(self, block: BlockRV, num_threads: int) -> None:
        _ffi_api.ScheduleLaunchThread(self, block, num_threads)

    def parallelize(self, loop: LoopRV) -> None:
        _ffi_api.ScheduleParallelizeLoop(self, loop)

    def pipeline(self, loop: LoopRV, num_stages: int) -> None:
        _ffi_api.SchedulePipelineLoop(self, loop, num_stages)

    def gemm_at(
        self,
        loop: LoopRV,
        block: BlockRV,
        transpose_a: bool = False,
        transpose_b: bool = False,
        clear_accum: bool = False,
        policy: str = "square",
        use_py: bool = False,
    ) -> None:
        """Replace a tiled matmul loop nest with a TileLang `T.gemm` call."""
        policy_id = {
            "square": 0,
            "full_row": 1,
            "full_col": 2,
        }.get(policy)
        if policy_id is None:
            raise ValueError(
                f"Unsupported gemm policy `{policy}`, expected one of "
                '{"square", "full_row", "full_col"}'
            )
        _ffi_api.ScheduleGemmAt(
            self,
            loop,
            block,
            transpose_a,
            transpose_b,
            clear_accum,
            policy_id,
            use_py,
        )

    def copy_at(
        self,
        loop: LoopRV,
        block: BlockRV,
        read_buffer_index: int = 0,
        write_buffer_index: int = 0,
    ) -> None:
        """Replace a tiled copy block with a TileLang `T.copy` call."""
        _ffi_api.ScheduleCopyAt(
            self,
            loop,
            block,
            read_buffer_index,
            write_buffer_index,
        )

    def cache_read_at(self, loop: LoopRV, block: BlockRV, read_buffer_index: int,
                      storage_scope: str, transform: str = "") -> None:
        """Insert a cached copy of a read buffer at the specified loop level.

        This creates a compact cache buffer inside the loop, inserts a T.copy
        to fill it from the original buffer's accessed region, and rewrites
        all references to use the cache with shifted indices.

        Parameters
        ----------
        loop : LoopRV
            The loop where the cache allocation and copy should be placed.
        block : BlockRV
            The consumer block that reads the buffer.
        read_buffer_index : int
            Index of the read buffer in the block's read list.
        storage_scope : str
            Storage scope for the cache buffer (e.g. "shared.dyn",
            "local.fragment").
        transform : str
            Optional elementwise transform applied on the cached tile after
            copy. Supported values:
            - "" (default): no transform
            - "square": dst = dst * dst
        """
        _ffi_api.ScheduleCacheReadAt(self, loop, block, read_buffer_index,
                                     storage_scope, transform)

    def cache_write_at(self, loop: LoopRV, block: BlockRV, write_buffer_index: int,
                       storage_scope: str, write_back: bool = True,
                       reduce_type: str = "", reducer_replication: str = "none") -> None:
        """Insert a cached copy of a write buffer at the specified loop level.

        This creates a compact cache buffer inside the loop, rewrites all
        write references to use the cache, and inserts a T.copy to write
        back from the cache to the original buffer.

        Parameters
        ----------
        loop : LoopRV
            The loop where the cache allocation and write-back should be placed.
        block : BlockRV
            The producer block that writes the buffer.
        write_buffer_index : int
            Index of the write buffer in the block's write list.
        storage_scope : str
            Storage scope for the cache buffer (e.g. "shared.dyn",
            "local.fragment").
        write_back : bool
            Whether to emit a final T.copy from cache back to the original
            buffer (default: True). Set to False for purely intermediate
            tensors whose consumers are all rewritten to read the cache.
        reduce_type : str
            Optional reducer annotation for the cached write buffer. When set
            to "sum", "max", or "min", the allocated cache buffer is marked
            as a TileLang reducer and finalized before write-back.
        reducer_replication : str
            Reducer replication strategy, either "none" (default) or "all".
        """
        _ffi_api.ScheduleCacheWriteAt(self, loop, block, write_buffer_index,
                                      storage_scope, write_back,
                                      reduce_type, reducer_replication)

    def fill_at(self, loop: LoopRV, block: BlockRV, write_buffer_index: int,
                value: float = 0.0) -> None:
        """Insert a T.fill to initialize a buffer at the specified loop level.

        This analyzes the block's write buffer access region within one
        iteration of the specified loop, then inserts a tl.fill statement
        at the beginning of the loop body to initialize the region with the
        given value.

        This is essential for reduction patterns where an accumulator buffer
        must be initialized before the reduction loop.

        Parameters
        ----------
        loop : LoopRV
            The loop where the fill should be inserted.
        block : BlockRV
            The block whose write buffer to fill.
        write_buffer_index : int
            Index of the write buffer in the block's write list.
        value : float
            The value to initialize the buffer with (default: 0.0).
            Common values: 0.0 for sum, -inf for max, +inf for min.
        """
        _ffi_api.ScheduleFillAt(self, loop, block, write_buffer_index, value)

    def reduce_at(self, loop: LoopRV, block: BlockRV,
                  read_buffer_index: int, write_buffer_index: int,
                  reduce_type: str, dim: int, clear: bool = True,
                  replace_loop_body: bool = False) -> None:
        """Insert a T.reduce at the specified loop level.

        This inserts a tile-level reduction operation for the specified loop.
        The reduction reads from the block's read buffer and writes to the
        block's write buffer, performing the specified reduction along the
        given dimension.

        Parameters
        ----------
        loop : LoopRV
            The loop where the reduce should be inserted.
        block : BlockRV
            The block whose buffers are used for the reduction.
        read_buffer_index : int
            Index of the source (read) buffer in the block's read list.
        write_buffer_index : int
            Index of the destination (write) buffer in the block's write list.
        reduce_type : str
            Type of reduction: "sum", "sumsq", "max", "min",
            "abssum", "absmax".
        dim : int
            The dimension along which to reduce.
        clear : bool
            Whether to clear the destination buffer before reduction
            (default: True).
        replace_loop_body : bool
            If True, replace the loop body with only the generated T.reduce.
            If False (default), append T.reduce at the end of loop body.
        """
        _ffi_api.ScheduleReduceAt(self, loop, block, read_buffer_index,
                                  write_buffer_index, reduce_type, dim, clear,
                                  replace_loop_body)

    def cache_reduce_at(self, loop: LoopRV, block: BlockRV,
                        write_buffer_index: int, storage_scope: str,
                        init_value: float = 0.0,
                        write_back: bool = True) -> None:
        """Cache a write buffer for reduction with initialization.

        This is a combined primitive for reduction scheduling that:

        1. Creates a compact accumulator buffer in the specified storage
           scope (analogous to cache_write_at).
        2. Inserts a T.fill to initialize the accumulator with the given
           init_value at the beginning of the loop body.
        3. Rewrites all write references to use the accumulator.
        4. Inserts a T.copy to write back from the accumulator to the
           original buffer at the end of the loop body.

        This is the recommended primitive for scheduling reduction patterns
        where an accumulator must be allocated in a fast memory scope,
        initialized before the reduction loop, and written back afterward.

        Parameters
        ----------
        loop : LoopRV
            The loop where the cached reduction should be placed.
        block : BlockRV
            The block whose write buffer to cache for reduction.
        write_buffer_index : int
            Index of the write buffer in the block's write list.
        storage_scope : str
            Storage scope for the accumulator buffer (e.g.
            "local.fragment", "shared.dyn").
        init_value : float
            The initial value for the accumulator (default: 0.0).
            Use 0.0 for sum, float('-inf') for max, float('inf') for min.
        write_back : bool
            Whether to emit a final T.copy from cache back to the original
            buffer (default: True). Set to False for purely intermediate
            tensors whose consumers are all rewritten to read the cache.
        """
        _ffi_api.ScheduleCacheReduceAt(self, loop, block, write_buffer_index,
                                       storage_scope, init_value, write_back)
