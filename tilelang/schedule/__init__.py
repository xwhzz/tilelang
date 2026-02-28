from . import _ffi_api
from tvm.tir import Schedule as TVMSchedule
from tvm.tir.schedule import LoopRV, BlockRV

class Schedule(TVMSchedule):
    
    def launch_thread(self, block: BlockRV, num_threads: int) -> None:
        _ffi_api.ScheduleLaunchThread(self, block, num_threads)

    def cache_read_at(self, loop: LoopRV, block: BlockRV, read_buffer_index: int,
                      storage_scope: str) -> None:
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
        """
        _ffi_api.ScheduleCacheReadAt(self, loop, block, read_buffer_index,
                                     storage_scope)

    def cache_write_at(self, loop: LoopRV, block: BlockRV, write_buffer_index: int,
                       storage_scope: str) -> None:
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
        """
        _ffi_api.ScheduleCacheWriteAt(self, loop, block, write_buffer_index,
                                      storage_scope)