# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""The profiler and convert to torch utils"""

from abc import ABC, abstractmethod
from typing import Any, List, Callable, Optional
from tilelang.engine.param import KernelParam


class BaseKernelAdapter(ABC):

    func: Optional[Callable] = None

    def __init__(self, mod, params: List[KernelParam], result_idx: List[int]) -> None:
        self.mod = mod
        self.params = params
        self.result_idx = self._legalize_result_idx(result_idx)
        self._post_init()

    def _legalize_result_idx(self, result_idx: Optional[List[int]]) -> List[int]:
        params = self.params
        # result_idx is a list of indices of the output tensors
        if result_idx is None:
            result_idx = []
        elif isinstance(result_idx, int):
            if result_idx > len(params) or result_idx < -len(params):
                raise ValueError(
                    f"result_idx should be an integer between {-len(params) - 1} and {len(params) - 1}"
                )
            if result_idx < 0:
                result_idx = len(params) + result_idx
            result_idx = [result_idx]
        elif isinstance(result_idx, list):
            for i, idx in enumerate(result_idx):
                if idx >= len(params) or idx <= -len(params):
                    raise ValueError(
                        f"result_idx should be an integer between {-len(params) - 1} and {len(params) - 1}"
                    )
                if idx < 0:
                    result_idx[i] = len(params) + idx
        else:
            raise ValueError("result_idx should be a list of integers")

        return result_idx

    @abstractmethod
    def _convert_torch_func(self) -> callable:
        pass

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.func(*args, **kwds)

    def get_kernel_source(self) -> str:
        return self.mod.imported_modules[0].get_source()

    def _post_init(self):
        self.func = self._convert_torch_func()
