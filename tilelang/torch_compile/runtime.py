"""Runtime wrapper for generated TileLang torch.compile graph modules."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any, Callable

import torch

logger = logging.getLogger(__name__)

_DEBUG_WRAPPER = os.environ.get("TILELANG_DEBUG_WRAPPER", "0") not in (
    "0",
    "false",
    "no",
)


def _exec_wrapper(
    code_str: str,
    rt_mod: Any,
    cache_key: str,
    extern_ops: dict[str, Any] | None = None,
    constants: dict[str, Any] | None = None,
) -> Callable:
    source_name = f"<tilelang_graph_{cache_key[:12]}>"
    code_obj = compile(code_str, source_name, "exec")
    namespace: dict[str, Any] = {}
    exec(code_obj, namespace)  # noqa: S102

    setup_fn = namespace["_setup"]
    from tilelang import tvm as _tvm

    return setup_fn(rt_mod, _tvm.runtime.from_dlpack, extern_ops or {}, constants or {})


@dataclass
class CompiledGraphModule:
    """A compiled TileLang subgraph ready to execute."""

    wrapper_code: str
    callable: Callable
    rt_mod: Any
    cache_key: str
    disk_cacheable: bool = True

    @staticmethod
    def from_codegen(codegen: "WrapperCodeGen", rt_mod: Any, cache_key: str) -> "CompiledGraphModule":
        code_str = codegen.generate()
        if _DEBUG_WRAPPER:
            logger.info("TileLang generated wrapper [%s]:\n%s", cache_key[:8], code_str)

        extern_ops = getattr(codegen, "extern_ops", None)
        constants = getattr(codegen, "constants", None)
        call_fn = _exec_wrapper(
            code_str,
            rt_mod,
            cache_key,
            extern_ops=extern_ops,
            constants=constants,
        )

        disk_cacheable = not extern_ops and not constants
        return CompiledGraphModule(
            wrapper_code=code_str,
            callable=call_fn,
            rt_mod=rt_mod,
            cache_key=cache_key,
            disk_cacheable=disk_cacheable,
        )

    @staticmethod
    def from_disk(cache_dir: str, cache_key: str) -> CompiledGraphModule | None:
        wrapper_path = os.path.join(cache_dir, f"{cache_key}.py")
        so_path = os.path.join(cache_dir, f"{cache_key}.so")
        if not os.path.isfile(wrapper_path) or not os.path.isfile(so_path):
            return None

        try:
            with open(wrapper_path, "r", encoding="utf-8") as file:
                code_str = file.read()

            from tilelang import tvm as _tvm

            rt_mod = _tvm.runtime.load_module(so_path)
            call_fn = _exec_wrapper(code_str, rt_mod, cache_key)
            logger.info("Loaded graph module from disk cache: %s", cache_key[:8])
            return CompiledGraphModule(
                wrapper_code=code_str,
                callable=call_fn,
                rt_mod=rt_mod,
                cache_key=cache_key,
                disk_cacheable=True,
            )
        except Exception:
            logger.debug("Disk cache load failed for %s", cache_key[:8], exc_info=True)
            return None

    def save_to_disk(self, cache_dir: str) -> None:
        if not self.disk_cacheable:
            logger.debug("Skipping disk cache for %s: wrapper depends on in-memory state", self.cache_key[:8])
            return

        os.makedirs(cache_dir, exist_ok=True)
        wrapper_path = os.path.join(cache_dir, f"{self.cache_key}.py")
        try:
            with open(wrapper_path, "w", encoding="utf-8") as file:
                file.write(self.wrapper_code)
            logger.info("Saved graph module to disk cache: %s", self.cache_key[:8])
        except Exception:
            logger.debug("Disk cache save failed for %s", self.cache_key[:8], exc_info=True)

    def __call__(self, *args: torch.Tensor) -> tuple[torch.Tensor, ...]:
        return self.callable(*args)

    def show_wrapper(self) -> str:
        return self.wrapper_code
