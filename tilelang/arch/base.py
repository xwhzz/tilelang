from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Self


class ArchitectureConfig(ABC):
    """Shared interface for GPU architecture capability descriptions."""

    def __init__(self, name: str) -> None:
        self.name = name

    @abstractmethod
    def set_to_spec(self) -> Self:
        """Configure this architecture to the vendor specification."""

    @abstractmethod
    def set_to_microbench(self) -> Self:
        """Configure this architecture using microbenchmark measurements."""

    @abstractmethod
    def set_to_ncu(self) -> Self:
        """Configure this architecture using Nsight Compute measurements."""


__all__ = ["ArchitectureConfig"]
