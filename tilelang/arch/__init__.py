from __future__ import annotations

from typing import Literal, Dict, Type, Callable

from .base import ArchitectureConfig
from .h100_pcie import H100_PCIE


_ARCHITECTURE_REGISTRY: Dict[str, Type[ArchitectureConfig]] = {
	"H100_PCIE": H100_PCIE,
}


def register_architecture(name: str, arch_cls: Type[ArchitectureConfig]) -> None:
	"""Register a new architecture configuration class."""
	_ARCHITECTURE_REGISTRY[_normalize_name(name)] = arch_cls


def get_arch_config(name: str, profile:  Literal["spec", "microbench", "ncu"] = "microbench") -> ArchitectureConfig:
	"""Return an architecture config optionally initialized for a profile."""
	try:
		arch = _ARCHITECTURE_REGISTRY[_normalize_name(name)]()
	except KeyError as err:
		raise ValueError(f"Unknown architecture '{name}'") from err

	if profile is None:
		return arch

	_apply_profile(arch, profile)
	return arch


def _apply_profile(arch: ArchitectureConfig, profile: str) -> None:
	profile_key = profile.strip().lower()
	profile_setters: Dict[str, Callable[[], ArchitectureConfig]] = {
		"spec": arch.set_to_spec,
		"microbench": arch.set_to_microbench,
		"ncu": arch.set_to_ncu,
	}
	try:
		profile_setters[profile_key]()
	except KeyError as err:
		raise ValueError(
			f"Unsupported profile '{profile}' for architecture '{arch.name}'"
		) from err


def _normalize_name(value: str) -> str:
	return value.strip().upper()


__all__ = [
	"ArchitectureConfig",
	"H100_PCIE",
	"register_architecture",
	"get_arch_config",
]
