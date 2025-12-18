from __future__ import annotations

import contextlib
import importlib.util
import hashlib
import inspect
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
import tempfile
from typing import Any, Callable, Iterable, Sequence

try:
	from tabulate import tabulate
except Exception:  # pragma: no cover
	tabulate = None  # type: ignore

@dataclass(frozen=True)
class PerfResult:
	name: str
	latency: float


_RESULTS: list[PerfResult] = []


_RESULTS_JSON_PREFIX = "__TILELANG_PERF_RESULTS_JSON__="


def _results_to_jsonable() -> list[dict[str, float | str]]:
	return [{"name": r.name, "latency": r.latency} for r in _RESULTS]


def _emit_results() -> None:
	"""Emit results for parent collectors.

	Default output remains the historical text format. Set
	`TL_PERF_REGRESSION_FORMAT=json` to emit a single JSON marker line which is
	robust against extra prints from benchmark code.
	"""
	fmt = os.environ.get("TL_PERF_REGRESSION_FORMAT", "text").strip().lower()
	if fmt == "json":
		print(_RESULTS_JSON_PREFIX + json.dumps(_results_to_jsonable(), separators=(",", ":")))
		return
	# Fallback (human-readable): one result per line.
	for r in _RESULTS:
		print(f"{r.name}: {r.latency}")


def _reset_results() -> None:
	_RESULTS.clear()


@contextlib.contextmanager
def _pushd(path: Path) -> Iterable[None]:
	"""Temporarily change working directory (process-wide; avoid in concurrent contexts)."""
	cwd = Path.cwd()
	os.chdir(path)
	try:
		yield
	finally:
		os.chdir(cwd)


@contextlib.contextmanager
def _prepend_sys_path(path: Path) -> Iterable[None]:
	orig = list(sys.path)
	sys.path.insert(0, str(path))
	try:
		yield
	finally:
		sys.path[:] = orig


def _iter_regression_functions(namespace: dict[str, Any], prefixes: Sequence[str]) -> Iterable[tuple[str, Callable[..., Any]]]:
	for k, v in namespace.items():
		if not callable(v):
			continue
		if any(k.startswith(p) for p in prefixes):
			yield k, v


def _run_bench_file(bench_file: Path, *, prefixes: Sequence[str] = ("regression_",)) -> None:
	bench_file = bench_file.resolve()
	if not bench_file.is_file():
		raise FileNotFoundError(f"Benchmark driver not found: {bench_file}")

	with _pushd(bench_file.parent), _prepend_sys_path(bench_file.parent):
		module_tag = hashlib.sha256(str(bench_file).encode("utf-8")).hexdigest()[:12]
		parent_stem = bench_file.parent.name.replace("-", "_") or "root"
		stem = bench_file.stem.replace("-", "_")
		module_name = f"tilelang.testing.perf_regression.bench_{parent_stem}_{stem}_{module_tag}"
		spec = importlib.util.spec_from_file_location(module_name, bench_file)
		if spec is None or spec.loader is None:
			raise ImportError(f"Cannot import benchmark driver: {bench_file}")
		module = importlib.util.module_from_spec(spec)
		prev = sys.modules.get(module_name)
		sys.modules[module_name] = module
		try:
			spec.loader.exec_module(module)

			for _, fn in sorted(_iter_regression_functions(module.__dict__, prefixes), key=lambda kv: kv[0]):
				fn()
		finally:
			if prev is None:
				sys.modules.pop(module_name, None)
			else:
				sys.modules[module_name] = prev


def _build_pytest_wrapper(bench_files: Sequence[Path]) -> str:
	lines = [
		"from pathlib import Path",
		"import tilelang.testing.perf_regression as _pr",
		"",
		"def _make_test(path_str):",
		"    path = Path(path_str)",
		"    def _inner():",
		"        _pr._run_bench_file(path)",
		"    return _inner",
		"",
	]

	for idx, bench in enumerate(bench_files):
		lines.append(f"test_perf_regression_{idx} = _make_test({str(bench)!r})")

	lines.append("")
	return "\n".join(lines)


def process_func(func: Callable[..., float], name: str | None = None, /, **kwargs: Any) -> float:
	"""Execute a single perf function and record its latency.

	`func` is expected to return a positive latency scalar (seconds or ms; we
	treat it as an opaque number, only ratios matter for regression).
	"""
	result_name = getattr(func, "__module__", "<unknown>") if name is None else name
	if result_name.startswith("regression_"):
		result_name = result_name[len("regression_") :]
	latency = float(func(**kwargs))
	if not (latency > 0.0):
		raise ValueError(f"Invalid latency from {result_name}: {latency}")
	_RESULTS.append(PerfResult(name=result_name, latency=latency))
	return latency


def regression(prefixes: Sequence[str] = ("regression_",)) -> None:
	"""Run entrypoints in the caller module and print a markdown table.

	This is invoked by many example scripts.
	"""

	caller_globals = inspect.currentframe().f_back.f_globals  # type: ignore[union-attr]

	_reset_results()
	functions: list[tuple[str, Callable[[], Any]]] = []
	for k, v in list(caller_globals.items()):
		if not callable(v):
			continue
		if any(k.startswith(p) for p in prefixes):
			functions.append((k, v))

	for _, fn in sorted(functions, key=lambda kv: kv[0]):
		fn()

	_emit_results()

def _parse_table(output: str) -> dict[str, float]:
	# Prefer a single JSON marker line if present.
	for line in reversed(output.splitlines()):
		if line.startswith(_RESULTS_JSON_PREFIX):
			payload = line[len(_RESULTS_JSON_PREFIX) :].strip()
			items = json.loads(payload)
			data: dict[str, float] = {}
			for item in items:
				name = str(item["name"]).strip()
				latency = float(item["latency"])
				data[name] = latency
			return data

	# Backward-compatible text parsing (best-effort).
	data = {}
	for line in output.splitlines():
		line = line.strip()
		if not line or ":" not in line:
			continue
		name, _, val = line.partition(":")
		name = name.strip()
		val = val.strip()
		if not name:
			continue
		try:
			data[name] = float(val)
		except ValueError:
			# Ignore unrelated prints/logs.
			continue
	return data


def _examples_root() -> Path:
	# repo_root/tilelang/testing/perf_regression.py -> repo_root
	return Path(__file__).resolve().parents[2] / "examples"


def _discover_bench_files(examples_root: Path) -> list[Path]:
	patterns = ("regression_*.py",)
	files: list[Path] = []
	for pat in patterns:
		files.extend(examples_root.rglob(pat))
	# Avoid picking up things like __pycache__ etc.
	return sorted({p for p in files if p.is_file() and p.name != "__init__.py"})


def regression_all(examples_root: str | os.PathLike[str] | None = None, *, pytest_args: Sequence[str] | None = None) -> None:
	"""Run all example benchmark drivers and print a consolidated table.

	Intended usage (CI): `python -c "import tilelang.testing.perf_regression as pr; pr.regression_all()"`
	Additional pytest arguments can be passed via `pytest_args`.
	"""

	root = Path(examples_root) if examples_root is not None else _examples_root()
	if not root.exists():
		raise FileNotFoundError(f"Examples root not found: {root}")

	bench_files = [p.resolve() for p in _discover_bench_files(root)]
	if not bench_files:
		raise RuntimeError(f"No benchmark drivers found under: {root}")

	_reset_results()
	wrapper_source = _build_pytest_wrapper(bench_files)
	merged: dict[str, float] = {}
	with tempfile.TemporaryDirectory() as td:
		wrapper = Path(td) / "test_perf_regression_wrapper.py"
		wrapper.write_text(wrapper_source, encoding="utf-8")

		try:
			import pytest  # type: ignore
		except ImportError as exc:  # pragma: no cover - tested via stubbed import
			raise RuntimeError("pytest is required to run perf regression suite. Install with: pip install pytest") from exc

		# Disable output capturing so benchmark progress remains visible.
		args = [str(wrapper), "-s"]
		if pytest_args:
			args.extend(pytest_args)

		exit_code = pytest.main(args)

		for res in _RESULTS:
			if res.name not in merged:
				merged[res.name] = res.latency

		if not merged:
			if exit_code != 0:
				raise RuntimeError("All benchmark drivers failed")
			raise RuntimeError("No benchmark results collected")
		if exit_code != 0:
			# Don't hard-fail if we have some results; pytest already reported details.
			print("# Some benchmark drivers failed (partial results)")

	rows = [[k, merged[k]] for k in sorted(merged.keys())]
	headers = ["File", "Latency"]
	if tabulate is None:
		print(f"| {headers[0]} | {headers[1]} |")
		print("|---|---|")
		for name, latency in rows:
			print(f"| {name} | {latency} |")
	else:
		print(tabulate(rows, headers=headers, tablefmt="github", stralign="left", numalign="decimal"))
