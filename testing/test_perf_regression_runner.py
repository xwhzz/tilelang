from __future__ import annotations

import importlib.util
import runpy
import sys
import types
from pathlib import Path


def _load_perf_module(monkeypatch):
	module_path = Path(__file__).resolve().parents[1] / "tilelang/testing/perf_regression.py"
	spec = importlib.util.spec_from_file_location("tilelang.testing.perf_regression", module_path)
	assert spec is not None and spec.loader is not None
	module = importlib.util.module_from_spec(spec)
	sys.modules["tilelang.testing.perf_regression"] = module
	spec.loader.exec_module(module)

	tilelang_pkg = types.ModuleType("tilelang")
	tilelang_pkg.__path__ = []  # type: ignore[attr-defined]
	testing_pkg = types.ModuleType("tilelang.testing")
	testing_pkg.__path__ = []  # type: ignore[attr-defined]
	testing_pkg.process_func = module.process_func  # type: ignore[attr-defined]
	testing_pkg.regression = module.regression  # type: ignore[attr-defined]
	testing_pkg.perf_regression = module  # type: ignore[attr-defined]
	tilelang_pkg.testing = testing_pkg  # type: ignore[attr-defined]

	monkeypatch.setitem(sys.modules, "tilelang", tilelang_pkg)
	monkeypatch.setitem(sys.modules, "tilelang.testing", testing_pkg)
	monkeypatch.setitem(sys.modules, "tilelang.testing.perf_regression", module)

	return module


def test_run_bench_file_executes_regressions(monkeypatch, tmp_path):
	perf = _load_perf_module(monkeypatch)
	bench_file = tmp_path / "regression_sample.py"
	bench_file.write_text(
		"import tilelang.testing\n"
		"\n"
		"def regression_sample():\n"
		"    tilelang.testing.process_func(lambda: 1.0, 'sample')\n",
		encoding="utf-8",
	)

	perf._reset_results()
	perf._run_bench_file(bench_file)

	assert perf._results_to_jsonable() == [{"name": "sample", "latency": 1.0}]


def test_regression_all_uses_pytest_wrapper(monkeypatch, tmp_path):
	perf = _load_perf_module(monkeypatch)
	bench_file = tmp_path / "regression_sample.py"
	bench_file.write_text(
		"import tilelang.testing\n"
		"\n"
		"def regression_sample():\n"
		"    tilelang.testing.process_func(lambda: 2.5, 'sample')\n",
		encoding="utf-8",
	)

	calls: dict[str, list[str]] = {}

	def fake_pytest_main(args, plugins=None):
		calls["args"] = args
		module_vars = runpy.run_path(args[0])
		for name, fn in module_vars.items():
			if name.startswith("test_perf_regression_") and callable(fn):
				fn()
		return 0

	monkeypatch.setitem(sys.modules, "pytest", types.SimpleNamespace(main=fake_pytest_main))

	perf._reset_results()
	perf.regression_all(examples_root=tmp_path)

	assert Path(calls["args"][0]).name.startswith("test_perf_regression_wrapper")
	assert perf._results_to_jsonable() == [{"name": "sample", "latency": 2.5}]
