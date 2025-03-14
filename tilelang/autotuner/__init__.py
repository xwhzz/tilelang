# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""The auto-tune module for tilelang programs."""

import tilelang
from tilelang import tvm as tvm
import inspect
from functools import wraps
from typing import Any, Callable, List, Literal
from tqdm import tqdm
import logging
from dataclasses import dataclass
import concurrent.futures
import os
from functools import partial

logger = logging.getLogger(__name__)

logging.basicConfig(
    filename='out.log',
    filemode='w',
    level=logging.INFO,
    format='%(asctime)s %(levelname)s:%(message)s')


@dataclass(frozen=True)
class JITContext:
    mod: tilelang.Profiler
    out_idx: List[int]
    supply_type: tilelang.TensorSupplyType
    ref_prog: Callable
    rtol: float
    atol: float
    max_mismatched_ratio: float
    skip_check: bool
    profiler: Literal['torch', 'tvm']
    target: Literal['cuda', 'hip']


class Autotuner:

    def __init__(
        self,
        fn: Callable,
        configs: Any,
        keys: List[str],
        warmup: int = 25,
        rep: int = 100,
        timeout: int = 30,
    ):
        self.fn = fn
        self.configs = configs
        self.keys = keys
        self.warmup = warmup
        self.rep = rep
        self.timeout = timeout

        # Precompute cached variables
        self.ref_latency_cache = None
        self.jit_input_tensors = None
        self.ref_input_tensors = None

    def jit_compile(self, args: Any, **kwds: Any) -> JITContext:
        jit_context = self.fn(*args, **kwds)
        return jit_context

    def run(self, *args: Any, **kwds: Any) -> Any:
        sig = inspect.signature(self.fn)
        bound_args = sig.bind(*args, **kwds)
        bound_args.apply_defaults()

        best_latency = 1e8
        best_config = None

        def target_fn(jit_context):
            # Unpack the context
            mod = jit_context.mod
            profiler = jit_context.profiler
            skip_check = jit_context.skip_check
            ref_prog = jit_context.ref_prog
            rtol = jit_context.rtol
            atol = jit_context.atol
            max_mismatched_ratio = jit_context.max_mismatched_ratio

            self.jit_input_tensors = mod._get_inputs(
                with_output=profiler ==
                "tvm") if self.jit_input_tensors is None else self.jit_input_tensors

            if (not skip_check) and (ref_prog is not None):
                mod.assert_allclose(
                    ref_prog, rtol=rtol, atol=atol, max_mismatched_ratio=max_mismatched_ratio)

            latency = mod.do_bench(
                mod.func,
                n_warmup=self.warmup,
                n_repeat=self.rep,
                profiler=profiler,
                input_tensors=self.jit_input_tensors)
            if self.ref_latency_cache is None and ref_prog is not None:
                self.ref_input_tensors = mod._get_inputs(
                    with_output=False) if self.ref_input_tensors is None else self.ref_input_tensors
                self.ref_latency_cache = mod.do_bench(
                    ref_prog,
                    n_warmup=self.warmup,
                    n_repeat=self.rep,
                    profiler="torch",
                    input_tensors=self.ref_input_tensors)

            return latency, self.ref_latency_cache

        # Parallel compilation
        config_args = []

        for config in self.configs:
            new_args = []
            for name, value in bound_args.arguments.items():
                if name not in self.keys:
                    new_args.append(value)
                else:
                    new_args.append(config[name])
            new_args = tuple(new_args)
            config_args.append(new_args)

        worker = partial(
            self.jit_compile,
            **kwds,
        )

        # 90% utilization
        num_workers = max(1, int(os.cpu_count() * 0.9))
        pool = concurrent.futures.ThreadPoolExecutor(max_workers=num_workers)

        # Submit all compilation jobs
        futures = []
        future_to_index = {}  # Track which future corresponds to which config
        for i, config_arg in enumerate(config_args):
            future = pool.submit(worker, config_arg)
            futures.append(future)
            future_to_index[future] = i

        # Process results with error handling
        results_with_configs = []
        for future in tqdm(
                concurrent.futures.as_completed(futures),
                total=len(futures),
                desc="Compiling configurations"):
            idx = future_to_index[future]
            config = config_args[idx]
            try:
                result = future.result()
                results_with_configs.append((result, config))
            except Exception:
                logger.debug(f"Compilation failed for config {config} at index {idx}")
                continue

        ref_latency = None
        progress_bar = tqdm(range(len(results_with_configs)), desc="Bench configurations")
        for i in progress_bar:
            jit_context, config = results_with_configs[i]
            try:
                # Use ThreadPoolExecutor to enforce timeout on target_fn execution
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(target_fn, jit_context)
                    latency, ref_latency = future.result(timeout=self.timeout)
            except concurrent.futures.TimeoutError:
                logger.debug(f"Timeout exceeded for config {config}. Skipping this configuration.")
                continue
            except Exception as e:
                logger.debug(f"An error occurred while testing config {config}: {e}")
                continue

            logging.debug(f"Config {config} latency: {latency} at index {i}")

            if latency < best_latency:
                best_latency = latency
                best_config = config

            progress_bar.set_postfix({"best_latency": best_latency})
            tqdm.write(f"Tuned Latency {latency} with config {config} at index {i}")

        pool.shutdown()
        return best_latency, best_config, ref_latency

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.run(*args, **kwds)


def autotune(configs: Any,
             keys: List[str],
             warmup: int = 25,
             rep: int = 100,
             timeout: int = 100) -> Callable:
    """
    Decorator for tilelang program
    """

    def decorator(fn: Callable) -> Autotuner:
        return Autotuner(fn, configs=configs, keys=keys, warmup=warmup, rep=rep, timeout=timeout)

    return decorator


def jit(out_idx: List[int],
        supply_type: tilelang.TensorSupplyType = tilelang.TensorSupplyType.Normal,
        ref_prog: Callable = None,
        rtol: float = 1e-2,
        atol: float = 1e-2,
        max_mismatched_ratio: float = 0.01,
        skip_check: bool = False,
        profiler: Literal['auto', 'torch', 'tvm'] = 'auto',
        target: Literal['auto', 'cuda', 'hip'] = 'auto') -> Callable:

    def wrapper(fn: Callable):

        @wraps(fn)
        def decorator(*args, **kwargs) -> float:
            # Enabling Efficient Fusion
            with tvm.transform.PassContext(config={"tir.merge_static_smem": True}):
                mod, params = tilelang.lower(fn(*args, **kwargs), target=target)

            mod = tilelang.Profiler(mod, params, out_idx, supply_type)

            return JITContext(
                mod=mod,
                out_idx=out_idx,
                supply_type=supply_type,
                ref_prog=ref_prog,
                rtol=rtol,
                atol=atol,
                max_mismatched_ratio=max_mismatched_ratio,
                skip_check=skip_check,
                profiler=profiler,
                target=target)

        return decorator

    return wrapper
