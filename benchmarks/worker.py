"""
Isolated benchmark worker.

Receives a benchmark configuration as JSON via stdin,
executes a single benchmark scenario,
and prints the result as JSON to stdout.

The worker always runs in a separate process to isolate:
- memory measurements,
- CUDA state,
- model allocations,
- crashes and exceptions.

Used by:
- quality benchmarks
- overhead benchmarks
- performance benchmarks
"""

from __future__ import annotations

import importlib
import json
import os
import sys
import tempfile
import time
import traceback
from typing import Any, cast

import numpy as np

from benchmarks._utils import (
    CPUMemoryMonitor,
    GPUMemoryMonitor,
    clear_memory,
    safe_auc,
    timed_call,
)
from benchmarks.data.registry import load_data

try:
    import torch
except ImportError:
    torch = None


def _build_omniad_model(
    algo_name: str,
    kwargs: dict[str, Any],
) -> Any:
    """Build an OmniAD detector."""
    from omniad import get_detector

    return get_detector(algo_name, **kwargs)


def _import_callable(dotted_path: str) -> Any:
    """Import a callable by its fully-qualified dotted path.

    Example
    -------
    >>> fn = _import_callable("benchmarks.overhead.natives.iforest_native")
    """
    module_path, func_name = dotted_path.rsplit(".", 1)
    mod = importlib.import_module(module_path)
    return getattr(mod, func_name)


def _detect_cuda(algo_kwargs: dict[str, Any]) -> bool:
    """Check if this run uses CUDA."""
    device = algo_kwargs.get("device", "auto")
    if device == "cpu":
        return False
    if device == "cuda":
        return True
    return bool(torch.cuda.is_available())


def _measure_model_size(model: Any) -> dict[str, float]:
    """Measure save/load time and archive size."""
    try:
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "model.zip")

            t0 = time.perf_counter()
            model.save(path)
            save_time = time.perf_counter() - t0

            size_mb = os.path.getsize(path) / (1024 * 1024)

            from omniad import get_detector

            loader = get_detector(model.__class__.__name__.replace("Adapter", ""))
            t0 = time.perf_counter()
            loader.load(path)
            load_time = time.perf_counter() - t0

            return {
                "model_size_mb": round(size_mb, 3),
                "save_time": round(save_time, 4),
                "load_time": round(load_time, 4),
            }
    except Exception:
        return {"model_size_mb": 0.0, "save_time": 0.0, "load_time": 0.0}


def _load_kwargs(config: dict[str, Any]) -> dict[str, Any]:
    """Build keyword arguments for load_data from config.

    Filters out None values so that loaders receive only
    explicitly provided overrides.
    """
    kwargs: dict[str, Any] = {}
    for key in ("n_samples", "n_features"):
        val = config.get(key)
        if val is not None:
            kwargs[key] = val
    return kwargs


def _count_samples_features(X: Any, y: Any) -> tuple[int, int]:
    """Return (n_samples, n_features) from loaded data."""
    n_samples = len(y) if hasattr(y, "__len__") else 0
    n_features = 0
    if isinstance(X, np.ndarray) and X.ndim > 1:
        n_features = X.shape[-1]
    return n_samples, n_features


# ------------------------------------------------------------------ #
#  Mode runners                                                      #
# ------------------------------------------------------------------ #


def run_quality(config: dict[str, Any]) -> dict[str, Any]:
    """Run quality benchmark: fit, predict, compute AUC."""
    algo = config["algorithm"]
    use_cuda = _detect_cuda(algo["kwargs"])

    X, y = load_data(
        config["domain"],
        config["dataset"],
        **_load_kwargs(config),
    )
    n_samples, n_features = _count_samples_features(X, y)

    clear_memory()

    model = _build_omniad_model(algo["name"], algo["kwargs"])

    cpu_monitor = CPUMemoryMonitor()
    gpu_monitor = GPUMemoryMonitor() if use_cuda else None

    cpu_monitor.start()
    if gpu_monitor:
        gpu_monitor.start()

    fit_time, _ = timed_call(
        model.fit,
        args=(X,),
        n_runs=1,
        warmup_runs=0,
        use_cuda_sync=use_cuda,
    )

    predict_time, scores = timed_call(
        model.predict_score,
        args=(X,),
        n_runs=config.get("n_runs", 3),
        warmup_runs=config.get("warmup_runs", 1),
        use_cuda_sync=use_cuda,
    )

    ram_mb = cpu_monitor.stop()
    vram_mb = gpu_monitor.stop() if gpu_monitor else 0.0

    # Truncate y to match scores if needed (e.g., windowed TS)
    if len(scores) < len(y):
        y = y[-len(scores) :]

    pr_auc = safe_auc(y, scores, "pr_auc")
    roc_auc = safe_auc(y, scores, "roc_auc")

    size_info = _measure_model_size(model)

    return {
        "algorithm": algo["name"],
        "dataset": config["dataset"],
        "domain": config["domain"],
        "mode": "quality",
        "pr_auc": round(pr_auc, 4),
        "roc_auc": round(roc_auc, 4),
        "fit_time": round(fit_time, 4),
        "predict_time": round(predict_time, 6),
        "ram_mb": round(ram_mb, 3),
        "vram_mb": round(vram_mb, 1),
        "n_samples": n_samples,
        "n_features": n_features,
        **size_info,
        "native_time": 0.0,
        "overhead_ratio": 0.0,
        "parity_max_diff": 0.0,
        "error": None,
    }


def run_overhead(config: dict[str, Any]) -> dict[str, Any]:
    """
    Run overhead benchmark.

    Supports two profiles:

    - throughput: full-batch inference
    - latency: request-by-request inference

    Returns
    -------
    dict[str, Any]
        Master Schema benchmark result.
    """
    memory_probe = config.get("_memory_probe")

    algo = config["algorithm"]
    baseline = config["native_baseline"]
    assert baseline is not None

    mode = config.get("mode", "throughput")
    use_cuda = _detect_cuda(algo["kwargs"])

    # --- Data ---

    X, y = load_data(
        config["domain"],
        config["dataset"],
        **_load_kwargs(config),
    )

    n_samples, n_features = _count_samples_features(
        X,
        y,
    )

    n_runs = config.get("n_runs", 5)
    warmup_runs = config.get("warmup_runs", 2)
    min_total_seconds = config.get("min_total_seconds", 2.0)
    latency_samples = config.get("latency_samples", 600)

    # --- Models ---

    clear_memory()

    omniad_model = _build_omniad_model(
        algo["name"],
        algo["kwargs"],
    )
    omniad_model.fit(X)

    native_builder = _import_callable(
        baseline["runner"],
    )

    native_predict = native_builder(
        omniad_model,
    )

    # --- Targets ---

    if mode == "latency":
        if config["domain"] == "timeseries":
            window_size = getattr(
                omniad_model,
                "window_size",
                10,
            )

            def omniad_target() -> Any:
                limit = min(
                    len(X) - window_size + 1,
                    latency_samples,
                )

                for i in range(limit):
                    omniad_model.predict_score(
                        X[i : i + window_size],
                    )

            def native_target() -> Any:
                limit = min(
                    len(X) - window_size + 1,
                    latency_samples,
                )

                for i in range(limit):
                    native_predict(
                        X[i : i + window_size],
                    )

        else:

            def omniad_target() -> Any:
                limit = min(
                    len(X),
                    latency_samples,
                )

                for i in range(limit):
                    omniad_model.predict_score(
                        X[i : i + 1],
                    )

            def native_target() -> Any:
                limit = min(
                    len(X),
                    latency_samples,
                )

                for i in range(limit):
                    native_predict(
                        X[i : i + 1],
                    )

    else:

        def omniad_target() -> Any:
            return omniad_model.predict_score(X)

        def native_target() -> Any:
            return native_predict(X)

    # --- Internal memory probe ---

    if memory_probe is not None:
        clear_memory()

        cpu_monitor = CPUMemoryMonitor()
        gpu_monitor = GPUMemoryMonitor() if use_cuda else None

        cpu_monitor.start()

        if gpu_monitor is not None:
            gpu_monitor.start()

        if memory_probe == "omniad":
            omniad_target()
        else:
            native_target()

        ram_mb = cpu_monitor.stop()

        vram_mb = gpu_monitor.stop() if gpu_monitor is not None else 0.0

        return {
            "ram_mb": round(ram_mb, 3),
            "vram_mb": round(vram_mb, 3),
            "error": None,
        }

    # --- Warmup ---

    for _ in range(warmup_runs):
        omniad_target()
        native_target()

    # --- Probe ---

    t0 = time.perf_counter()
    omniad_target()
    omniad_probe = time.perf_counter() - t0

    t0 = time.perf_counter()
    native_target()
    native_probe = time.perf_counter() - t0

    slowest_probe = max(omniad_probe, native_probe)

    min_seconds_runs = int(min_total_seconds / max(slowest_probe, 1e-9)) + 1

    if mode == "latency":
        actual_runs = n_runs
    else:
        actual_runs = max(
            n_runs,
            min_seconds_runs,
        )

    omniad_ram = 0.0
    native_ram = 0.0

    omniad_vram = 0.0
    native_vram = 0.0

    # --- OmniAD measurement ---

    clear_memory()

    cpu_monitor = CPUMemoryMonitor()
    gpu_monitor = GPUMemoryMonitor() if use_cuda else None

    cpu_monitor.start()

    if gpu_monitor is not None:
        gpu_monitor.start()

    omniad_time, _ = timed_call(
        omniad_target,
        n_runs=actual_runs,
        warmup_runs=0,
        use_cuda_sync=use_cuda,
        reducer="median",
    )

    omniad_ram = cpu_monitor.stop()

    omniad_vram = gpu_monitor.stop() if gpu_monitor is not None else 0.0

    # --- Native measurement ---

    clear_memory()

    cpu_monitor = CPUMemoryMonitor()
    gpu_monitor = GPUMemoryMonitor() if use_cuda else None

    cpu_monitor.start()

    if gpu_monitor is not None:
        gpu_monitor.start()

    native_time, _ = timed_call(
        native_target,
        n_runs=actual_runs,
        warmup_runs=0,
        use_cuda_sync=use_cuda,
        reducer="median",
    )

    native_ram = cpu_monitor.stop()

    native_vram = gpu_monitor.stop() if gpu_monitor is not None else 0.0

    # --- Time metrics ---

    overhead_ratio: float | None

    if native_time > 1e-12:
        overhead_ratio = omniad_time / native_time
    else:
        overhead_ratio = None

    overhead_ms = (omniad_time - native_time) * 1000.0 / max(n_samples, 1)

    # --- Memory metrics ---

    ram_overhead_mb = omniad_ram - native_ram

    vram_overhead_mb = omniad_vram - native_vram

    # --- Parity ---

    omniad_scores = omniad_model.predict_score(X)
    native_scores = native_predict(X)

    omniad_arr = np.asarray(
        omniad_scores,
        dtype=np.float64,
    ).ravel()

    native_arr = np.asarray(
        native_scores,
        dtype=np.float64,
    ).ravel()

    min_len = min(
        len(omniad_arr),
        len(native_arr),
    )

    if min_len > 0:
        parity_check = bool(
            np.allclose(
                omniad_arr[:min_len],
                native_arr[:min_len],
                rtol=1e-5,
            )
        )

        parity_max_diff = float(
            np.max(np.abs(omniad_arr[:min_len] - native_arr[:min_len]))
        )
    else:
        parity_check = False
        parity_max_diff = float("nan")

    # --- Result ---

    return {
        "algorithm": algo["name"],
        "dataset": config["dataset"],
        "domain": config["domain"],
        "mode": mode,
        "n_samples": n_samples,
        "n_features": n_features,
        # QUALITY
        "pr_auc": None,
        "roc_auc": None,
        # TIME
        "fit_time": None,
        "predict_time": round(omniad_time, 6),
        "native_time": round(native_time, 6),
        # OVERHEAD
        "overhead_ratio": (
            round(overhead_ratio, 4) if overhead_ratio is not None else None
        ),
        "overhead_ms": round(
            overhead_ms,
            6,
        ),
        # MEMORY
        "ram_mb": round(
            omniad_ram,
            3,
        ),
        "native_ram_mb": round(
            native_ram,
            3,
        ),
        "ram_overhead_mb": round(
            ram_overhead_mb,
            3,
        ),
        "vram_mb": round(
            omniad_vram,
            3,
        ),
        "native_vram_mb": round(
            native_vram,
            3,
        ),
        "vram_overhead_mb": round(
            vram_overhead_mb,
            3,
        ),
        # LIFECYCLE
        "model_size_mb": None,
        "save_time": None,
        "load_time": None,
        # VALIDATION
        "native_label": baseline.get(
            "native_label",
        ),
        "parity_check": parity_check,
        "parity_max_diff": round(
            parity_max_diff,
            10,
        ),
        "n_actual_runs": actual_runs,
        "error": None,
    }


def run_performance(config: dict[str, Any]) -> dict[str, Any]:
    """Run performance benchmark: scalability, model size."""
    algo = config["algorithm"]
    use_cuda = _detect_cuda(algo["kwargs"])

    n_samples = config.get("n_samples", 10000)
    n_features = config.get("n_features", 20)
    dataset = config.get("dataset", "synthetic")

    X, y = load_data(
        config["domain"],
        "synthetic",
        n_samples=n_samples,
        n_features=n_features,
    )
    actual_n, actual_f = _count_samples_features(X, y)

    clear_memory()

    model = _build_omniad_model(algo["name"], algo["kwargs"])

    cpu_monitor = CPUMemoryMonitor()
    gpu_monitor = GPUMemoryMonitor() if use_cuda else None

    cpu_monitor.start()
    if gpu_monitor:
        gpu_monitor.start()

    fit_time, _ = timed_call(
        model.fit,
        args=(X,),
        n_runs=1,
        warmup_runs=0,
        use_cuda_sync=use_cuda,
    )

    predict_time, scores = timed_call(
        model.predict_score,
        args=(X,),
        n_runs=config.get("n_runs", 5),
        warmup_runs=config.get("warmup_runs", 2),
        use_cuda_sync=use_cuda,
    )

    ram_mb = cpu_monitor.stop()
    vram_mb = gpu_monitor.stop() if gpu_monitor else 0.0

    size_info = _measure_model_size(model)

    label = f"{dataset}_{n_samples}" if dataset == "synthetic" else dataset

    return {
        "algorithm": algo["name"],
        "dataset": label,
        "domain": config["domain"],
        "mode": "performance",
        "pr_auc": 0.0,
        "roc_auc": 0.0,
        "fit_time": round(fit_time, 4),
        "predict_time": round(predict_time, 6),
        "ram_mb": round(ram_mb, 3),
        "vram_mb": round(vram_mb, 1),
        "n_samples": actual_n,
        "n_features": actual_f,
        **size_info,
        "native_time": 0.0,
        "overhead_ratio": 0.0,
        "parity_max_diff": 0.0,
        "error": None,
    }


RUNNERS = {
    "quality": run_quality,
    "throughput": run_overhead,
    "latency": run_overhead,
    "performance": run_performance,
}


def main() -> None:
    """Entry point: read config from stdin, run benchmark, print JSON result."""
    raw = sys.stdin.read()
    config = cast(dict[str, Any], json.loads(raw))

    mode = config.get("mode", "quality")
    runner = RUNNERS.get(mode)

    if runner is None:
        result = {"error": f"Unknown mode: {mode}"}
        print(json.dumps(result))
        sys.exit(1)

    try:
        result = runner(config)
    except Exception:
        result = {"error": traceback.format_exc()}

    print(json.dumps(result))


if __name__ == "__main__":
    main()
