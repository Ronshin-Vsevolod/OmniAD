"""
Low-level utilities for timing, memory measurement, and CUDA sync.
"""

from __future__ import annotations

import gc
import os
import threading
import time
from typing import Any, Callable

import numpy as np
import numpy.typing as npt
import psutil

try:
    import torch
except ImportError:
    torch = None


class CPUMemoryMonitor:
    def __init__(self, interval: float = 0.01) -> None:
        self.interval = interval
        self._process = psutil.Process(os.getpid())
        self._stop = False
        self._peak = 0

    def _run(self) -> None:
        while not self._stop:
            rss = self._process.memory_info().rss
            self._peak = max(self._peak, rss)
            time.sleep(self.interval)

    def start(self) -> None:
        self._base = self._process.memory_info().rss
        self._peak = self._base
        self._stop = False

        self._thread = threading.Thread(target=self._run)
        self._thread.daemon = True
        self._thread.start()

    def stop(self) -> float:
        self._stop = True
        self._thread.join()

        return float((self._peak - self._base) / (1024**2))


class GPUMemoryMonitor:
    def __init__(self) -> None:
        if torch is None:
            raise RuntimeError("Torch not available for GPU monitoring.")

    def start(self) -> None:
        torch.cuda.reset_peak_memory_stats()

    def stop(self) -> float:
        value = torch.cuda.max_memory_allocated()
        return float(value / (1024**2))


def _cuda_available() -> bool:
    """Check if CUDA is available without importing torch at module level."""
    try:
        import torch

        return bool(torch.cuda.is_available())
    except ImportError:
        return False


def cuda_sync() -> None:
    """Synchronize CUDA if available. No-op on CPU."""
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except ImportError:
        pass


def clear_memory() -> None:
    """Force garbage collection and clear CUDA cache."""
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
    except ImportError:
        pass


def measure_vram_mb() -> float:
    """Return peak VRAM allocated in MB. Returns 0.0 if no GPU."""
    try:
        import torch

        if torch.cuda.is_available():
            return float(torch.cuda.max_memory_allocated() / (1024 * 1024))
    except ImportError:
        pass
    return 0.0


def timed_call(
    fn: Callable[..., Any],
    args: tuple[Any, ...] | None = None,
    kwargs: dict[str, Any] | None = None,
    n_runs: int = 5,
    warmup_runs: int = 2,
    use_cuda_sync: bool = False,
    reducer: str = "median",
) -> tuple[float, Any]:
    """
    Run fn exactly n_runs times, return (reduced_time, last_result).

    Parameters
    ----------
    fn : callable
        Function to benchmark.
    args : tuple or None
        Positional arguments.
    kwargs : dict or None
        Keyword arguments.
    n_runs : int
        Number of timed runs.
    warmup_runs : int
        Number of untimed warm-up runs (cache/JIT priming).
    use_cuda_sync : bool
        If True, call torch.cuda.synchronize() before/after timing.
    reducer : str
        "median" (default) or "min".

    Returns
    -------
    time : float
        Reduced execution time in seconds.
    result : Any
        Return value from the last call.
    """
    _args = args or ()
    _kwargs = kwargs or {}

    result = None
    for _ in range(warmup_runs):
        result = fn(*_args, **_kwargs)

    times: list[float] = []

    for _ in range(n_runs):
        clear_memory()

        if use_cuda_sync:
            cuda_sync()

        t0 = time.perf_counter()
        result = fn(*_args, **_kwargs)

        if use_cuda_sync:
            cuda_sync()

        times.append(time.perf_counter() - t0)

    if reducer == "median":
        return float(np.median(times)), result
    return float(np.min(times)), result


def safe_auc(
    y_true: npt.NDArray[Any],
    scores: npt.NDArray[Any],
    metric: str = "pr_auc",
) -> float:
    """
    Compute AUC metric, returning NaN on failure.

    Parameters
    ----------
    y_true : array of {0, 1}
    scores : array of floats (higher = more anomalous)
    metric : "pr_auc" or "roc_auc"
    """
    from sklearn.metrics import average_precision_score, roc_auc_score

    try:
        if len(np.unique(y_true)) < 2:
            return float("nan")

        if metric == "pr_auc":
            return float(average_precision_score(y_true, scores))
        elif metric == "roc_auc":
            return float(roc_auc_score(y_true, scores))
        else:
            return float("nan")
    except ValueError:
        return float("nan")
