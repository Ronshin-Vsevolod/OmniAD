"""
At the moment, this is an unnecessary file, but it could potentially be useful.


Typed configurations for the benchmarking subsystem.
"""
from __future__ import annotations

from typing import Any

from typing_extensions import TypedDict


class AlgoConfig(TypedDict):
    """Configuration for a single algorithm benchmark run."""

    name: str
    domain: str
    kwargs: dict[str, Any]


class NativeBaseline(TypedDict):
    """Configuration for a native (non-omniad) baseline.

    ``runner`` is a fully-qualified dotted path to a function with
    signature ``fn(X, **kwargs) -> Callable[[], NDArray]``.
    The returned callable produces anomaly scores; only it is timed.
    """

    runner: str


class ScenarioConfig(TypedDict):
    """Full scenario passed to the worker process."""

    mode: str  # "quality", "overhead", "performance"
    algorithm: AlgoConfig
    dataset: str
    domain: str
    n_runs: int
    warmup_runs: int
    # Only for overhead mode
    native_baseline: NativeBaseline | None
    # Only for scalability mode
    n_samples: int | None
    n_features: int | None


class BenchmarkResult(TypedDict):
    """Result from a single worker run."""

    algorithm: str
    dataset: str
    domain: str
    mode: str
    pr_auc: float
    roc_auc: float
    fit_time: float
    predict_time: float
    ram_mb: float
    vram_mb: float
    n_samples: int
    n_features: int
    # Overhead-specific
    native_time: float
    overhead_ratio: float
    parity_check: bool
    parity_max_diff: float
    # Model size
    model_size_mb: float
    save_time: float
    load_time: float
    # Metadata
    error: str | None
