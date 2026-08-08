"""
Universal anomaly scoring metrics.

Designed to be backend-agnostic. Currently supports NumPy and
PyTorch tensors natively.
Extensions for JAX/TensorFlow can be added via the _ops dispatcher.
"""
from __future__ import annotations

from typing import Any, Callable, Literal

import numpy as np

from omniad.core.exceptions import ConfigError

ScoreFunction = Callable[[Any, Any], Any]

MetricName = Literal["mse", "mae", "rmse", "log_cosh", "huber"]


# --- Lazy backend dispatch ---


def _ops(x: Any) -> Any:
    """
    Return math module (torch or numpy) matching the input type.
    """
    module_name = type(x).__module__
    if module_name.startswith("torch"):
        import torch

        return torch
    if module_name.startswith("jax"):  # Future-proof
        import jax.numpy as jnp

        return jnp
    return np


def _sample_axes(x: Any) -> tuple[int, ...]:
    """
    All axes except the batch axis (0), for per-sample reduction.
    """
    return tuple(range(1, x.ndim))


# --- Built-in metrics ---


def _mse(target: Any, output: Any) -> Any:
    """Mean Squared Error per sample."""
    diff = (target - output) ** 2
    return diff.mean(axis=_sample_axes(diff))


def _mae(target: Any, output: Any) -> Any:
    """Mean Absolute Error per sample."""
    diff = abs(target - output)
    return diff.mean(axis=_sample_axes(diff))


def _rmse(target: Any, output: Any) -> Any:
    """Root Mean Squared Error per sample."""
    diff = (target - output) ** 2
    return diff.mean(axis=_sample_axes(diff)) ** 0.5


def _log_cosh(target: Any, output: Any) -> Any:
    """Log-Cosh loss per sample."""
    ops = _ops(target)
    diff = target - output
    loss = ops.log(ops.cosh(diff))
    return loss.mean(axis=_sample_axes(loss))


def _huber(target: Any, output: Any, delta: float = 1.0) -> Any:
    """Huber loss per sample."""
    diff = abs(target - output)
    quadratic = diff.clip(max=delta)
    linear = diff - quadratic
    loss = 0.5 * quadratic**2 + delta * linear
    return loss.mean(axis=_sample_axes(loss))


# --- Registry ---

_METRIC_REGISTRY: dict[str, ScoreFunction] = {
    "mse": _mse,
    "mae": _mae,
    "rmse": _rmse,
    "log_cosh": _log_cosh,
    "huber": _huber,
}


def register_metric(name: str, func: ScoreFunction) -> None:
    """
    Register a custom scoring metric globally.

    Parameters
    ----------
    name : str
        Name to reference the metric by.
    func : callable
        Function (target, output) -> scores.
    """
    if not callable(func):
        raise TypeError(f"Expected callable, got {type(func)}")
    _METRIC_REGISTRY[name] = func


def get_available_metrics() -> list[str]:
    """List registered metric names."""
    return sorted(_METRIC_REGISTRY.keys())


def resolve_metric(metric: MetricName | str | ScoreFunction) -> ScoreFunction:
    """
    Resolve metric: validate callable or look up string in registry.
    """
    if callable(metric):
        return metric

    if isinstance(metric, str):
        if metric not in _METRIC_REGISTRY:
            raise ConfigError(
                f"Unknown metric '{metric}'. Available: {get_available_metrics()}. "
                "Use register_metric() to add custom ones."
            )
        return _METRIC_REGISTRY[metric]

    raise ConfigError(f"score_metric must be str or callable, got {type(metric)}")


def reverse_lookup_metric(func: Any) -> str | None:
    """Find registry name for a callable, or None."""
    for name, registered in _METRIC_REGISTRY.items():
        if registered is func:
            return name
    return None
