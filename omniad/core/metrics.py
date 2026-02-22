"""
Universal anomaly scoring metrics.

Designed to be backend-agnostic. Currently supports NumPy and
PyTorch tensors natively.
Extensions for JAX/TensorFlow can be added via the _ops dispatcher.
"""
from __future__ import annotations

from typing import Any, Callable

import numpy as np

from omniad.core.exceptions import ConfigError

ScoreFunction = Callable[[Any, Any], Any]


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


# --- Built-in metrics ---


def _mse(target: Any, output: Any) -> Any:
    """Mean Squared Error per sample."""
    return ((target - output) ** 2).mean(-1)


def _mae(target: Any, output: Any) -> Any:
    """Mean Absolute Error per sample."""
    # abs() uses __abs__ dunder method, works for both tensor/ndarray
    return abs(target - output).mean(-1)


def _rmse(target: Any, output: Any) -> Any:
    """Root Mean Squared Error per sample."""
    return ((target - output) ** 2).mean(-1) ** 0.5


def _log_cosh(target: Any, output: Any) -> Any:
    """Log-Cosh loss per sample."""
    ops = _ops(target)
    diff = target - output
    # ops.cosh/log works for both torch/numpy modules
    return ops.log(ops.cosh(diff)).mean(-1)


def _huber(target: Any, output: Any, delta: float = 1.0) -> Any:
    """Huber loss per sample."""
    diff = abs(target - output)
    # clip is a method on both ndarray and Tensor (since torch 1.7+)
    quadratic = diff.clip(max=delta)
    linear = diff - quadratic
    return (0.5 * quadratic**2 + delta * linear).mean(-1)


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


def resolve_metric(metric: str | ScoreFunction) -> ScoreFunction:
    """
    Resolve metric: validate callable or look up string in registry.
    """
    if callable(metric):
        return metric

    if isinstance(metric, str):
        if metric not in _METRIC_REGISTRY:
            available = ", ".join(sorted(_METRIC_REGISTRY.keys()))
            raise ConfigError(
                f"Unknown metric '{metric}'. Available: [{available}]. "
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
