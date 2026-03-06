from __future__ import annotations

from typing import Any, Callable

import numpy as np
import numpy.typing as npt

from omniad.core.exceptions import ConfigError

ThresholdFunction = Callable[[npt.NDArray[Any], float], float]
# signature: (scores, contamination) -> threshold


def _quantile(scores: npt.NDArray[Any], contamination: float) -> float:
    """Percentile-based threshold."""
    return float(np.quantile(scores, 1 - contamination))


def _sigma3(scores: npt.NDArray[Any], contamination: float) -> float:
    """Mean + 3 standard deviations."""
    return float(scores.mean() + 3 * scores.std())


def _iqr(scores: npt.NDArray[Any], contamination: float) -> float:
    """Interquartile range: Q3 + 1.5 * IQR."""
    q1, q3 = np.quantile(scores, [0.25, 0.75])
    return float(q3 + 1.5 * (q3 - q1))


_THRESHOLD_REGISTRY: dict[str, ThresholdFunction] = {
    "quantile": _quantile,
    "sigma3": _sigma3,
    "iqr": _iqr,
}


def register_threshold(name: str, func: ThresholdFunction) -> None:
    """Register a custom threshold strategy."""
    if not callable(func):
        raise TypeError(f"Expected callable, got {type(func)}")
    _THRESHOLD_REGISTRY[name] = func


def get_available_thresholds() -> list[str]:
    return sorted(_THRESHOLD_REGISTRY.keys())


def resolve_threshold(
    strategy: str | ThresholdFunction,
) -> ThresholdFunction:
    if callable(strategy):
        return strategy
    if isinstance(strategy, str):
        if strategy not in _THRESHOLD_REGISTRY:
            raise ConfigError(
                f"Unknown threshold strategy '{strategy}'. "
                f"Available: {get_available_thresholds()}"
            )
        return _THRESHOLD_REGISTRY[strategy]
    raise ConfigError(
        f"threshold_strategy must be str, callable, float, or None. "
        f"Got {type(strategy)}"
    )


def reverse_lookup_threshold(func: ThresholdFunction) -> str | None:
    for name, registered in _THRESHOLD_REGISTRY.items():
        if registered is func:
            return name
    return None
