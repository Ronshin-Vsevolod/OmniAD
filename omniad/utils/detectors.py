"""
Registry of tabular anomaly detectors usable as backends
in embedding-based pipelines (text, CV, graph).

Uses the library's own registry and adapters,
"""
from __future__ import annotations

from typing import Any

from omniad.core.exceptions import ConfigError

# Subset of the main registry that makes sense as embedding backends.
_TABULAR_DETECTORS: set[str] = {
    "IsolationForest",
}


def register_tabular_detector(name: str) -> None:
    """
    Register an OmniAD algorithm as a valid backend for embedding pipelines.

    The algorithm must already be registered in omniad.registry
    and must accept 2D numpy input (n_samples, n_features).

    Parameters
    ----------
    name : str
        Algorithm name as registered in omniad.registry.
    """
    _TABULAR_DETECTORS.add(name)


def get_available_detectors() -> list[str]:
    """Return names of all detectors usable as embedding backends."""
    return sorted(_TABULAR_DETECTORS)


def build_detector(
    name: str,
    caller: str = "Detector",
    **kwargs: Any,
) -> Any:
    """
    Build an OmniAD detector for use as backend in embedding pipelines.

    Parameters
    ----------
    name : str
        Detector name (must be in _TABULAR_DETECTORS).
    caller : str
        Calling adapter name (for error messages).
    **kwargs
        Parameters passed to the detector's __init__
        (contamination, random_state, n_estimators, etc.).

    Returns
    -------
    detector : BaseDetector
        OmniAD detector with fit() and predict_score().
        predict_score() returns higher = more anomalous.
    """
    if name not in _TABULAR_DETECTORS:
        raise ConfigError(
            f"{caller}: '{name}' is not a valid tabular detector. "
            f"Available: {get_available_detectors()}"
        )

    # Lazy import to avoid circular dependency
    from omniad import get_detector

    return get_detector(name, **kwargs)
