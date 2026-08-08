"""
Layer 1.5 template tests for BaseSklearnAdapter's own glue logic —
independent of any specific wrapped algorithm (see
tests/algos/tabular/test_iforest.py for IsolationForest-specific
parity/injection checks).
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from omniad.core.adapters.sklearn_adapter import BaseSklearnAdapter
from omniad.core.exceptions import ConfigError

X = np.random.default_rng(0).normal(size=(20, 3))


class _MissingBackendClsAdapter(BaseSklearnAdapter):
    """Adapter that never sets `_backend_cls` — misconfiguration guard."""

    def _save_backend(self, path: str) -> None:
        pass

    def _load_backend(self, path: str) -> None:
        pass


class _FitOnlyEstimator:
    """A minimal sklearn-shaped estimator exposing neither
    `decision_function` nor `score_samples`."""

    def __init__(self, **kwargs: Any) -> None:
        pass

    def fit(self, X: Any, y: Any = None) -> _FitOnlyEstimator:
        return self


class _NoScoringMethodAdapter(BaseSklearnAdapter):
    _backend_cls = _FitOnlyEstimator

    def _save_backend(self, path: str) -> None:
        pass

    def _load_backend(self, path: str) -> None:
        pass


def test_missing_backend_cls_raises_config_error_on_fit() -> None:
    with pytest.raises(ConfigError, match="_backend_cls"):
        _MissingBackendClsAdapter().fit(X)


def test_backend_without_scoring_methods_raises_config_error() -> None:
    with pytest.raises(ConfigError, match="decision_function"):
        _NoScoringMethodAdapter().fit(X)
