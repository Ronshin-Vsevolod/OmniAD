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


class _ScoreSamplesEstimator:
    """Minimal estimator exposing only score_samples()."""

    def __init__(self, **kwargs: Any) -> None:
        pass

    def fit(self, X: Any, y: Any = None) -> _ScoreSamplesEstimator:
        return self

    def score_samples(self, X: Any) -> np.ndarray:
        return np.asarray(X).sum(axis=1)


class _ScoreSamplesAdapter(BaseSklearnAdapter):
    _backend_cls = _ScoreSamplesEstimator

    def _save_backend(self, path: str) -> None:
        pass

    def _load_backend(self, path: str) -> None:
        pass


def test_score_samples_fallback_is_inverted() -> None:
    """score_samples() is supported when decision_function() is absent."""
    model = _ScoreSamplesAdapter(threshold_strategy=None).fit(X)

    expected = -X.sum(axis=1)
    np.testing.assert_allclose(model.predict_score(X), expected)


def test_load_backend_raises_when_model_file_is_missing(tmp_path) -> None:
    """Sklearn backend loading requires its native model file."""
    from omniad import get_detector

    model = get_detector("IsolationForest")

    with pytest.raises(FileNotFoundError, match="model.joblib"):
        model._load_backend(str(tmp_path))
