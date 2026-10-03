"""
Minimal concrete BaseDetector implementations, used to exercise core
lifecycle logic (fit/predict/save/load/partial_fit/error translation)
without depending on any real ML backend.
"""
from __future__ import annotations

import os
from typing import Any

import numpy as np
import numpy.typing as npt

from omniad.core.base import BaseDetector
from omniad.core.mixins import FeatureImportanceMixin, IncrementalLearningMixin


class DummyDetector(BaseDetector):
    """
    Backend: Euclidean distance to the per-feature training mean.

    Parameters
    ----------
    fail_on : {"none", "fit", "predict", "save", "load"}, default="none"
        Raise a plain RuntimeError from the corresponding backend hook.
    """

    def __init__(
        self, contamination: float = 0.1, fail_on: str = "none", **kwargs: Any
    ) -> None:
        super().__init__(contamination=contamination, **kwargs)
        self.fail_on = fail_on
        self._center: npt.NDArray[Any] | None = None

    def _fit_backend(self, X: Any, y: Any | None = None) -> None:
        if self.fail_on == "fit":
            raise RuntimeError("boom")
        self._center = X.mean(axis=0)
        self._backend_model = {"center": self._center}

    def _predict_score_backend(self, X: Any) -> npt.NDArray[Any]:
        if self.fail_on == "predict":
            raise RuntimeError("boom")
        assert self._center is not None
        return np.linalg.norm(X - self._center, axis=1)

    def _save_backend(self, path: str) -> None:
        if self.fail_on == "save":
            raise RuntimeError("boom")
        assert self._center is not None
        np.save(os.path.join(path, "center.npy"), self._center)

    def _load_backend(self, path: str) -> None:
        if self.fail_on == "load":
            raise RuntimeError("boom")
        self._center = np.load(os.path.join(path, "center.npy"))
        self._backend_model = {"center": self._center}


class DummyCachingDetector(DummyDetector):
    """DummyDetector that pre-populates `_cached_train_scores`, mirroring
    the real Layer 1.5 templates (sklearn/torch/river/transformers)."""

    def _fit_backend(self, X: Any, y: Any | None = None) -> None:
        super()._fit_backend(X, y)
        self._cached_train_scores = self._predict_score_backend(X)


class DummyStreamingDetector(DummyDetector, IncrementalLearningMixin):
    """DummyDetector variant declaring streaming support, for partial_fit() tests."""

    def _partial_fit_backend(self, x: Any, y: Any | None = None) -> None:
        assert self._center is not None
        self._center = 0.9 * self._center + 0.1 * x[0]


class _TestOnlyDummyAdapter(DummyDetector):
    """
    Adapter registered only transiently, by registry_and_factory tests,
    to exercise get_detector()/describe() plumbing without depending
    on any real algorithm's tunable hyperparameters.
    """

    def __init__(
        self, contamination: float = 0.1, n_estimators: int = 7, **kwargs: Any
    ) -> None:
        self.n_estimators = n_estimators
        super().__init__(contamination=contamination, **kwargs)


class DummyFeatureImportanceDetector(DummyDetector, FeatureImportanceMixin):
    """DummyDetector + FeatureImportanceMixin, for exercising the
    permutation-importance algorithm against a fully known scoring
    function, independent of any real anomaly-detection backend."""
