"""
Layer 1.5 template tests for BaseCompositionAdapter's delegation glue
— independent of any specific composed algorithm (see
tests/algos/text/test_tfidf.py for TfidfDetector-specific behavior).
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from omniad.core.adapters.composition_adapter import BaseCompositionAdapter
from omniad.core.exceptions import CapabilityError, ModelNotFittedError
from tests.doubles import DummyDetector

X = np.random.default_rng(0).normal(size=(10, 2))


class _FakeCompositionHost(BaseCompositionAdapter):
    """Minimal host satisfying BaseCompositionAdapter's contract."""

    detector = "fake"

    def __init__(self) -> None:
        self._detector: Any = None

    def _to_vectors(self, X: Any) -> Any:
        return X  # identity — vectorization itself isn't under test here


class _CapableInnerDetector:
    """Fake inner detector declaring every delegatable capability."""

    @classmethod
    def get_capabilities(cls) -> set[str]:
        return {"feature_importance", "reconstruction"}

    def get_feature_importances(self, X: Any, **kwargs: Any) -> np.ndarray:
        self.received_importances_input = X
        return np.array([1.0, 2.0])

    def predict_expected(self, X: Any) -> np.ndarray:
        self.received_expected_input = X
        return X


def test_require_delegated_raises_before_fit() -> None:
    host = _FakeCompositionHost()
    with pytest.raises(ModelNotFittedError):
        host.get_feature_importances(X)


def test_require_delegated_raises_when_inner_detector_lacks_capability() -> None:
    host = _FakeCompositionHost()
    host._detector = DummyDetector().fit(X)  # no FeatureImportanceMixin
    with pytest.raises(CapabilityError, match="feature_importance"):
        host.get_feature_importances(X)


def test_get_feature_importances_forwards_vectorized_input_to_inner_detector() -> None:
    host = _FakeCompositionHost()
    inner = _CapableInnerDetector()
    host._detector = inner

    result = host.get_feature_importances(X)

    np.testing.assert_array_equal(inner.received_importances_input, X)
    np.testing.assert_array_equal(result, [1.0, 2.0])


def test_predict_expected_forwards_vectorized_input_to_inner_detector() -> None:
    host = _FakeCompositionHost()
    inner = _CapableInnerDetector()
    host._detector = inner

    result = host.predict_expected(X)

    np.testing.assert_array_equal(inner.received_expected_input, X)
    np.testing.assert_array_equal(result, X)


def test_predict_expected_raises_when_inner_detector_lacks_capability() -> None:
    host = _FakeCompositionHost()
    host._detector = DummyDetector().fit(X)  # no ReconstructionMixin
    with pytest.raises(CapabilityError, match="reconstruction"):
        host.predict_expected(X)
