import numpy as np
import pytest

from omniad.core.exceptions import ModelNotFittedError
from tests.doubles import DummyDetector

X = np.random.default_rng(0).normal(size=(30, 4))


def test_unfitted_model_state() -> None:
    model = DummyDetector()
    assert not model._is_fitted
    assert model.threshold_ is None


def test_predict_requires_fit() -> None:
    with pytest.raises(ModelNotFittedError):
        DummyDetector().predict(X)


def test_fit_sets_threshold_and_fitted_flag() -> None:
    model = DummyDetector().fit(X)
    assert model._is_fitted
    assert model.threshold_ is not None


def test_predict_without_threshold_raises() -> None:
    model = DummyDetector(threshold_strategy=None).fit(X)
    with pytest.raises(ValueError):
        model.predict(X)


def test_predict_uses_explicit_threshold_override() -> None:
    model = DummyDetector().fit(X)
    default_count = model.predict(X).sum()
    forced_count = model.predict(X, threshold=-1e9).sum()
    assert forced_count >= default_count


def test_backend_model_property_before_fit_raises() -> None:
    with pytest.raises(ModelNotFittedError):
        _ = DummyDetector().backend_model
