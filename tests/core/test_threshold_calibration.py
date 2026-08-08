import numpy as np
import pytest

from omniad.core.exceptions import ConfigError
from omniad.utils.thresholds import QuantileThreshold
from tests.doubles import DummyCachingDetector, DummyDetector

X = np.random.default_rng(1).normal(size=(200, 3))


def test_quantile_threshold_matches_the_engine_it_delegates_to() -> None:
    model = DummyDetector(contamination=0.1, threshold_strategy="quantile").fit(X)
    scores = model.predict_score(X)
    expected = QuantileThreshold().fit(scores, 0.1)
    assert model.threshold_ == pytest.approx(expected)


def test_fixed_float_threshold() -> None:
    assert DummyDetector(threshold_strategy=2.5).fit(X).threshold_ == 2.5


def test_none_threshold_disables_calibration() -> None:
    model = DummyDetector(threshold_strategy=None).fit(X)
    assert model.threshold_ is None
    assert model._threshold_engine is None


def test_custom_callable_threshold() -> None:
    def double_median(scores, contamination):
        return float(np.median(scores) * 2)

    model = DummyDetector(threshold_strategy=double_median).fit(X)
    scores = model.predict_score(X)
    assert model.threshold_ == pytest.approx(np.median(scores) * 2, rel=1e-6)


def test_unknown_threshold_name_raises() -> None:
    with pytest.raises(ConfigError):
        DummyDetector(threshold_strategy="not_a_real_strategy").fit(X)


def test_cached_train_scores_skip_second_predict_call(monkeypatch) -> None:
    model = DummyCachingDetector()
    calls = {"count": 0}
    original = model.predict_score

    def counting(X):
        calls["count"] += 1
        return original(X)

    monkeypatch.setattr(model, "predict_score", counting)
    model.fit(X)
    assert calls["count"] == 0  # cache populated in _fit_backend already


def test_without_cache_calibration_calls_predict_score_once(monkeypatch) -> None:
    model = DummyDetector()
    calls = {"count": 0}
    original = model.predict_score

    def counting(X):
        calls["count"] += 1
        return original(X)

    monkeypatch.setattr(model, "predict_score", counting)
    model.fit(X)
    assert calls["count"] == 1
