import numpy as np
import pytest

from omniad.core.exceptions import CapabilityError
from tests.doubles import DummyDetector, DummyStreamingDetector

X = np.random.default_rng(0).normal(size=(30, 4))


def test_partial_fit_requires_mixin() -> None:
    model = DummyDetector().fit(X)
    with pytest.raises(CapabilityError):
        model.partial_fit(X[0])


def test_partial_fit_requires_prior_fit() -> None:
    with pytest.raises(CapabilityError):
        DummyStreamingDetector().partial_fit(X[0])


def test_partial_fit_updates_threshold_with_streaming_strategy() -> None:
    model = DummyStreamingDetector(threshold_strategy="ewma").fit(X)
    before = model.threshold_
    for row in X[:10]:
        model.partial_fit(row)
    assert model.threshold_ != before


def test_partial_fit_with_batch_only_strategy_raises() -> None:
    model = DummyStreamingDetector(threshold_strategy="quantile").fit(X)
    with pytest.raises(CapabilityError, match="batch-only"):
        model.partial_fit(X[0])


def test_partial_fit_updates_backend_state() -> None:
    model = DummyStreamingDetector(threshold_strategy="ewma").fit(X)
    center_before = model._center.copy()
    model.partial_fit(np.full(4, 100.0))
    assert not np.allclose(center_before, model._center)


def test_streaming_threshold_state_survives_roundtrip(tmp_path) -> None:
    model = DummyStreamingDetector(threshold_strategy="ewma").fit(X)

    for row in X[:10]:
        model.partial_fit(row)

    path = tmp_path / "model.zip"
    model.save(str(path))
    loaded = DummyStreamingDetector().load(str(path))

    before = loaded.threshold_
    loaded.partial_fit(X[10])

    assert loaded.threshold_ != before
