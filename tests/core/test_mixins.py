import numpy as np
import pytest

from omniad.core.exceptions import CapabilityError
from tests.doubles import DummyFeatureImportanceDetector


def test_permutation_importance_ignores_constant_feature() -> None:
    """
    A zero-variance feature cannot affect a permutation-based score,
    regardless of which backend computes it: shuffling identical
    values is a no-op. This is a property of the mixin's algorithm
    itself, not of any specific detector — hence tested here, once,
    against DummyDetector's fully known Euclidean-distance scorer,
    instead of per-algorithm against a real, noisier backend.
    """
    rng = np.random.default_rng(0)
    varying = rng.normal(0, 5, size=200)
    constant = np.full(200, 3.0)
    X = np.column_stack([varying, constant])

    model = DummyFeatureImportanceDetector().fit(X)
    imp = model.get_feature_importances(X, n_repeats=5, random_state=0)

    assert imp[1] == pytest.approx(0.0, abs=1e-9)
    assert imp[0] > imp[1]


def test_native_mode_raises_when_backend_lacks_attribute() -> None:
    """method='native' must fail structurally (CapabilityError), not
    silently fall back to permutation, when the backend has no
    feature_importances_ — DummyDetector's backend is a plain dict."""
    X = np.random.default_rng(0).normal(size=(20, 3))
    model = DummyFeatureImportanceDetector().fit(X)
    with pytest.raises(CapabilityError):
        model.get_feature_importances(method="native")
