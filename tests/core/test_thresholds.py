from typing import Any, cast

import numpy as np
import numpy.typing as npt
import pytest

from omniad import get_detector
from omniad.core.exceptions import ConfigError
from omniad.utils.thresholds import (
    get_available_thresholds,
    register_threshold,
    resolve_threshold,
    reverse_lookup_threshold,
)

# --- Built-in strategies ---


def test_quantile_threshold(random_xy_dataset: tuple[Any, Any, Any]) -> None:
    """Quantile strategy sets threshold at (1 - contamination) percentile."""
    X_train, _, _ = random_xy_dataset
    contamination = 0.1

    model = get_detector(
        "IsolationForest",
        contamination=contamination,
        threshold_strategy="quantile",
    )
    model.fit(X_train)

    scores = model.predict_score(X_train)
    expected = float(np.quantile(scores, 1 - contamination))

    assert model.threshold_ == pytest.approx(expected, rel=1e-6)


def test_sigma3_threshold(random_xy_dataset: tuple[Any, Any, Any]) -> None:
    """Sigma3 strategy sets threshold at mean + 3 * std."""
    X_train, _, _ = random_xy_dataset

    model = get_detector(
        "IsolationForest",
        threshold_strategy="sigma3",
    )
    model.fit(X_train)

    scores = model.predict_score(X_train)
    expected = float(scores.mean() + 3 * scores.std())

    assert model.threshold_ == pytest.approx(expected, rel=1e-6)


def test_iqr_threshold(random_xy_dataset: tuple[Any, Any, Any]) -> None:
    """IQR strategy sets threshold at Q3 + 1.5 * IQR."""
    X_train, _, _ = random_xy_dataset

    model = get_detector(
        "IsolationForest",
        threshold_strategy="iqr",
    )
    model.fit(X_train)

    scores = model.predict_score(X_train)
    q1, q3 = np.quantile(scores, [0.25, 0.75])
    expected = float(q3 + 1.5 * (q3 - q1))

    assert model.threshold_ == pytest.approx(expected, rel=1e-6)


# --- Float threshold ---


def test_float_threshold(random_xy_dataset: tuple[Any, Any, Any]) -> None:
    """Float threshold_strategy sets threshold directly."""
    X_train, _, _ = random_xy_dataset
    fixed_threshold = 0.42

    model = get_detector(
        "IsolationForest",
        threshold_strategy=fixed_threshold,
    )
    model.fit(X_train)

    assert model.threshold_ == pytest.approx(fixed_threshold)


# --- None threshold ---


def test_none_threshold(random_xy_dataset: tuple[Any, Any, Any]) -> None:
    """None threshold_strategy leaves threshold_ as None."""
    X_train, _, _ = random_xy_dataset

    model = get_detector(
        "IsolationForest",
        threshold_strategy=None,
    )
    model.fit(X_train)

    assert model.threshold_ is None


# --- resolve_threshold ---


def test_resolve_threshold_string() -> None:
    """resolve_threshold resolves string to callable."""
    fn = resolve_threshold("quantile")
    assert callable(fn)


def test_resolve_threshold_callable() -> None:
    """resolve_threshold passes callable through unchanged."""

    def my_fn(scores: npt.NDArray[Any], contamination: float) -> float:
        return float(scores.max())

    fn = resolve_threshold(my_fn)
    assert fn is my_fn


def test_resolve_threshold_unknown_raises() -> None:
    """resolve_threshold raises ConfigError for unknown strategy."""
    with pytest.raises(ConfigError, match="Unknown threshold strategy"):
        resolve_threshold("nonexistent")


# --- register_threshold ---


def test_register_custom_threshold(random_xy_dataset: tuple[Any, Any, Any]) -> None:
    """Custom threshold strategy can be registered and used."""
    X_train, _, _ = random_xy_dataset

    def max_threshold(scores: npt.NDArray[Any], contamination: float) -> float:
        return float(scores.max())

    register_threshold("max", max_threshold)

    model = get_detector(
        "IsolationForest",
        threshold_strategy="max",
    )
    model.fit(X_train)

    scores = model.predict_score(X_train)
    assert model.threshold_ == pytest.approx(float(scores.max()), rel=1e-6)


def test_register_threshold_non_callable_raises() -> None:
    """register_threshold raises TypeError for non-callable."""
    with pytest.raises(TypeError):
        register_threshold("bad", "not_a_function")  # type: ignore[arg-type]


# --- reverse_lookup_threshold ---


def test_reverse_lookup_builtin() -> None:
    """reverse_lookup_threshold finds name for built-in strategy."""
    fn = resolve_threshold("quantile")
    name = reverse_lookup_threshold(fn)
    assert name == "quantile"


def test_reverse_lookup_custom() -> None:
    """reverse_lookup_threshold finds name for registered custom strategy."""

    def my_custom(scores: npt.NDArray[Any], contamination: float) -> npt.NDArray[Any]:
        threshold = cast("npt.NDArray[Any]", scores).median()
        return threshold

    register_threshold("my_custom", my_custom)
    name = reverse_lookup_threshold(my_custom)
    assert name == "my_custom"


def test_reverse_lookup_unregistered_returns_none() -> None:
    """reverse_lookup_threshold returns None for unregistered callable."""

    def unregistered(scores: npt.NDArray[Any], contamination: float) -> float:
        return 0.0

    assert reverse_lookup_threshold(unregistered) is None


# --- get_available_thresholds ---


def test_get_available_thresholds() -> None:
    """get_available_thresholds returns all built-in strategies."""
    available = get_available_thresholds()
    assert "quantile" in available
    assert "sigma3" in available
    assert "iqr" in available
