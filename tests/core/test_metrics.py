"""
Layer 1 core tests for the backend-agnostic scoring metrics registry.

Deliberately independent of any real anomaly-detection algorithm —
these are pure functions on arrays, tested against inline reference
formulas built directly from numpy/torch primitives (not by calling
the tested private helpers a second time under a different name).
"""
from __future__ import annotations

import numpy as np
import pytest

from omniad.core.exceptions import ConfigError
from omniad.core.metrics import (
    get_available_metrics,
    register_metric,
    resolve_metric,
    reverse_lookup_metric,
)

# Sample 0: diff straddles Huber's delta=1.0 in one feature (0.5, below)
# and exceeds it in the other (2.0, above) — exercises both branches.
# Sample 1: target == output, a zero-diff sanity check.
TARGET = np.array([[0.0, 0.0], [1.0, 1.0]])
OUTPUT = np.array([[0.5, 2.0], [1.0, 1.0]])


def test_mse_matches_manual_per_sample_formula() -> None:
    scores = resolve_metric("mse")(TARGET, OUTPUT)
    expected = np.mean((TARGET - OUTPUT) ** 2, axis=1)
    np.testing.assert_allclose(scores, expected, rtol=1e-10)


def test_mae_matches_manual_per_sample_formula() -> None:
    scores = resolve_metric("mae")(TARGET, OUTPUT)
    expected = np.mean(np.abs(TARGET - OUTPUT), axis=1)
    np.testing.assert_allclose(scores, expected, rtol=1e-10)


def test_rmse_is_sqrt_of_mse() -> None:
    mse_scores = resolve_metric("mse")(TARGET, OUTPUT)
    rmse_scores = resolve_metric("rmse")(TARGET, OUTPUT)
    np.testing.assert_allclose(rmse_scores, np.sqrt(mse_scores), rtol=1e-10)


def test_log_cosh_matches_manual_per_sample_formula() -> None:
    scores = resolve_metric("log_cosh")(TARGET, OUTPUT)
    diff = TARGET - OUTPUT
    expected = np.mean(np.log(np.cosh(diff)), axis=1)
    np.testing.assert_allclose(scores, expected, rtol=1e-10)


def test_huber_matches_manual_per_sample_formula_with_clipping() -> None:
    scores = resolve_metric("huber")(TARGET, OUTPUT)
    diff = np.abs(TARGET - OUTPUT)
    quadratic = np.clip(diff, a_min=None, a_max=1.0)
    linear = diff - quadratic
    expected = np.mean(0.5 * quadratic**2 + 1.0 * linear, axis=1)
    np.testing.assert_allclose(scores, expected, rtol=1e-10)


def test_all_metrics_agree_on_zero_diff_sample() -> None:
    """Sample 1 (target == output) must score exactly 0 for every
    built-in metric, independent of the formula's shape."""
    for name in ("mse", "mae", "rmse", "log_cosh", "huber"):
        scores = resolve_metric(name)(TARGET, OUTPUT)
        assert scores[1] == pytest.approx(0.0, abs=1e-12), name


def test_ops_dispatch_gives_same_result_for_numpy_and_torch() -> None:
    """The `_ops` trick must be truly backend-agnostic: identical
    values through numpy vs torch must produce identical scores."""
    torch = pytest.importorskip("torch")
    metric = resolve_metric("mse")

    np_scores = metric(TARGET, OUTPUT)
    torch_scores = metric(torch.from_numpy(TARGET), torch.from_numpy(OUTPUT)).numpy()

    np.testing.assert_allclose(np_scores, torch_scores, rtol=1e-6)


def test_get_available_metrics_includes_builtins() -> None:
    available = get_available_metrics()
    assert {"mse", "mae", "rmse", "log_cosh", "huber"} <= set(available)


def test_resolve_metric_passes_through_callables() -> None:
    def my_func(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return a - b

    assert resolve_metric(my_func) is my_func


def test_resolve_metric_unknown_name_raises_config_error() -> None:
    with pytest.raises(ConfigError, match="Unknown metric"):
        resolve_metric("not_a_real_metric")


def test_resolve_metric_rejects_non_str_non_callable() -> None:
    with pytest.raises(ConfigError):
        resolve_metric(123)  # type: ignore[arg-type]


def test_register_metric_rejects_non_callable() -> None:
    with pytest.raises(TypeError):
        register_metric("bad_metric_test", "not_callable")  # type: ignore[arg-type]


def test_register_and_reverse_lookup_metric_roundtrip() -> None:
    def my_custom_metric(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return np.abs(a - b).max(axis=1)

    register_metric("my_custom_metric_test", my_custom_metric)
    assert resolve_metric("my_custom_metric_test") is my_custom_metric
    assert reverse_lookup_metric(my_custom_metric) == "my_custom_metric_test"


def test_reverse_lookup_metric_returns_none_for_unregistered_function() -> None:
    def never_registered(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return a

    assert reverse_lookup_metric(never_registered) is None
