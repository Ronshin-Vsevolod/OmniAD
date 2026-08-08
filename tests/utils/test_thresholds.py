import numpy as np
import pytest

from omniad.core.exceptions import CapabilityError, ConfigError
from omniad.utils.thresholds import (
    EWMAThreshold,
    IQRThreshold,
    QuantileThreshold,
    Sigma3Threshold,
    ThresholdStrategy,
    get_available_thresholds,
    register_threshold,
    resolve_threshold,
    reverse_lookup_threshold,
)

scores = np.array([1.0, 2.0, 3.0, 4.0, 100.0])


# Spec pins: one direct formula check per strategy. Catches typos
# in constants (e.g. `contamination` instead of `1 - contamination`),
# not conceptual errors — the strategy's *behavioral* contract is
# covered separately below.


def test_quantile_threshold_formula_pin() -> None:
    assert QuantileThreshold().fit(scores, 0.2) == pytest.approx(
        np.quantile(scores, 0.8)
    )


def test_sigma3_threshold_formula_pin() -> None:
    assert Sigma3Threshold().fit(scores, 0.1) == pytest.approx(
        scores.mean() + 3 * scores.std()
    )


def test_iqr_threshold_formula_pin() -> None:
    q1, q3 = np.quantile(scores, [0.25, 0.75])
    assert IQRThreshold().fit(scores, 0.1) == pytest.approx(q3 + 1.5 * (q3 - q1))


# --- Behavioral contracts ---


def test_sigma3_threshold_ignores_contamination_by_design() -> None:
    """Documents a real API surprise: unlike Quantile/IQR, Sigma3 does not use `contamination` at all."""
    t = Sigma3Threshold()
    assert t.fit(scores, 0.01) == t.fit(scores, 0.5)


def test_quantile_threshold_is_monotonic_in_contamination() -> None:
    """Higher expected contamination must lower the bar for 'anomalous'."""
    low = QuantileThreshold().fit(scores, 0.05)
    high = QuantileThreshold().fit(scores, 0.5)
    assert high < low


def test_iqr_threshold_is_insensitive_to_contamination() -> None:
    """IQR is a fixed statistical rule; contamination is accepted but unused."""
    t = IQRThreshold()
    assert t.fit(scores, 0.01) == t.fit(scores, 0.9)


@pytest.mark.parametrize("cls", [QuantileThreshold, Sigma3Threshold, IQRThreshold])
def test_batch_only_strategies_raise_on_update(cls) -> None:
    t = cls()
    t.fit(scores, 0.1)
    with pytest.raises(CapabilityError, match="batch-only"):
        t.update(5.0, 0.1)


def test_ewma_converges_toward_a_repeated_new_score() -> None:
    """A genuine mechanism check: repeated evidence should pull the running mean toward it."""
    t = EWMAThreshold(alpha=0.3, k=0.0)  # k=0 isolates the running mean itself
    t.fit(scores, 0.1)
    first = t.update(1000.0, 0.1)
    second = t.update(1000.0, 0.1)
    assert second > first  # still climbing toward 1000 after two pulls


def test_ewma_update_without_prior_fit_seeds_from_first_score() -> None:
    assert EWMAThreshold().update(5.0, 0.1) == pytest.approx(5.0)


def test_get_available_thresholds_streaming_filter() -> None:
    streaming = get_available_thresholds(streaming=True)
    batch_only = get_available_thresholds(streaming=False)
    assert "ewma" in streaming
    assert "quantile" in batch_only
    assert not set(streaming) & set(batch_only)


def test_resolve_threshold_dispatches_every_supported_input_kind() -> None:
    assert isinstance(resolve_threshold("quantile"), QuantileThreshold)
    instance = EWMAThreshold()
    assert resolve_threshold(instance) is instance
    assert isinstance(resolve_threshold(EWMAThreshold), EWMAThreshold)
    assert resolve_threshold(lambda s, c: 1.0).fit(scores, 0.1) == 1.0


def test_resolve_threshold_unknown_name_raises() -> None:
    with pytest.raises(ConfigError):
        resolve_threshold("not_a_real_strategy")


def test_register_custom_function_threshold_and_reverse_lookup() -> None:
    def half_max(s, c):
        return float(s.max() / 2)

    register_threshold("half_max_test", half_max)
    assert resolve_threshold("half_max_test").fit(scores, 0.1) == pytest.approx(50.0)
    assert reverse_lookup_threshold(half_max) == "half_max_test"


def test_register_custom_class_threshold_supports_streaming() -> None:
    class AlwaysTen(ThresholdStrategy):
        def fit(self, scores, contamination):
            return 10.0

        def update(self, score, contamination):
            return 10.0

    register_threshold("always_ten_test", AlwaysTen)
    assert "always_ten_test" in get_available_thresholds(streaming=True)
