from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pytest

import omniad.viz.plotting_ts as plotting


@pytest.mark.parametrize(
    ("mask", "expected"),
    [
        ([False, False, False], []),
        ([True, True, True], [(0, 2)]),
        ([False, True, True, False], [(1, 2)]),
        ([True, False, True, True, False, True], [(0, 0), (2, 3), (5, 5)]),
    ],
)
def test_find_anomaly_intervals(mask, expected) -> None:
    """Continuous anomaly regions are converted to inclusive intervals."""
    result = plotting._find_anomaly_intervals(np.asarray(mask))
    assert result == expected


def test_plot_timeseries_anomalies_saves_file(tmp_path) -> None:
    """Time-series visualization can be rendered headlessly."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=50)
    scores = np.abs(rng.normal(size=50))
    path = tmp_path / "timeseries.png"

    plotting.plot_timeseries_anomalies(
        X,
        scores,
        threshold=1.0,
        save_path=str(path),
    )

    assert path.exists()
    assert path.stat().st_size > 0


def test_plot_timeseries_anomalies_supports_expected_signal(tmp_path) -> None:
    """Expected values can be overlaid on the original signal."""
    X = np.sin(np.linspace(0, 4, 50))
    expected = X * 0.9
    scores = np.zeros(50)
    path = tmp_path / "expected.png"

    plotting.plot_timeseries_anomalies(
        X,
        scores,
        threshold=1.0,
        expected=expected,
        save_path=str(path),
    )

    assert path.exists()


def test_plot_timeseries_aligns_short_scores_with_leading_nan(monkeypatch) -> None:
    """Windowed scores are aligned to the end of the original series."""
    captured: dict[str, object] = {}

    def capture(fig, path=None):
        captured["figure"] = fig

    monkeypatch.setattr(plotting, "_save_or_show", capture)

    X = np.arange(10, dtype=float)
    scores = np.arange(7, dtype=float)

    plotting.plot_timeseries_anomalies(
        X,
        scores,
        threshold=100.0,
    )

    fig = captured["figure"]
    score_values = np.asarray(fig.axes[1].lines[0].get_ydata())

    assert np.isnan(score_values[:3]).all()
    np.testing.assert_array_equal(score_values[3:], scores)
    plt.close(fig)


def test_plot_timeseries_truncates_long_scores(monkeypatch) -> None:
    """Scores longer than the signal are truncated to its length."""
    captured: dict[str, object] = {}

    def capture(fig, path=None):
        captured["figure"] = fig

    monkeypatch.setattr(plotting, "_save_or_show", capture)

    X = np.arange(5, dtype=float)
    scores = np.arange(8, dtype=float)

    plotting.plot_timeseries_anomalies(
        X,
        scores,
        threshold=100.0,
    )

    fig = captured["figure"]
    score_values = np.asarray(fig.axes[1].lines[0].get_ydata())

    np.testing.assert_array_equal(score_values, scores[: len(X)])
    plt.close(fig)


def test_plot_timeseries_uses_first_feature_for_multivariate_input(
    monkeypatch,
) -> None:
    """Multivariate input visualizes its first feature."""
    captured: dict[str, object] = {}

    def capture(fig, path=None):
        captured["figure"] = fig

    monkeypatch.setattr(plotting, "_save_or_show", capture)

    X = np.column_stack(
        [
            np.arange(6, dtype=float),
            np.full(6, 100.0),
        ]
    )
    scores = np.zeros(6)

    plotting.plot_timeseries_anomalies(
        X,
        scores,
        threshold=1.0,
    )

    fig = captured["figure"]
    signal_values = np.asarray(fig.axes[0].lines[0].get_ydata())

    np.testing.assert_array_equal(signal_values, X[:, 0])
    plt.close(fig)


def test_plot_timeseries_aligns_short_expected_signal(monkeypatch) -> None:
    """Windowed expected values are aligned to the end of the signal."""
    captured: dict[str, object] = {}

    def capture(fig, path=None):
        captured["figure"] = fig

    monkeypatch.setattr(plotting, "_save_or_show", capture)

    X = np.arange(8, dtype=float)
    expected = np.arange(5, dtype=float)

    plotting.plot_timeseries_anomalies(
        X,
        np.zeros(8),
        threshold=1.0,
        expected=expected,
    )

    fig = captured["figure"]
    expected_values = np.asarray(fig.axes[0].lines[1].get_ydata())

    assert np.isnan(expected_values[:3]).all()
    np.testing.assert_array_equal(expected_values[3:], expected)
    plt.close(fig)
