from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

import omniad.viz.plotting_tabular as plotting


def test_plot_anomaly_scores_saves_file(tmp_path) -> None:
    """Score histogram can be rendered headlessly."""
    scores = np.random.default_rng(0).normal(size=100)
    path = tmp_path / "scores.png"

    plotting.plot_anomaly_scores(
        scores,
        threshold=1.0,
        save_path=str(path),
    )

    assert path.exists()
    assert path.stat().st_size > 0


def test_plot_anomaly_scores_without_threshold_saves_file(tmp_path) -> None:
    """Threshold is optional."""
    scores = np.random.default_rng(0).normal(size=100)
    path = tmp_path / "scores.png"

    plotting.plot_anomaly_scores(scores, save_path=str(path))

    assert path.exists()
    assert path.stat().st_size > 0


def test_plot_scatter_2d_saves_labeled_data(tmp_path) -> None:
    """Two-dimensional labeled data can be rendered directly."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(30, 2))
    labels = np.array([0] * 25 + [1] * 5)
    path = tmp_path / "scatter.png"

    plotting.plot_scatter_2d(
        X,
        labels=labels,
        save_path=str(path),
    )

    assert path.exists()
    assert path.stat().st_size > 0


def test_plot_scatter_2d_supports_unlabeled_data(tmp_path) -> None:
    """Labels are optional."""
    X = np.random.default_rng(0).normal(size=(20, 2))
    path = tmp_path / "scatter.png"

    plotting.plot_scatter_2d(X, save_path=str(path))

    assert path.exists()


def test_plot_scatter_2d_pads_single_feature(monkeypatch) -> None:
    """One-dimensional input is plotted against a zero second component."""
    captured: dict[str, object] = {}

    def capture(fig, path=None):
        captured["figure"] = fig

    monkeypatch.setattr(plotting, "_save_or_show", capture)

    X = np.array([[1.0], [2.0], [3.0]])
    plotting.plot_scatter_2d(X)

    fig = captured["figure"]
    ax = fig.axes[0]
    offsets = ax.collections[0].get_offsets()

    np.testing.assert_allclose(offsets[:, 0], X[:, 0])
    np.testing.assert_allclose(offsets[:, 1], 0.0)
    plt.close(fig)


def test_plot_scatter_2d_uses_pca_for_high_dimensional_data(monkeypatch) -> None:
    """Inputs wider than two features are projected with PCA."""
    captured: dict[str, object] = {}

    def capture(fig, path=None):
        captured["figure"] = fig

    monkeypatch.setattr(plotting, "_save_or_show", capture)

    X = np.random.default_rng(0).normal(size=(30, 5))
    plotting.plot_scatter_2d(X, title="Projection")

    fig = captured["figure"]
    ax = fig.axes[0]

    assert "PCA" in ax.get_title()
    assert len(ax.collections[0].get_offsets()) == len(X)
    plt.close(fig)
