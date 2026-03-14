from pathlib import Path

import numpy as np
import pytest

from omniad.viz.plotting_tabular import plot_anomaly_scores, plot_scatter_2d
from omniad.viz.plotting_ts import plot_timeseries_anomalies

matplotlib = pytest.importorskip("matplotlib")
seaborn = pytest.importorskip("seaborn")


def test_plot_anomaly_scores(tmp_path: Path) -> None:
    scores = np.random.rand(100)
    save_path = tmp_path / "hist.png"

    plot_anomaly_scores(scores, threshold=0.8, save_path=str(save_path))
    assert save_path.exists()


def test_plot_scatter_2d(tmp_path: Path) -> None:
    # 3 features -> PCA will work
    X = np.random.randn(50, 3)
    labels = np.random.randint(0, 2, 50)
    save_path = tmp_path / "scatter.png"

    plot_scatter_2d(X, labels=labels, save_path=str(save_path))
    assert save_path.exists()


def test_plot_ts_anomalies_mismatch(tmp_path: Path) -> None:
    # We verify that the graph is constructed even if the lengths do not match
    X = np.random.randn(100)
    # scores less than data (due to window)
    scores = np.random.rand(90)
    save_path = tmp_path / "ts.png"

    plot_timeseries_anomalies(X, scores, threshold=0.8, save_path=str(save_path))
    assert save_path.exists()
