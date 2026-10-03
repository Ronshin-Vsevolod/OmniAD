from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

import omniad.viz.plotting_cv as plotting


def test_plot_anomaly_heatmap_saves_rgb_chw_image(tmp_path) -> None:
    """CHW RGB images can be rendered with an anomaly overlay."""
    rng = np.random.default_rng(0)
    image = rng.random((3, 16, 16)).astype(np.float32)
    anomaly_map = rng.random((16, 16)).astype(np.float32)
    path = tmp_path / "heatmap.png"

    plotting.plot_anomaly_heatmap(
        image,
        anomaly_map,
        save_path=str(path),
    )

    assert path.exists()
    assert path.stat().st_size > 0


def test_plot_anomaly_heatmap_supports_grayscale(tmp_path) -> None:
    """Two-dimensional grayscale images are supported."""
    image = np.arange(64, dtype=np.float32).reshape(8, 8)
    anomaly_map = np.zeros((8, 8), dtype=np.float32)
    path = tmp_path / "grayscale.png"

    plotting.plot_anomaly_heatmap(
        image,
        anomaly_map,
        save_path=str(path),
    )

    assert path.exists()


def test_plot_anomaly_heatmap_supports_single_channel_chw(tmp_path) -> None:
    """Single-channel CHW images are converted to grayscale."""
    image = np.ones((1, 8, 8), dtype=np.float32)
    anomaly_map = np.zeros((8, 8), dtype=np.float32)
    path = tmp_path / "single-channel.png"

    plotting.plot_anomaly_heatmap(
        image,
        anomaly_map,
        save_path=str(path),
    )

    assert path.exists()


def test_plot_anomaly_heatmap_normalizes_uint8(monkeypatch) -> None:
    """uint8 images are normalized to the [0, 1] display range."""
    captured: dict[str, object] = {}

    def capture(fig, path=None):
        captured["figure"] = fig

    monkeypatch.setattr(plotting, "_save_or_show", capture)

    image = np.full((3, 4, 4), 255, dtype=np.uint8)
    anomaly_map = np.zeros((4, 4), dtype=np.float32)

    plotting.plot_anomaly_heatmap(image, anomaly_map)

    fig = captured["figure"]
    displayed = np.asarray(fig.axes[0].images[0].get_array())

    np.testing.assert_allclose(displayed, 1.0)
    plt.close(fig)


def test_plot_anomaly_heatmap_scales_large_float_values(monkeypatch) -> None:
    """Float images above one are scaled for display."""
    captured: dict[str, object] = {}

    def capture(fig, path=None):
        captured["figure"] = fig

    monkeypatch.setattr(plotting, "_save_or_show", capture)

    image = np.full((4, 4), 10.0, dtype=np.float32)
    anomaly_map = np.zeros((4, 4), dtype=np.float32)

    plotting.plot_anomaly_heatmap(image, anomaly_map)

    fig = captured["figure"]
    displayed = np.asarray(fig.axes[0].images[0].get_array())

    np.testing.assert_allclose(displayed, 1.0)
    plt.close(fig)
