import omniad.viz as viz


def test_public_visualization_api() -> None:
    """Visualization functions are exposed from omniad.viz."""
    assert callable(viz.plot_anomaly_scores)
    assert callable(viz.plot_scatter_2d)
    assert callable(viz.plot_timeseries_anomalies)
    assert callable(viz.plot_anomaly_heatmap)
