"""
Visualization module.

Provides plotting utilities for
tabular data (scatter, hist),
time series (signal reconstruction, anomaly shading).
"""

from omniad.viz.plotting_tabular import plot_anomaly_scores, plot_scatter_2d
from omniad.viz.plotting_ts import plot_timeseries_anomalies

__all__ = [
    "plot_anomaly_scores",
    "plot_scatter_2d",
    "plot_timeseries_anomalies",
]
