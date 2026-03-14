from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

from omniad.viz.utils import _check_viz_deps, _save_or_show

try:
    import matplotlib.pyplot as plt
except ImportError:
    pass


def _find_anomaly_intervals(mask: npt.NDArray[Any]) -> list[tuple[int, int]]:
    """
    Find continuous intervals where mask is True.
    Returns list of (start, end) indices.
    """
    if not np.any(mask):
        return []

    # Pad with False to detect edges correctly
    extended_mask = np.concatenate(([False], mask, [False]))
    diff = np.diff(extended_mask.astype(int))

    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0] - 1

    return list(zip(starts, ends))


def plot_timeseries_anomalies(
    X: npt.NDArray[Any],
    scores: npt.NDArray[Any],
    threshold: float,
    expected: npt.NDArray[Any] | None = None,
    title: str = "Time Series Anomalies",
    save_path: str | None = None,
) -> None:
    """
    Plot time series with anomaly highlighting and optional expected.

    This function aligns scores and expected to X (padding with NaN
    at the beginning) if they are shorter due to windowing.

    Parameters
    ----------
    X : np.ndarray
        Original time series signal. Should be 1D array.
        If 2D (N, F) is passed, only the first feature (column 0) is plotted.
    scores : np.ndarray
        Anomaly scores per time step.
    threshold : float
        Threshold value. Scores above this value are highlighted as anomalies.
    expected : np.ndarray, optional
        Expected signal (e.g., reconstruction from an autoencoder or prediction
        from a forecasting model).
        Should have the same length as X
        (after any necessary alignment, e.g., window padding).
        If provided, it will be plotted as a dashed line over the original signal.
    title : str, default="Time Series Anomalies"
        Plot title.
    save_path : str, optional
        Path to save the figure. If None, shows interactively.
    """
    _check_viz_deps()

    # We visualize only the first feature if data is multivariate
    if X.ndim > 1:
        X_plot = X[:, 0]
    else:
        X_plot = X

    X_plot = np.asarray(X_plot).flatten()
    len_x = len(X_plot)

    # 2. Align Scores (Padding)
    scores_plot = np.asarray(scores).flatten()
    len_s = len(scores_plot)

    if len_s < len_x:
        pad_width = len_x - len_s
        scores_padded = np.concatenate([np.full(pad_width, np.nan), scores_plot])
    elif len_s > len_x:
        scores_padded = scores_plot[:len_x]
    else:
        scores_padded = scores_plot

    # 3. Align Expected (Padding)
    recon_plot: npt.NDArray[Any] | None = None
    if expected is not None:
        if expected.ndim > 1:
            recon_raw = expected[:, 0]
        else:
            recon_raw = expected
        recon_raw = np.asarray(recon_raw).flatten()

        len_r = len(recon_raw)
        if len_r < len_x:
            pad_r = len_x - len_r
            recon_plot = np.concatenate([np.full(pad_r, np.nan), recon_raw])
        elif len_r > len_x:
            recon_plot = recon_raw[:len_x]
        else:
            recon_plot = recon_raw

    # 4. Plotting
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # --- Top Plot: Signal & Expected ---
    ax1.plot(X_plot, color="black", label="Original", linewidth=1.5, alpha=0.7)

    if recon_plot is not None:
        ax1.plot(
            recon_plot,
            color="green",
            linestyle="--",
            label="Expected",
            linewidth=1.5,
        )

    # Highlight Anomalies (Red Zones)
    # Ignore NaNs during comparison
    with np.errstate(invalid="ignore"):
        anomaly_mask = scores_padded > threshold

    intervals = _find_anomaly_intervals(anomaly_mask)
    for i, (start, end) in enumerate(intervals):
        # Label only the first interval to avoid duplicate legend entries
        label = "Anomaly" if i == 0 else None
        ax1.axvspan(start, end, alpha=0.3, color="red", label=label)
        ax2.axvspan(start, end, alpha=0.3, color="red")

    ax1.set_ylabel("Value")
    ax1.legend(loc="upper left")
    ax1.set_title(title)
    ax1.grid(True, alpha=0.3)

    # --- Bottom Plot: Scores ---
    ax2.plot(scores_padded, color="blue", linewidth=1.2, label="Anomaly Score")
    ax2.axhline(
        threshold,
        color="red",
        linestyle="--",
        label=f"Threshold = {threshold:.3f}",
    )
    ax2.set_xlabel("Time Step")
    ax2.set_ylabel("Score")
    ax2.legend(loc="upper left")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    _save_or_show(fig, save_path)
