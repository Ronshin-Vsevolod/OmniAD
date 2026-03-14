from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

from omniad.viz.utils import _check_viz_deps, _save_or_show

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
except ImportError:
    pass


def plot_anomaly_scores(
    scores: npt.NDArray[Any],
    threshold: float | None = None,
    title: str = "Anomaly Scores",
    save_path: str | None = None,
) -> None:
    """
    Plot histogram of anomaly scores with optional threshold line.

    Parameters
    ----------
    scores : npt.NDArray[Any]
        1D array of anomaly scores.
    threshold : float, optional
        Threshold value to draw a vertical line.
    title : str, default="Anomaly Scores"
        Plot title.
    save_path : str, optional
        Path to save the image. If None, shows interactively.
    """
    _check_viz_deps()
    fig, ax = plt.subplots(figsize=(8, 5))

    sns.histplot(scores, bins=30, kde=True, ax=ax)

    if threshold is not None:
        ax.axvline(
            threshold, color="red", linestyle="--", label=f"Threshold = {threshold:.3f}"
        )
        ax.legend()

    ax.set_title(title)
    ax.set_xlabel("Score")
    ax.set_ylabel("Frequency")

    _save_or_show(fig, save_path)


def plot_scatter_2d(
    X: npt.NDArray[Any],
    labels: npt.NDArray[Any] | None = None,
    title: str = "2D Projection",
    save_path: str | None = None,
) -> None:
    """
    Plot 2D scatter of points. Applies PCA if n_features > 2.

    Parameters
    ----------
    X : npt.NDArray[Any]
        Input data. If n_features > 2, PCA is applied.
    labels : npt.NDArray[Any], optional
        Binary labels (0=Normal, 1=Anomaly).
    title : str, default="2D Projection"
        Plot title.
    save_path : str, optional
        Path to save the image.
    """
    _check_viz_deps()

    # Dimensionality reduction
    X_2d = X
    if X.shape[1] > 2:
        try:
            from sklearn.decomposition import PCA
        except ImportError as e:
            raise ImportError(
                "scikit-learn is required for PCA visualization. "
                "Install it with: pip install scikit-learn"
            ) from e
        pca = PCA(n_components=2, random_state=42)
        X_2d = pca.fit_transform(X)
        explained_var = pca.explained_variance_ratio_.sum()
        title += f" (PCA, exp. var: {explained_var:.2%})"
    elif X.shape[1] == 1:
        X_2d = np.column_stack([X, np.zeros_like(X)])

    fig, ax = plt.subplots(figsize=(8, 6))

    if labels is not None:
        normal = labels == 0
        anomaly = labels == 1

        ax.scatter(
            X_2d[normal, 0],
            X_2d[normal, 1],
            c="blue",
            label="Normal",
            alpha=0.6,
            edgecolors="w",
            linewidth=0.5,
        )
        ax.scatter(
            X_2d[anomaly, 0],
            X_2d[anomaly, 1],
            c="red",
            label="Anomaly",
            alpha=0.8,
            edgecolors="w",
            linewidth=0.5,
        )
        ax.legend()
    else:
        ax.scatter(
            X_2d[:, 0], X_2d[:, 1], c="blue", alpha=0.6, edgecolors="w", linewidth=0.5
        )

    ax.set_title(title)
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    ax.grid(True, linestyle="--", alpha=0.7)

    _save_or_show(fig, save_path)
