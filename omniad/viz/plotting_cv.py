from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

from omniad.viz.utils import _check_viz_deps, _save_or_show


def plot_anomaly_heatmap(
    image: npt.NDArray[Any],
    anomaly_map: npt.NDArray[Any],
    title: str = "Anomaly Heatmap",
    cmap: str = "hot",
    alpha: float = 0.5,
    save_path: str | None = None,
    **kwargs: Any,
) -> None:
    """
    Overlay anomaly heatmap on an image.

    Parameters
    ----------
    image : npt.NDArray[Any]
        Original image. Accepts:
        - (H, W) grayscale
        - (H, W, C) RGB/BGR
        - (C, H, W) CHW format (auto-converted)
    anomaly_map : npt.NDArray[Any] of shape (H, W)
        Per-pixel anomaly scores from predict_map().
    title : str, default="Anomaly Heatmap"
        Plot title.
    cmap : str, default="hot"
        Colormap for the heatmap overlay.
    alpha : float, default=0.5
        Transparency of the heatmap overlay (0=invisible, 1=opaque).
    save_path : str or None
        If provided, save figure to this path instead of showing.
    """
    _check_viz_deps()
    import matplotlib.pyplot as plt

    # Handle CHW -> HWC
    img = image.copy()
    if img.ndim == 3 and img.shape[0] in (1, 3):
        img = np.transpose(img, (1, 2, 0))
    if img.ndim == 3 and img.shape[2] == 1:
        img = img.squeeze(2)

    if img.dtype == np.uint8:
        img = img.astype(np.float32) / 255.0
    elif img.max() > 1.0:
        img = img / img.max()

    fig, ax = plt.subplots(1, 1, figsize=(6, 6))

    if img.ndim == 2:
        ax.imshow(img, cmap="gray")
    else:
        ax.imshow(img)

    im = ax.imshow(anomaly_map, cmap=cmap, alpha=alpha)
    fig.colorbar(im, ax=ax, label="Anomaly Score")
    ax.set_title(title)
    ax.axis("off")

    _save_or_show(fig, save_path)
