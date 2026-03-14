"""
Shared utilities for visualization module.
"""
from __future__ import annotations

from typing import Any

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

try:
    import seaborn as sns
except ImportError:
    sns = None


def _check_viz_deps() -> None:
    """
    Check if matplotlib and seaborn are installed.

    Raises
    ------
    ImportError
        If dependencies are missing.
    """
    if plt is None:
        raise ImportError(
            "matplotlib is required for visualization. "
            "Install it with: pip install omniad[viz]"
        )
    if sns is None:
        raise ImportError(
            "seaborn is required for visualization. "
            "Install it with: pip install omniad[viz]"
        )


def _save_or_show(fig: Any, path: str | None = None) -> None:
    """
    Save figure to file or show it interactively.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The figure object.
    path : str, optional
        Path to save the figure. If None, plt.show() is called.
    """
    _check_viz_deps()  # Ensures plt is available
    if path is not None:
        fig.savefig(path, bbox_inches="tight")
    else:
        plt.show()
    plt.close(fig)
