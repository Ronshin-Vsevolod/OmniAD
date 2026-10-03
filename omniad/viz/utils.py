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


def _check_viz_deps(require_seaborn: bool = False) -> None:
    """
    Check visualization dependencies.

    Parameters
    ----------
    require_seaborn : bool, default=False
        Whether seaborn is required by the caller.

    Raises
    ------
    ImportError
        If a required visualization dependency is missing.
    """
    if plt is None:
        raise ImportError(
            "matplotlib is required for visualization. "
            "Install it with: pip install omniad[viz]"
        )
    if require_seaborn and sns is None:
        raise ImportError(
            "seaborn is required for this visualization. "
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
