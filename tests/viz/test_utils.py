from __future__ import annotations

from unittest.mock import MagicMock

import pytest

import omniad.viz.utils as viz_utils


def test_check_viz_deps_requires_matplotlib(monkeypatch) -> None:
    """Matplotlib is required by every visualization."""
    monkeypatch.setattr(viz_utils, "plt", None)

    with pytest.raises(ImportError, match="matplotlib"):
        viz_utils._check_viz_deps()


def test_check_viz_deps_requires_seaborn_when_requested(monkeypatch) -> None:
    """Seaborn is required only by seaborn-based visualizations."""
    monkeypatch.setattr(viz_utils, "sns", None)

    with pytest.raises(ImportError, match="seaborn"):
        viz_utils._check_viz_deps(require_seaborn=True)


def test_check_viz_deps_allows_missing_seaborn_by_default(monkeypatch) -> None:
    """Matplotlib-only visualizations do not require seaborn."""
    monkeypatch.setattr(viz_utils, "sns", None)

    viz_utils._check_viz_deps()


def test_save_or_show_saves_and_closes_figure(tmp_path, monkeypatch) -> None:
    """save_path writes the figure and closes it."""
    fig = MagicMock()
    path = tmp_path / "figure.png"
    close = MagicMock()
    monkeypatch.setattr(viz_utils.plt, "close", close)

    viz_utils._save_or_show(fig, str(path))

    fig.savefig.assert_called_once_with(str(path), bbox_inches="tight")
    close.assert_called_once_with(fig)


def test_save_or_show_displays_and_closes_without_path(monkeypatch) -> None:
    """Missing save_path uses interactive display and closes the figure."""
    fig = MagicMock()
    show = MagicMock()
    close = MagicMock()
    monkeypatch.setattr(viz_utils.plt, "show", show)
    monkeypatch.setattr(viz_utils.plt, "close", close)

    viz_utils._save_or_show(fig)

    show.assert_called_once_with()
    close.assert_called_once_with(fig)
