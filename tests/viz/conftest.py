from __future__ import annotations

import matplotlib
import pytest


@pytest.fixture(scope="session", autouse=True)
def headless_matplotlib() -> None:
    """Use a non-interactive backend for visualization tests."""
    matplotlib.use("Agg")
