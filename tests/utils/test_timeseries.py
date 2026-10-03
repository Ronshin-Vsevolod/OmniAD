"""
Layer 3 tests for time-series window construction.
"""
import numpy as np
import pytest

from omniad.core.exceptions import DataFormatError
from omniad.utils.timeseries import create_windows


def test_create_windows_shape_and_values() -> None:
    """Sliding windows preserve ordering and feature dimensions."""
    X = np.arange(12).reshape(6, 2)

    windows = create_windows(X, window_size=3)

    assert windows.shape == (4, 3, 2)
    np.testing.assert_array_equal(windows[0], X[:3])
    np.testing.assert_array_equal(windows[-1], X[-3:])


def test_create_windows_respects_step() -> None:
    """step controls the stride between consecutive windows."""
    X = np.arange(16).reshape(8, 2)

    windows = create_windows(X, window_size=3, step=2)

    np.testing.assert_array_equal(windows[0], X[0:3])
    np.testing.assert_array_equal(windows[1], X[2:5])
    np.testing.assert_array_equal(windows[2], X[4:7])


def test_create_windows_rejects_short_input() -> None:
    """Window size cannot exceed the available time series."""
    X = np.zeros((2, 3))

    with pytest.raises(DataFormatError, match="smaller than window_size"):
        create_windows(X, window_size=3)


def test_create_windows_returns_independent_array() -> None:
    """Returned windows do not share writable memory with the source."""
    X = np.arange(12).reshape(6, 2)
    windows = create_windows(X, window_size=3)

    before = windows.copy()
    X[:] = -1

    np.testing.assert_array_equal(windows, before)
