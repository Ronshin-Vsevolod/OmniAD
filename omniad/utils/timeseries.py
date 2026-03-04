from __future__ import annotations

from typing import Any, cast

import numpy as np
import numpy.typing as npt

from omniad.core.exceptions import DataFormatError


def create_windows(
    data: npt.NDArray[Any], window_size: int, step: int = 1
) -> npt.NDArray[Any]:
    """
    Slice a 2D time series into 3D windows.

    Parameters
    ----------
    data : np.ndarray, shape (n_time_steps, n_features)
        The input time series.
    window_size : int
        The length of the look-back window.
    step : int, default=1
        Stride between windows.

    Returns
    -------
    windows : np.ndarray, shape (n_windows, window_size, n_features)
        The sliding windows.

    Raises
    ------
    DataFormatError
        If data length is smaller than window_size.
    """
    n_samples, n_features = data.shape

    if n_samples < window_size:
        raise DataFormatError(
            f"Data length ({n_samples}) is smaller than window_size ({window_size})."
        )

    num_windows = (n_samples - window_size) // step + 1

    # Fast numpy striding trick
    shape = (num_windows, window_size, n_features)
    strides = (
        data.strides[0] * step,
        data.strides[0],
        data.strides[1],
    )

    windows = np.lib.stride_tricks.as_strided(
        data, shape=shape, strides=strides, writeable=False
    )

    # Force copy to avoid memory issues with strided views in PyTorch later
    return cast("npt.NDArray[Any]", windows.copy())
