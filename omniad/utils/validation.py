from __future__ import annotations

import inspect
from typing import Any, cast

import numpy as np
import numpy.typing as npt
from sklearn.utils.validation import check_array

from omniad.core.exceptions import DataFormatError


def _is_pandas_object(X: Any) -> bool:
    """Check if X is a pandas DataFrame/Series without importing pandas."""
    module = getattr(type(X), "__module__", "")
    return module.startswith("pandas.")


def _to_numpy(X: Any) -> npt.NDArray[Any]:
    """
    Convert supported inputs to a numpy ndarray.

    Parameters
    ----------
    X : Any
        Input object.

    Returns
    -------
    arr : npt.NDArray[Any]
        Converted numpy array.

    Raises
    ------
    DataFormatError
        If conversion fails.
    """
    if _is_pandas_object(X):
        if hasattr(X, "to_numpy"):
            return cast("npt.NDArray[Any]", np.asarray(X.to_numpy()))

        if hasattr(X, "values"):
            return cast("npt.NDArray[Any]", np.asarray(X))

        raise DataFormatError(
            "Input looks like a pandas object but cannot be converted to numpy."
        )

    if isinstance(X, np.ndarray):
        return X

    return cast("npt.NDArray[Any]", np.asarray(X))


def _check_array_compat(X: npt.NDArray[Any], **kwargs: Any) -> npt.NDArray[Any]:
    """
    Call sklearn check_array in a version-compatible way.

    Handles the rename of 'force_all_finite' to 'ensure_all_finite'
    in newer scikit-learn versions.
    """
    sig = inspect.signature(check_array)
    params = sig.parameters

    # Default logic: we want to ban NaNs and Infs for tabular data
    common_args = {
        "accept_sparse": False,
        "ensure_2d": True,
        "dtype": None,  # Keep original dtype (float32/64)
    }
    common_args.update(kwargs)

    # Scikit-learn version compatibility
    if "ensure_all_finite" in params:
        if "force_all_finite" in common_args:
            common_args["ensure_all_finite"] = common_args.pop("force_all_finite")
        return cast("npt.NDArray[Any]", check_array(X, **common_args))

    # Old version
    return cast("npt.NDArray[Any]", check_array(X, **common_args))


def validate_input(X: Any, **kwargs: Any) -> npt.NDArray[Any]:
    """
    Validate input data for tabular and timeseries anomaly detectors.

    Converts input to numpy array, checks dimensions, and ensures no NaNs/Infs exist.

    Parameters
    ----------
    X : Any
        Input data. Supported formats:
        - numpy.ndarray
        - list (including list of lists)
        - pandas.DataFrame / Series
    **kwargs : Any
        Additional keyword arguments forwarded to sklearn's check_array.

    Returns
    -------
    X_valid : npt.NDArray[Any]
        Validated 2D array.

    Raises
    ------
    DataFormatError
        If input contains NaN/Inf, has invalid dimensionality,
        or cannot be converted.
    """
    try:
        X_arr = _to_numpy(X)
    except Exception as e:
        raise DataFormatError(f"Failed to convert input to numpy array: {e}") from e

    # We explicitly handle 1D arrays by reshaping them to (n_samples, 1)
    if X_arr.ndim == 1:
        X_arr = X_arr.reshape(-1, 1)

    try:
        X_arr = _check_array_compat(X_arr, **kwargs)
    except ValueError as e:
        raise DataFormatError(f"Input validation failed: {str(e)}") from e

    return X_arr


def validate_text(X: Any) -> list[str]:
    """
    Validate input data for text-based anomaly detectors.

    Parameters
    ----------
    X : Any
        Input data. Expected: list of non-empty strings.

    Returns
    -------
    X_valid : list[str]
        Validated list of strings.

    Raises
    ------
    DataFormatError
        If input is not a valid list of strings.
    """
    # Support npt.NDArray[Any] of strings (e.g. from pandas .values)
    if isinstance(X, np.ndarray):
        if X.dtype.kind not in ("U", "S", "O"):
            raise DataFormatError(f"Expected array of strings, got dtype={X.dtype}")
        X = X.ravel().tolist()

    if not isinstance(X, list):
        raise DataFormatError(
            f"Text input must be a list of strings, got {type(X).__name__}"
        )

    if len(X) == 0:
        raise DataFormatError("Input list is empty.")

    non_strings = [i for i, s in enumerate(X) if not isinstance(s, str)]
    if non_strings:
        raise DataFormatError(
            f"All elements must be strings. "
            f"Non-string elements at indices: {non_strings[:5]}"
        )

    empty = [i for i, s in enumerate(X) if len(s.strip()) == 0]
    if empty:
        raise DataFormatError(
            f"Input contains empty/whitespace-only strings at indices: {empty[:5]}"
        )

    return X


def validate_image(X: Any) -> npt.NDArray[Any]:
    """
    Validate image input for CV anomaly detectors.

    Expects (N, C, H, W) with C in {1, 3}.
    Auto-normalizes uint8 [0, 255] to float32 [0, 1].

    Parameters
    ----------
    X : Any
        Expected: (N, C, H, W), C in {1, 3}, values in [0, 1].

    Returns
    -------
    X_valid : npt.NDArray[Any] of shape (N, C, H, W), dtype float32.
    """
    if isinstance(X, np.ndarray):
        X_arr = X
    else:
        X_arr = np.asarray(X)

    if X_arr.ndim != 4:
        raise DataFormatError(f"Image input must be 4D (N, C, H, W), got {X_arr.ndim}D")

    channels = X_arr.shape[1]
    if channels not in (1, 3):
        raise DataFormatError(
            f"Expected 1 or 3 channels at dim=1, got {channels}. "
            f"Shape: {X_arr.shape}. Expected (N, C, H, W)."
        )

    if X_arr.dtype == np.uint8:
        return X_arr.astype(np.float32) / 255.0

    return X_arr.astype(np.float32)
