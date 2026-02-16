from __future__ import annotations

import inspect
from typing import Any, cast

import numpy as np
from sklearn.utils.validation import check_array

from omniad.core.exceptions import DataFormatError


def _is_pandas_object(X: Any) -> bool:
    """
    Check if X is a pandas DataFrame/Series without importing pandas.

    Parameters
    ----------
    X : Any
        Input object.

    Returns
    -------
    is_pandas : bool
        True if X looks like a pandas DataFrame/Series.
    """
    module = getattr(type(X), "__module__", "")
    return module.startswith("pandas.")


def _to_numpy(X: Any) -> np.ndarray[Any, Any]:
    """
    Convert supported inputs to a numpy ndarray.

    Parameters
    ----------
    X : Any
        Input object.

    Returns
    -------
    arr : np.ndarray
        Converted numpy array.

    Raises
    ------
    DataFormatError
        If conversion fails.
    """
    if _is_pandas_object(X):
        if hasattr(X, "to_numpy"):
            return cast("np.ndarray[Any, Any]", np.asarray(X.to_numpy()))

        if hasattr(X, "values"):
            return cast("np.ndarray[Any, Any]", np.asarray(X))

        raise DataFormatError(
            "Input looks like a pandas object but cannot be converted to numpy."
        )

    if isinstance(X, np.ndarray):
        return X

    return cast("np.ndarray[Any, Any]", np.asarray(X))


def _check_array_compat(X: np.ndarray[Any, Any], **kwargs: Any) -> np.ndarray[Any, Any]:
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
        return cast("np.ndarray[Any, Any]", check_array(X, **common_args))

    # Old version
    return cast("np.ndarray[Any, Any]", check_array(X, **common_args))


def validate_input(X: Any, **kwargs: Any) -> np.ndarray[Any, Any]:
    """
    Validate input data for tabular anomaly detectors.

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
    X_valid : np.ndarray
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
