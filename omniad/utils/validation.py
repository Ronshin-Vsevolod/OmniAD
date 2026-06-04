"""
Rule-based input validation for OmniAD detectors.

Design
------
Adapters *declare* what they need via get_validation_rules() -> set[str].
The validator *executes* rules in the globally defined safe order
(_VALIDATION_ORDER). Adapters never control execution order.

Adding a new rule
-----------------
1. Write _rule_<name>(X: Any) -> Any.
2. Register it in _VALIDATION_REGISTRY.
3. Insert it at the correct position in _VALIDATION_ORDER.
4. Add the key to the relevant adapter's get_validation_rules().
"""

from __future__ import annotations

import inspect
import logging
from typing import Any, Callable, cast

import numpy as np
import numpy.typing as npt
from sklearn.utils.validation import check_array

from omniad.core.exceptions import DataFormatError

logger = logging.getLogger(__name__)


#  --- Internal helpers (preserved from original) ---


def _is_pandas_object(X: Any) -> bool:
    """Check if X is a pandas DataFrame/Series without importing pandas."""
    module = getattr(type(X), "__module__", "")
    return module.startswith("pandas.")


def _to_numpy(X: Any) -> npt.NDArray[Any]:
    """
    Convert supported inputs to numpy, passing sparse through unchanged.

    Parameters
    ----------
    X : Any
        Input object: numpy array, list, pandas DataFrame/Series,
        or scipy sparse matrix.

    Returns
    -------
    arr : npt.NDArray[Any]
        Converted numpy array, or original sparse matrix.

    Raises
    ------
    DataFormatError
        If conversion fails or pandas object cannot be converted.
    """
    import scipy.sparse as sp

    if sp.issparse(X):
        return X

    if _is_pandas_object(X):
        if hasattr(X, "to_numpy"):
            return cast("npt.NDArray[Any]", np.asarray(X.to_numpy()))
        if hasattr(X, "values"):
            return cast("npt.NDArray[Any]", np.asarray(X.values))
        raise DataFormatError(
            "Input looks like a pandas object but cannot be converted to numpy."
        )

    if isinstance(X, np.ndarray):
        return cast("npt.NDArray[Any]", X)

    return cast("npt.NDArray[Any]", np.asarray(X))


def _check_array_compat(
    X: npt.NDArray[Any],
    accept_sparse: bool = False,
    **kwargs: Any,
) -> npt.NDArray[Any]:
    """
    Call sklearn check_array in a version-compatible way.

    Handles the rename of force_all_finite → ensure_all_finite
    in scikit-learn >= 1.6.
    """
    sig = inspect.signature(check_array)
    params = sig.parameters

    common_args: dict[str, Any] = {
        "accept_sparse": accept_sparse,
        "ensure_2d": True,
        "dtype": None,  # preserve original dtype (float32/64)
    }
    common_args.update(kwargs)

    if "ensure_all_finite" in params:
        if "force_all_finite" in common_args:
            common_args["ensure_all_finite"] = common_args.pop("force_all_finite")

    return cast("npt.NDArray[Any]", check_array(X, **common_args))


#  --- Rules ---


def _rule_to_numpy(X: Any) -> Any:
    """
    Convert input to numpy array.

    Pass-through for sparse matrices (sparse check happens separately).
    Handles: pandas DataFrame/Series, lists, plain arrays.
    """
    try:
        return _to_numpy(X)
    except DataFormatError:
        raise
    except Exception as e:
        raise DataFormatError(f"Failed to convert input to array: {e}") from e


def _rule_reject_sparse(X: Any) -> Any:
    """Raise DataFormatError if X is a sparse matrix."""
    import scipy.sparse as sp

    if sp.issparse(X):
        raise DataFormatError(
            "Sparse matrix passed but this detector requires dense input. "
            "Use X.toarray() or choose a detector that supports sparse data."
        )
    return X


def _rule_require_2d(X: Any) -> Any:
    """
    Ensure X is 2D.

    Reshapes 1D arrays to (N, 1) for univariate convenience.
    Raises for ndim > 2.
    Pass-through for sparse (shape is always 2D for sparse).
    """
    import scipy.sparse as sp

    if sp.issparse(X):
        return X
    if X.ndim == 1:
        return X.reshape(-1, 1)
    if X.ndim != 2:
        raise DataFormatError(
            f"Expected 2D array (n_samples, n_features), got {X.ndim}D. "
            f"Shape: {X.shape}."
        )
    return X


def _rule_reject_nan(X: Any) -> Any:
    """
    Raise if X contains NaN or Inf values.

    Uses _check_array_compat to leverage sklearn's version-compatible
    finite check. Skips the check for sparse matrices.
    """
    import scipy.sparse as sp

    if sp.issparse(X):
        return X

    arr = np.asarray(X)
    if not np.isfinite(arr).all():
        raise DataFormatError(
            "Input contains NaN or infinite values. "
            "Clean your data or use a detector that supports NaN."
        )
    return X


def _rule_require_float32(X: Any) -> npt.NDArray[Any]:
    """Cast to float32 when needed."""
    try:
        if isinstance(X, np.ndarray) and X.dtype == np.float32:
            return X

        return X.astype(np.float32)

    except (AttributeError, ValueError) as e:
        raise DataFormatError(
            f"Cannot cast input to float32: {e}. " f"Got type {type(X).__name__}."
        ) from e


def _rule_domain_text(X: Any) -> list[str]:
    """
    Validate text input: non-empty list of non-empty strings.

    Also accepts numpy arrays of string dtype (e.g. from pandas .values).
    """
    if isinstance(X, np.ndarray):
        if X.dtype.kind not in ("U", "S", "O"):
            raise DataFormatError(f"Expected array of strings, got dtype={X.dtype}.")
        X = X.ravel().tolist()

    if not isinstance(X, list):
        raise DataFormatError(
            f"Text input must be a list of strings, got {type(X).__name__}."
        )
    if not X:
        raise DataFormatError("Input list is empty.")

    non_strings = [i for i, s in enumerate(X) if not isinstance(s, str)]
    if non_strings:
        raise DataFormatError(
            f"All elements must be strings. "
            f"Non-string elements at indices: {non_strings[:5]}."
        )
    empty = [i for i, s in enumerate(X) if not s.strip()]
    if empty:
        raise DataFormatError(
            f"Input contains empty/whitespace-only strings " f"at indices: {empty[:5]}."
        )
    return X


def _rule_domain_image(X: Any) -> npt.NDArray[Any]:
    """
    Validate image input: (N, C, H, W) with C in {1, 3}.

    Auto-normalizes uint8 [0, 255] to float32 [0, 1].
    """
    if not isinstance(X, np.ndarray):
        X = np.asarray(X)

    if X.ndim != 4:
        raise DataFormatError(f"Image input must be 4D (N, C, H, W), got {X.ndim}D.")
    channels = X.shape[1]
    if channels not in (1, 3):
        raise DataFormatError(
            f"Expected 1 or 3 channels at dim=1, got {channels}. "
            f"Shape: {X.shape}. Expected (N, C, H, W)."
        )
    if X.dtype == np.uint8:
        return X.astype(np.float32) / 255.0
    return X


#  --- Registry & execution order ---

_VALIDATION_REGISTRY: dict[str, Callable[[Any], Any]] = {
    "to_numpy": _rule_to_numpy,
    "reject_sparse": _rule_reject_sparse,
    "require_2d": _rule_require_2d,
    "reject_nan": _rule_reject_nan,
    "require_float32": _rule_require_float32,
    "domain_text": _rule_domain_text,
    "domain_image": _rule_domain_image,
}

# Execution order is owned by the validator, not the adapters.
# Rules absent from the adapter's declared set are skipped.
_VALIDATION_ORDER: tuple[str, ...] = (
    "domain_text",
    "domain_image",
    "to_numpy",
    "reject_sparse",
    "require_2d",
    "reject_nan",
    "require_float32",
)


def register_validation_rule(
    name: str,
    fn: Callable[[Any], Any],
    *,
    after: str | None = None,
) -> None:
    """
    Register a custom validation rule globally.

    Parameters
    ----------
    name : str
        Rule identifier for use in get_validation_rules().
    fn : callable
        fn(X: Any) -> Any — returns transformed X or raises
        DataFormatError.
    after : str or None
        Insert into _VALIDATION_ORDER immediately after this rule.
        If None or not found, appended at the end.

    Examples
    --------
    >>> from omniad.utils.validation import register_validation_rule
    >>> from omniad.core.exceptions import DataFormatError
    >>> def _rule_positive(X):
    ...     if (X < 0).any():
    ...         raise DataFormatError("Negative values not allowed.")
    ...     return X
    >>> register_validation_rule("require_positive", _rule_positive, after="reject_nan")
    """
    global _VALIDATION_ORDER

    _VALIDATION_REGISTRY[name] = fn

    if after is not None and after in _VALIDATION_ORDER:
        idx = _VALIDATION_ORDER.index(after)
        _VALIDATION_ORDER = (
            _VALIDATION_ORDER[: idx + 1] + (name,) + _VALIDATION_ORDER[idx + 1 :]
        )
    else:
        _VALIDATION_ORDER = _VALIDATION_ORDER + (name,)


#  --- Entry point ---


def validate_input(X: Any, rules: set[str]) -> Any:
    """
    Apply declared validation rules to X in the globally defined safe order.

    Parameters
    ----------
    X : Any
        Raw input data.
    rules : set[str]
        Rule names declared by the detector via get_validation_rules().
        Execution order is determined by _VALIDATION_ORDER.

    Returns
    -------
    X : Any
        Validated (and possibly transformed) data.

    Raises
    ------
    DataFormatError
        On any rule violation.
    ValueError
        If rules contains unknown rule names.
    """
    unknown = rules - set(_VALIDATION_REGISTRY.keys())
    if unknown:
        raise ValueError(
            f"Unknown validation rules: {unknown}. "
            f"Available: {sorted(_VALIDATION_REGISTRY.keys())}"
        )

    for rule in _VALIDATION_ORDER:
        if rule in rules:
            X = _VALIDATION_REGISTRY[rule](X)

    return X


#  --- Back-compat shortcuts ---


def validate_text(X: Any) -> list[str]:
    """Validate text input. Equivalent to validate_input(X, {"domain_text"})."""
    result = validate_input(X, {"domain_text"})
    return cast("list[str]", result)


def validate_image(X: Any) -> npt.NDArray[Any]:
    """Validate image input. Equivalent to validate_input(X, {"domain_image"})."""
    return cast("npt.NDArray[Any]", validate_input(X, {"domain_image"}))
