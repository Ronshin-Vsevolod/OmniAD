"""
Layer 3 tests for the declarative validation rule pipeline.

Exercised through the public validate_input() entry point (same path
every adapter uses via get_validation_rules()), not by calling the
private _rule_* functions directly — insulated from internal renames.
"""
from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

from omniad.core.exceptions import DataFormatError
from omniad.utils.validation import (
    register_validation_rule,
    validate_image,
    validate_input,
    validate_text,
)

# --- to_numpy ---


def test_to_numpy_converts_list() -> None:
    result = validate_input([1, 2, 3], {"to_numpy"})
    assert isinstance(result, np.ndarray)
    np.testing.assert_array_equal(result, [1, 2, 3])


def test_to_numpy_converts_pandas_dataframe() -> None:
    pd = pytest.importorskip("pandas")
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    result = validate_input(df, {"to_numpy"})
    assert isinstance(result, np.ndarray)
    np.testing.assert_array_equal(result, [[1, 3], [2, 4]])


def test_to_numpy_passes_sparse_through_unchanged() -> None:
    X = sp.csr_matrix(np.eye(3))
    result = validate_input(X, {"to_numpy"})
    assert sp.issparse(result)


# --- reject_sparse ---


def test_reject_sparse_raises_on_sparse_input() -> None:
    with pytest.raises(DataFormatError):
        validate_input(sp.csr_matrix(np.eye(3)), {"reject_sparse"})


def test_reject_sparse_passes_dense_through() -> None:
    X = np.eye(3)
    np.testing.assert_array_equal(validate_input(X, {"reject_sparse"}), X)


# --- require_2d ---


def test_require_2d_reshapes_1d_array() -> None:
    result = validate_input(np.array([1.0, 2.0, 3.0]), {"require_2d"})
    assert result.shape == (3, 1)


def test_require_2d_passes_2d_array_through() -> None:
    result = validate_input(np.zeros((4, 2)), {"require_2d"})
    assert result.shape == (4, 2)


def test_require_2d_rejects_3d_array() -> None:
    with pytest.raises(DataFormatError):
        validate_input(np.zeros((2, 3, 4)), {"require_2d"})


def test_require_2d_passes_sparse_through_unchanged() -> None:
    result = validate_input(sp.csr_matrix(np.eye(3)), {"require_2d"})
    assert sp.issparse(result)


# --- require_single_row ---


def test_require_single_row_reshapes_1d_vector() -> None:
    result = validate_input(np.array([1.0, 2.0]), {"require_single_row"})
    assert result.shape == (1, 2)


def test_require_single_row_accepts_single_row_2d() -> None:
    result = validate_input(np.array([[1.0, 2.0]]), {"require_single_row"})
    assert result.shape == (1, 2)


def test_require_single_row_rejects_multi_row_2d() -> None:
    with pytest.raises(DataFormatError):
        validate_input(np.zeros((2, 3)), {"require_single_row"})


def test_require_single_row_accepts_single_row_sparse() -> None:
    X = sp.csr_matrix(np.array([[1.0, 2.0]]))
    result = validate_input(X, {"require_single_row"})
    assert result.shape == (1, 2)


def test_require_single_row_rejects_multi_row_sparse() -> None:
    with pytest.raises(DataFormatError):
        validate_input(sp.csr_matrix(np.zeros((2, 3))), {"require_single_row"})


# --- reject_nan ---


def test_reject_nan_raises_on_nan() -> None:
    with pytest.raises(DataFormatError):
        validate_input(np.array([1.0, np.nan, 3.0]), {"reject_nan"})


def test_reject_nan_raises_on_inf() -> None:
    with pytest.raises(DataFormatError):
        validate_input(np.array([1.0, np.inf, 3.0]), {"reject_nan"})


def test_reject_nan_passes_finite_values() -> None:
    X = np.array([1.0, 2.0, 3.0])
    np.testing.assert_array_equal(validate_input(X, {"reject_nan"}), X)


def test_reject_nan_skips_sparse_input() -> None:
    result = validate_input(sp.csr_matrix(np.eye(3)), {"reject_nan"})
    assert sp.issparse(result)


def test_reject_nan_checks_sparse_matrix_data() -> None:
    X = sp.csr_matrix([[1.0, np.nan], [2.0, 3.0]])
    with pytest.raises(DataFormatError):
        validate_input(X, {"reject_nan"})


# --- require_float32 ---


def test_require_float32_casts_int_array() -> None:
    result = validate_input(np.array([1, 2, 3]), {"require_float32"})
    assert result.dtype == np.float32


def test_require_float32_is_a_noop_for_existing_float32() -> None:
    X = np.array([1.0, 2.0], dtype=np.float32)
    result = validate_input(X, {"require_float32"})
    assert result is X  # identity — no wasted copy


def test_require_float32_raises_on_uncastable_input() -> None:
    with pytest.raises(DataFormatError):
        validate_input(["not", "castable"], {"require_float32"})


# --- domain_text ---


def test_domain_text_converts_numpy_string_array_to_list() -> None:
    assert validate_input(np.array(["a", "b"]), {"domain_text"}) == ["a", "b"]


def test_domain_text_rejects_numpy_array_of_wrong_dtype() -> None:
    with pytest.raises(DataFormatError):
        validate_input(np.array([1, 2, 3]), {"domain_text"})


def test_domain_text_rejects_non_list_input() -> None:
    with pytest.raises(DataFormatError):
        validate_input("just a string, not a list", {"domain_text"})


def test_domain_text_rejects_empty_list() -> None:
    with pytest.raises(DataFormatError):
        validate_input([], {"domain_text"})


def test_domain_text_rejects_non_string_elements() -> None:
    with pytest.raises(DataFormatError, match="Non-string"):
        validate_input(["ok", 123, "also ok"], {"domain_text"})


def test_domain_text_rejects_whitespace_only_elements() -> None:
    with pytest.raises(DataFormatError, match="empty"):
        validate_input(["ok", "   ", "also ok"], {"domain_text"})


def test_domain_text_passes_valid_list_through() -> None:
    texts = ["hello world", "another sentence"]
    assert validate_input(texts, {"domain_text"}) == texts


# --- domain_image ---


def test_domain_image_rejects_wrong_ndim() -> None:
    with pytest.raises(DataFormatError):
        validate_input(np.zeros((10, 10)), {"domain_image"})


def test_domain_image_rejects_invalid_channel_count() -> None:
    with pytest.raises(DataFormatError):
        validate_input(np.zeros((2, 4, 8, 8)), {"domain_image"})


def test_domain_image_accepts_grayscale_and_rgb_channels() -> None:
    for channels in (1, 3):
        X = np.zeros((2, channels, 8, 8), dtype=np.float32)
        result = validate_input(X, {"domain_image"})
        assert result.shape == X.shape


def test_domain_image_normalizes_uint8_to_float32_0_1_range() -> None:
    X = np.full((1, 3, 4, 4), 255, dtype=np.uint8)
    result = validate_input(X, {"domain_image"})
    assert result.dtype == np.float32
    np.testing.assert_allclose(result, 1.0)


def test_domain_image_leaves_float_input_unchanged() -> None:
    X = np.random.default_rng(0).random((1, 3, 4, 4)).astype(np.float32)
    np.testing.assert_array_equal(validate_input(X, {"domain_image"}), X)


# --- orchestration: order, unknown rules, custom rule registration ---


def test_validate_input_rejects_unknown_rule_name() -> None:
    with pytest.raises(ValueError, match="Unknown validation rules"):
        validate_input(np.zeros((2, 2)), {"not_a_real_rule"})


def test_validate_input_applies_rules_in_registry_order_not_set_order() -> None:
    """`to_numpy` must run before `require_2d` regardless of how the
    caller's `set` happens to iterate — this is the entire point of
    the declarative rule system (adapters declare *what*, the
    validator owns *order*, per _VALIDATION_ORDER)."""
    result = validate_input([1, 2, 3], {"require_2d", "to_numpy"})
    assert isinstance(result, np.ndarray)
    assert result.shape == (3, 1)


def test_register_validation_rule_runs_and_is_positioned_after_target() -> None:
    calls = []

    def _record_and_pass(X: np.ndarray) -> np.ndarray:
        calls.append(X.shape)
        return X

    register_validation_rule("record_test_rule", _record_and_pass, after="to_numpy")

    result = validate_input([1, 2, 3], {"to_numpy", "record_test_rule", "require_2d"})

    assert calls == [(3,)]  # ran after to_numpy, before require_2d's reshape
    assert result.shape == (3, 1)


# --- shortcuts ---


def test_validate_text_shortcut_uses_text_validation() -> None:
    assert validate_text(["hello"]) == ["hello"]


def test_validate_image_shortcut_uses_image_validation() -> None:
    X = np.zeros((2, 1, 4, 4), dtype=np.float32)
    assert validate_image(X) is X
