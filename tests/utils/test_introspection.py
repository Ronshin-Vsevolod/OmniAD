"""
Layer 3 tests for shared introspection helpers used by both the .pyi
stub generator and omniad.describe() / composition adapters
class-level get_capabilities() (see BertDetectorAdapter/TfidfDetectorAdapter).
"""
from __future__ import annotations

import pytest

from omniad.core.exceptions import ConfigError
from omniad.utils.introspection import (
    extract_init_params,
    resolve_default_inner_detector,
    resolve_delegated_capabilities,
    resolve_delegated_capabilities_dict,
)


class _NoDetectorParam:
    """Adapter stand-in with no `detector` parameter at all."""

    def __init__(self, contamination: float = 0.1) -> None:
        pass


class _NoDetectorDefault:
    """Adapter stand-in whose `detector` has no default at all."""

    def __init__(self, detector: str, contamination: float = 0.1) -> None:
        pass


class _NonStringDetectorDefault:
    """Adapter stand-in whose `detector` default is not a string."""

    def __init__(self, detector: int = 5, contamination: float = 0.1) -> None:
        pass


def test_extract_init_params_reports_required_and_defaults() -> None:
    params = extract_init_params(_NoDetectorDefault)
    assert params["detector"] == "required"
    assert params["contamination"] == 0.1


def test_extract_init_params_excludes_self_and_var_keyword() -> None:
    class WithKwargs:
        def __init__(self, a: int = 1, **kwargs: object) -> None:
            pass

    assert set(extract_init_params(WithKwargs)) == {"a"}


def test_resolve_default_inner_detector_rejects_missing_param() -> None:
    """No `detector` param at all -> `.get()` returns None -> not a str \
    -> ConfigError."""
    with pytest.raises(ConfigError):
        resolve_default_inner_detector(_NoDetectorParam)


def test_resolve_default_inner_detector_rejects_missing_default() -> None:
    with pytest.raises(ConfigError):
        resolve_default_inner_detector(_NoDetectorDefault)


def test_resolve_default_inner_detector_rejects_non_string_default() -> None:
    with pytest.raises(ConfigError):
        resolve_default_inner_detector(_NonStringDetectorDefault)


def test_resolve_default_inner_detector_happy_path() -> None:
    from omniad.algos.text.tfidf import TfidfDetectorAdapter

    assert resolve_default_inner_detector(TfidfDetectorAdapter) == "IsolationForest"


def test_resolve_delegated_capabilities_intersects_with_delegatable_registry() -> None:
    caps = resolve_delegated_capabilities("IsolationForest")
    assert "feature_importance" in caps
    assert "segmentation" not in caps  # non-delegatable by design, and N/A anyway


def test_resolve_delegated_capabilities_dict_is_boolean_lookup() -> None:
    result = resolve_delegated_capabilities_dict("IsolationForest")
    assert result["feature_importance"] is True
    assert result["segmentation"] is False
