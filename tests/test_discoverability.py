"""
Layer 4 tests for the discoverability edge cases of omniad/__init__.py:
_resolve_adapter_class() failure modes and list_detectors(capability=...)
resilience when a registered algorithm can't be imported.
"""
from __future__ import annotations

import pytest

import omniad
from omniad.core.exceptions import ConfigError


def test_resolve_adapter_class_unknown_name_raises_config_error() -> None:
    with pytest.raises(ConfigError):
        omniad._resolve_adapter_class("TotallyMadeUpAlgorithm")


def test_resolve_adapter_class_raises_import_error_for_broken_module(
    monkeypatch
) -> None:
    monkeypatch.setitem(
        omniad._REGISTRY,
        "BrokenFakeAlgo",
        {
            "module": "omniad.algos.tabular.this_module_does_not_exist",
            "requires": None,
            "domain": "tabular",
        },
    )
    with pytest.raises(ImportError, match="Failed to import module"):
        omniad._resolve_adapter_class("BrokenFakeAlgo")


def test_resolve_adapter_class_raises_attribute_error_for_wrong_class_name(
    monkeypatch,
) -> None:
    # Points to a real, importable module that doesn't define
    # `FakeMissingClassAdapter`.
    monkeypatch.setitem(
        omniad._REGISTRY,
        "FakeMissingClass",
        {
            "module": "omniad.algos.tabular.iforest",
            "requires": None,
            "domain": "tabular",
        },
    )
    with pytest.raises(AttributeError, match="has no class"):
        omniad._resolve_adapter_class("FakeMissingClass")


def test_list_detectors_capability_filter_silently_skips_unimportable_algo(
    monkeypatch,
) -> None:
    monkeypatch.setitem(
        omniad._REGISTRY,
        "BrokenFakeAlgoForCapabilityFilter",
        {
            "module": "omniad.algos.tabular.this_module_does_not_exist_either",
            "requires": None,
            "domain": "tabular",
        },
    )
    # Must not raise, even though one registered algorithm is unimportable.
    result = omniad.list_detectors(capability="feature_importance")
    assert "IsolationForest" in result
    assert "BrokenFakeAlgoForCapabilityFilter" not in result


def test_list_detectors_capability_filter_combines_with_domain() -> None:
    result = omniad.list_detectors(domain="tabular", capability="feature_importance")
    assert result == ["IsolationForest"]
