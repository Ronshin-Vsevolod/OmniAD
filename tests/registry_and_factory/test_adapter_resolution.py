"""
Tests for omniad._resolve_adapter_class(), the shared resolution
mechanism behind get_detector(), describe(), and list_detectors().
"""
from __future__ import annotations

import pytest

from omniad import _resolve_adapter_class
from omniad.core.exceptions import ConfigError


def test_unknown_name_raises_config_error() -> None:
    with pytest.raises(ConfigError):
        _resolve_adapter_class("TotallyMadeUpAlgorithm")


def test_unimportable_module_raises_import_error(broken_registry_entry: str) -> None:
    with pytest.raises(ImportError, match="Failed to import module"):
        _resolve_adapter_class(broken_registry_entry)


def test_module_missing_expected_class_raises_attribute_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from omniad.registry import _REGISTRY

    # Real, importable module that does not define `FakeMissingClassAdapter`.
    monkeypatch.setitem(
        _REGISTRY,
        "_TestOnlyMissingClass",
        {
            "module": "omniad.algos.tabular.iforest",
            "requires": None,
            "domain": "tabular",
        },
    )
    with pytest.raises(AttributeError, match="has no class"):
        _resolve_adapter_class("_TestOnlyMissingClass")
