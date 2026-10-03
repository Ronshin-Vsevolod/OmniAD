"""
Layer 3 tests for omniad/registry.py itself — zero-import queries only.
"""
from __future__ import annotations

from omniad.registry import _REGISTRY, filter_by_domain, list_registered


def test_list_registered_returns_all_registry_keys() -> None:
    assert set(list_registered()) == set(_REGISTRY.keys())


def test_filter_by_domain_matches_static_domain_field() -> None:
    tabular = filter_by_domain("tabular")
    assert "IsolationForest" in tabular
    assert all(_REGISTRY[name]["domain"] == "tabular" for name in tabular)


def test_filter_by_domain_unknown_domain_returns_empty() -> None:
    assert filter_by_domain("not_a_real_domain") == []
