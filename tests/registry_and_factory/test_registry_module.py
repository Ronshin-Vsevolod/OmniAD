"""
Tests for omniad/registry.py itself — zero-import queries only.
Lives alongside test_get_detector.py / test_describe_and_list.py since
this is the data those mechanisms are built on.
"""
from __future__ import annotations

from omniad.registry import _REGISTRY, filter_by_domain, list_registered


def test_list_registered_returns_all_registry_keys() -> None:
    assert set(list_registered()) == set(_REGISTRY.keys())


def test_filter_by_domain_matches_static_domain_field() -> None:
    tabular = filter_by_domain("tabular")
    assert "IsolationForest" in tabular
    assert all(_REGISTRY[name]["domain"] == "tabular" for name in tabular)


def test_filter_by_domain_unknown_domain_returns_empty_list() -> None:
    assert filter_by_domain("not_a_real_domain") == []
