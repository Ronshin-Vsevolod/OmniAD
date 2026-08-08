"""
Test-suite helpers. Kept explicit (plain functions, no reflection into
pytest internals) so tests using them stay readable in isolation.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from omniad.presets import PRESETS
from omniad.registry import _DEPENDENCY_CHECKS, _REGISTRY
from omniad.utils.deps import is_available


def _missing_package(algo_name: str) -> str | None:
    for group in _REGISTRY[algo_name].get("requires") or []:
        pkg = _DEPENDENCY_CHECKS.get(group)
        if pkg and not is_available(pkg):
            return f"'{algo_name}' requires '{pkg}' (pip install omniad[{group}])"
    return None


def require_algo(algo_name: str) -> None:
    """
    Skip the current test if `algo_name`'s optional dependencies aren't
    installed.

    Not needed inside tests parametrized over `algo_name` — the
    autouse `skip_if_dependency_missing` fixture already handles that
    case. Use this only when the algorithm name is a hardcoded literal
    inside the test body, not a pytest parameter.
    """
    reason = _missing_package(algo_name)
    if reason:
        pytest.skip(reason)


def fast_kwargs(algo_name: str) -> dict[str, Any]:
    """contamination=0.1, plus preset='debug' when the algorithm defines one."""
    kwargs: dict[str, Any] = {"contamination": 0.1}
    if "debug" in PRESETS.get(algo_name, {}):
        kwargs["preset"] = "debug"
    return kwargs


@dataclass(frozen=True)
class AllDomainDatasets:
    """
    Bundle of one (train, test) pair per domain.

    Used only by the single genuinely domain-agnostic test in the
    suite (capability truthfulness — see tests/contracts/).
    Domain-specific test files should request the matching fixture
    directly instead.
    """

    tabular: tuple[Any, Any]
    timeseries: tuple[Any, Any]
    text: tuple[Any, Any]
    cv: tuple[Any, Any]

    def for_algo(self, algo_name: str) -> tuple[Any, Any]:
        domain = _REGISTRY[algo_name]["domain"]
        return getattr(self, domain)
