"""
Local fixtures for factory/registry mechanism tests, isolating them
from real algorithms' tunable hyperparameters (see fake_registry_entry).
"""
import pytest

from omniad.presets import PRESETS
from omniad.registry import _REGISTRY


@pytest.fixture
def fake_registry_entry(monkeypatch: pytest.MonkeyPatch) -> str:
    """
    Registers a throwaway algorithm backed by _TestOnlyDummyAdapter.

    Returns
    -------
    name : str
        Registry name of the fake algorithm.
    """
    name = "_TestOnlyDummy"
    monkeypatch.setitem(
        _REGISTRY,
        name,
        {"module": "tests.doubles", "requires": None, "domain": "tabular"},
    )
    monkeypatch.setitem(PRESETS, name, {"fast": {"n_estimators": 999}})
    return name
