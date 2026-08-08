"""
Module registry mapping algorithm names to their implementation modules.
"""
from __future__ import annotations

from typing import TypedDict


class RegistryEntry(TypedDict):
    module: str
    requires: list[str] | None
    domain: str


_REGISTRY: dict[str, RegistryEntry] = {
    "IsolationForest": {
        "module": "omniad.algos.tabular.iforest",
        "requires": None,
        "domain": "tabular",
    },
    "LSTM": {
        "module": "omniad.algos.timeseries.lstm",
        "requires": ["deep"],
        "domain": "timeseries",
    },
    "BertDetector": {
        "module": "omniad.algos.text.bert",
        "requires": ["text", "deep"],
        "domain": "text",
    },
    "TfidfDetector": {
        "module": "omniad.algos.text.tfidf",
        "requires": None,
        "domain": "text",
    },
    "ConvAutoencoder": {
        "module": "omniad.algos.cv.autoencoder",
        "requires": ["deep"],
        "domain": "cv",
    },
    "HalfSpaceTrees": {
        "module": "omniad.algos.tabular.half_space_trees",
        "requires": ["streaming"],
        "domain": "tabular",
    },
}

# Mapping from group name to a main dependence.
# Used for runtime checks before importing the module.
_DEPENDENCY_CHECKS = {
    "deep": "torch",
    "text": "transformers",
    "graph": "torch_geometric",
    "viz": "matplotlib",
    "streaming": "river",
}


def list_registered() -> list[str]:
    """
    List names of all registered algorithms.
    """
    return list(_REGISTRY.keys())


def filter_by_domain(domain: str) -> list[str]:
    """
    Zero-import filtering by domain — `domain` is a static field of
    RegistryEntry, so this never needs to import any adapter module.
    """
    return [name for name, entry in _REGISTRY.items() if entry["domain"] == domain]
