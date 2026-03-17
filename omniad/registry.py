"""
Module registry mapping algorithm names to their implementation modules.
"""
from typing import Optional, TypedDict


class RegistryEntry(TypedDict):
    module: str
    requires: Optional[list[str]]


_REGISTRY: dict[str, RegistryEntry] = {
    "IsolationForest": {
        "module": "omniad.algos.tabular.iforest",
        "requires": None,
    },
    "LSTM": {
        "module": "omniad.algos.timeseries.lstm",
        "requires": ["deep"],
    },
    "BertDetector": {
        "module": "omniad.algos.text.bert",
        "requires": ["text", "deep"],
    },
    "TfidfDetector": {"module": "omniad.algos.text.tfidf", "requires": None},
    "ConvAutoencoder": {
        "module": "omniad.algos.cv.autoencoder",
        "requires": ["deep"],
    },
}

# Mapping from group name to a main dependence.
# Used for runtime checks before importing the module.
_DEPENDENCY_CHECKS = {
    "deep": "torch",
    "text": "transformers",
    "graph": "torch_geometric",
    "viz": "matplotlib",
}
