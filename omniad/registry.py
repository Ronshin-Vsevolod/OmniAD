"""
Module registry mapping algorithm names to their implementation modules.
"""

# Key: Algorithm Name (used in get_detector)
# Value: Module path (dot-separated string)
_REGISTRY = {
    "IsolationForest": "omniad.algos.tabular.iforest",
}
