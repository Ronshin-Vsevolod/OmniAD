"""
Per-capability contract checks, keyed by CAPABILITY_REGISTRY slugs.
"""
from __future__ import annotations

from typing import Any, Callable

import numpy as np

from omniad.core.base import BaseDetector

CapabilityCheck = Callable[[BaseDetector, Any], None]


def _check_feature_importance(model: BaseDetector, X: Any) -> None:
    importances = model.get_feature_importances(X)
    assert isinstance(importances, np.ndarray)
    assert np.isfinite(importances).all()
    if hasattr(X, "shape"):
        assert importances.shape == (X.shape[1],)


def _check_reconstruction(model: BaseDetector, X: Any) -> None:
    reconstructed = model.predict_expected(X)
    assert isinstance(reconstructed, np.ndarray)
    assert len(reconstructed) > 0


def _check_segmentation(model: BaseDetector, X: Any) -> None:
    anomaly_map = model.predict_map(X)
    assert isinstance(anomaly_map, np.ndarray)
    assert anomaly_map.shape[0] == len(X)


def _check_incremental_learning(model: BaseDetector, X: Any) -> None:
    model.partial_fit(X[0])


CAPABILITY_CHECKS: dict[str, CapabilityCheck] = {
    "feature_importance": _check_feature_importance,
    "reconstruction": _check_reconstruction,
    "segmentation": _check_segmentation,
    "incremental_learning": _check_incremental_learning,
}
