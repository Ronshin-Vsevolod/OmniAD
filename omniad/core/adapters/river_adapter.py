"""
Base adapter for stream-native anomaly detectors backed by `river`
(Layer 1.5).

Unlike sklearn (`BaseSklearnAdapter`) or PyTorch (`BaseTorchAdapter`),
every river model shares the same two primitives: `learn_one(x)` and
`score_one(x)`, operating on a single dict-encoded sample. This means
IncrementalLearningMixin support is unconditional for this entire
family — declared once here via ordinary multiple inheritance, not
detected dynamically per-subclass.

Batch fitting (`fit()`) is implemented as a loop over the same
`learn_one`/`score_one` primitives used by `partial_fit()` — for this
family of backends, batch is the degenerate case of
streaming, not the other way around (contrast with BaseTorchAdapter,
where the relationship is reversed: fit() is primary and partial_fit()
is a secondary, opt-in mode).
"""
from __future__ import annotations

import logging
import os
from typing import Any

import joblib
import numpy as np
import numpy.typing as npt

from omniad.core.base import BaseDetector
from omniad.core.mixins import IncrementalLearningMixin

logger = logging.getLogger(__name__)

try:
    import river  # noqa: F401
except ImportError:
    river = None  # noqa: N816


class BaseRiverAdapter(BaseDetector, IncrementalLearningMixin):
    """
    Base template for river-backed streaming anomaly detectors.

    Concrete subclasses only need to implement `_build_backend()`,
    returning a configured river anomaly detector instance. Everything
    else (batch fit as a learn/score loop, single-row streaming
    updates, dict encoding, joblib serialization) is shared.

    Parameters
    ----------
    contamination : float, default=0.1
        Expected proportion of anomalies.
    threshold_strategy : str, ThresholdStrategy, float, or None, default="ewma"
    """

    def __init__(
        self,
        contamination: float = 0.1,
        threshold_strategy: ThresholdName
        | str
        | Callable[..., float]
        | float
        | None = "ewma",
        **kwargs: Any,
    ) -> None:
        super().__init__(
            contamination=contamination, threshold_strategy=threshold_strategy, **kwargs
        )

    def _check_river(self) -> None:
        """Verify that river is installed."""
        if river is None:
            raise ImportError(
                "river is required for streaming-native detectors.\n"
                "  pip install omniad[streaming]"
            )

    def _build_backend(self) -> Any:
        """
        Construct the underlying river model.

        Returns
        -------
        model : Any
            A river model exposing `.learn_one(x)` and `.score_one(x)`.
        """
        raise NotImplementedError("Concrete adapter must implement _build_backend()")

    @staticmethod
    def _row_to_dict(row: npt.NDArray[Any]) -> dict[int, float]:
        """Convert a single numpy row into river's dict-of-features format."""
        return {i: float(v) for i, v in enumerate(row)}

    def _fit_backend(self, X: Any, y: Any | None = None) -> None:
        """
        Batch fit as a loop over the same learn_one/score_one primitive
        used by partial_fit(). Caches training scores (test-then-train)
        to avoid a second pass during threshold calibration.
        """
        self._check_river()
        self._backend_model = self._build_backend()

        scores = np.empty(len(X))
        for i, row in enumerate(X):
            x_dict = self._row_to_dict(row)
            scores[i] = self.backend_model.score_one(x_dict)
            self.backend_model.learn_one(x_dict)
        self._cached_train_scores = scores

    def _predict_score_backend(self, X: Any) -> npt.NDArray[Any]:
        """Score each row via `score_one()`, without mutating model state."""
        return np.array(
            [self.backend_model.score_one(self._row_to_dict(row)) for row in X]
        )

    def _partial_fit_backend(self, x: Any, y: Any | None = None) -> None:
        """Update the backend with a single streaming sample."""
        self.backend_model.learn_one(self._row_to_dict(x[0]))

    def _save_backend(self, path: str) -> None:
        """Save the river model via joblib (river objects are picklable)."""
        joblib.dump(self.backend_model, os.path.join(path, "model.joblib"))

    def _load_backend(self, path: str) -> None:
        """Load the river model via joblib."""
        self._backend_model = joblib.load(os.path.join(path, "model.joblib"))
