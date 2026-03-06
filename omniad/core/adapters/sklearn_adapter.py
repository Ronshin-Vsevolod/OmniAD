from __future__ import annotations

import logging
import os
from typing import Any, ClassVar, cast

import joblib
import numpy as np
import numpy.typing as npt

from omniad.core.base import BaseDetector
from omniad.core.exceptions import ConfigError

logger = logging.getLogger(__name__)


class BaseSklearnAdapter(BaseDetector):
    """
    Base template for Scikit-learn based anomaly detectors.

    Implements common logic for parameter mapping, input validation,
    score inversion, and serialization via joblib.

    Parameters
    ----------
    contamination : float, default=0.1
        The proportion of outliers in the data set.
    """

    _backend_cls: ClassVar[type | None] = None
    _param_mapping: ClassVar[dict[str, str]] = {}
    _invert_score: ClassVar[bool] = True

    def __init__(self, contamination: float = 0.1, **kwargs: Any) -> None:
        super().__init__(contamination=contamination, **kwargs)
        self.backend_options = kwargs.get("backend_options", {})
        # Store extra kwargs for backend passthrough
        self.kwargs = kwargs

    def _fit_backend(self, X: Any, y: Any | None = None) -> None:
        """
        Fits the sklearn backend model.
        """
        # 1. Validate Input
        if self._backend_cls is None:
            raise ConfigError(
                f"Adapter {self.__class__.__name__} must define '_backend_cls'."
            )

        # 2. Prepare Parameters
        # Priority: backend_options > kwargs > mapped params > defaults
        init_params = self.kwargs.copy()

        # Remove special keys that shouldn't go to backend
        init_params.pop("backend_options", None)
        init_params.pop("threshold_strategy", None)

        # Apply mapping: keys in self.__dict__ -> keys expected by backend
        for local_name, backend_name in self._param_mapping.items():
            if hasattr(self, local_name):
                value = getattr(self, local_name)
                init_params[backend_name] = value

        # Explicit backend options override everything
        init_params.update(self.backend_options)

        logger.debug("Params: %s", init_params)

        # 3. Initialize & Fit
        self._backend_model = self._backend_cls(**init_params)
        self._backend_model.fit(X, y)

        self._cached_train_scores = self.predict_score(X)

        logger.debug("Backend fitted: %s", self._backend_cls.__name__)

    def predict_score(self, X: Any) -> npt.NDArray[Any]:
        """
        Predict anomaly scores using the backend model.

        Attempts to use 'score_samples' first, then 'decision_function'.
        Inverts scores if '_invert_score' is True.
        """
        X = self._validate(X)

        logger.debug("predict_score: n_samples=%d", len(X))

        # Try standard sklearn methods
        scores: Any
        if hasattr(self.backend_model, "decision_function"):
            scores = self.backend_model.decision_function(X)
        elif hasattr(self.backend_model, "score_samples"):
            scores = self.backend_model.score_samples(X)
        else:
            raise ConfigError(
                f"Backend {type(self.backend_model)} has neither "
                "'score_samples' nor 'decision_function'."
            )

        scores_arr = cast("npt.NDArray[Any]", np.asarray(scores))

        # Handle inversion (1 is anomaly)
        if self._invert_score:
            scores_arr = -scores_arr

        return scores_arr

    def _save_backend(self, path: str) -> None:
        """Save sklearn model using joblib."""
        model_path = os.path.join(path, "model.joblib")
        joblib.dump(self.backend_model, model_path)

    def _load_backend(self, path: str) -> None:
        """Load sklearn model using joblib."""
        model_path = os.path.join(path, "model.joblib")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Backend model not found at {model_path}")

        self._backend_model = joblib.load(model_path)
