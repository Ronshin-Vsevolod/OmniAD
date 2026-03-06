from __future__ import annotations

import logging
import os
from typing import Any, cast

import numpy.typing as npt
from sklearn.feature_extraction.text import TfidfVectorizer

from omniad.core.base import BaseDetector
from omniad.core.exceptions import ConfigError
from omniad.utils.detectors import build_detector, get_available_detectors
from omniad.utils.validation import validate_text

logger = logging.getLogger(__name__)


class TfidfDetectorAdapter(BaseDetector):
    """
    Text anomaly detector using TF-IDF vectorization.

    Converts raw text to TF-IDF vectors, then fits a configurable
    OmniAD tabular detector on the resulting feature matrix.

    Parameters
    ----------
    detector : str, default="IsolationForest"
        Backend anomaly detector. Supported: "IsolationForest", "LOF", "OCSVM".
    max_features : int, default=1000
        Maximum number of TF-IDF features (vocabulary size).
    ngram_range : tuple[int, int], default=(1, 1)
        Range of n-grams for TF-IDF. (1, 2) includes bigrams.
    contamination : float, default=0.1
        Expected proportion of anomalies.
    random_state : int or None, default=None
        Random seed for reproducibility.
    detector_kwargs : dict or None, default=None
        Additional parameters passed directly to the detector constructor.
        Example: {"n_estimators": 200} for IForest,
                 {"n_neighbors": 20} for LOF.

    Examples
    --------
    >>> from omniad import get_detector
    >>> logs = ["User login", "User logout", "KERNEL PANIC segfault"]
    >>> model = get_detector("TfidfDetector", contamination=0.1)
    >>> model.fit(logs)

    >>> model = get_detector(
    ...     "TfidfDetector",
    ...     detector="LOF",
    ...     detector_kwargs={"n_neighbors": 10},
    ... )
    """

    def __init__(
        self,
        detector: str = "IsolationForest",
        max_features: int = 1000,
        ngram_range: tuple[int, int] = (1, 1),
        contamination: float = 0.1,
        random_state: int | None = None,
        detector_kwargs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        self.detector = detector
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.random_state = random_state
        self.detector_kwargs = detector_kwargs or {}

        if detector not in get_available_detectors():
            raise ConfigError(
                f"Unknown detector '{detector}'."
                f"Available: {get_available_detectors()}"
            )

        super().__init__(contamination=contamination, **kwargs)

    def _validate(self, X: Any) -> Any:
        """Validate that input is a list of non-empty strings."""
        return validate_text(X)

    def _fit_backend(self, X: Any, y: Any | None = None) -> None:
        logger.debug(
            "Vectorizer: max_features=%d, ngram_range=%s",
            self.max_features,
            self.ngram_range,
        )
        logger.debug("Detector: %s", self.detector)

        self._vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            ngram_range=self.ngram_range,
        )
        vectors = self._vectorizer.fit_transform(X)

        self._detector = build_detector(
            name=self.detector,
            caller="TfidfDetector",
            contamination=self.contamination,
            random_state=self.random_state,
            **(self.detector_kwargs or {}),
        )
        self._detector.fit(vectors.toarray())
        self._backend_model = self._detector.backend_model

    def predict_score(self, X: Any) -> npt.NDArray[Any]:
        X = self._validate(X)
        vectors = self._vectorizer.transform(X)
        return cast(npt.NDArray[Any], self._detector.predict_score(vectors.toarray()))

    def _save_backend(self, path: str) -> None:
        import joblib

        joblib.dump(self._vectorizer, os.path.join(path, "vectorizer.joblib"))
        self._detector.save(os.path.join(path, "detector.zip"))

    def _load_backend(self, path: str) -> None:
        import joblib

        self._vectorizer = joblib.load(os.path.join(path, "vectorizer.joblib"))
        self._detector = build_detector(
            name=self.detector,
            caller="TfidfDetector",
            contamination=self.contamination,
        )
        self._detector.load(os.path.join(path, "detector.zip"))
        self._backend_model = self._detector.backend_model
