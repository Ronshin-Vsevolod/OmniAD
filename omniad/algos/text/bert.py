from __future__ import annotations

import os
from typing import Any, cast

import numpy.typing as npt

from omniad.core.adapters.transformers_adapter import BaseTransformersAdapter
from omniad.core.exceptions import ConfigError
from omniad.utils.detectors import build_detector, get_available_detectors


class BertDetectorAdapter(BaseTransformersAdapter):
    """
    Text anomaly detector using BERT embeddings.

    Extracts embeddings from a HuggingFace transformer model,
    then fits a configurable anomaly detector on the embedding space.

    Parameters
    ----------
    detector : str, default="IsolationForest"
        Backend anomaly detector. See get_available_detectors().
    model_name : str, default="bert-base-uncased"
        HuggingFace model identifier.
    device : str, default="auto"
        Device for inference: "auto", "cpu", or "cuda".
    batch_size : int, default=32
        Batch size for embedding extraction.
    max_length : int, default=512
        Maximum token length per chunk.
    pooling : str or callable, default="cls"
        Pooling strategy for transformer output.
    chunking_strategy : str, callable, or None, default=None
        Strategy for texts longer than max_length.
    contamination : float, default=0.1
        Expected proportion of anomalies.
    random_state : int or None, default=None
        Random seed for reproducibility.
    save_weights : bool, default=False
        Whether to save transformer weights in the ZIP archive.
        If False, only model_name is saved and weights are
        re-downloaded from HuggingFace on load.
    detector_kwargs : dict or None, default=None
        Additional parameters for the backend detector constructor.

    Examples
    --------
    >>> from omniad import get_detector
    >>> texts = ["normal log entry", "routine check OK", "CRITICAL FAILURE"]
    >>> model = get_detector("BertDetector", contamination=0.1)
    >>> model.fit(texts)
    >>> scores = model.predict_score(["all clear", "KERNEL PANIC segfault"])

    >>> # With custom detector and chunking
    >>> model = get_detector(
    ...     "BertDetector",
    ...     detector="LOF",
    ...     detector_kwargs={"n_neighbors": 10},
    ...     chunking_strategy="mean",
    ...     model_name="distilbert-base-uncased",
    ... )
    """

    def __init__(
        self,
        detector: str = "IsolationForest",
        model_name: str = "bert-base-uncased",
        device: str = "auto",
        batch_size: int = 32,
        max_length: int = 512,
        pooling: str = "cls",
        chunking_strategy: str | None = None,
        contamination: float = 0.1,
        random_state: int | None = None,
        save_weights: bool = False,
        detector_kwargs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        self.detector = detector
        self.random_state = random_state
        self.detector_kwargs = detector_kwargs or {}

        if detector not in get_available_detectors():
            raise ConfigError(
                f"Unknown detector '{detector}'. "
                f"Available: {get_available_detectors()}"
            )

        super().__init__(
            model_name=model_name,
            device=device,
            batch_size=batch_size,
            max_length=max_length,
            pooling=pooling,
            chunking_strategy=chunking_strategy,
            contamination=contamination,
            save_weights=save_weights,
            **kwargs,
        )

        self._detector: Any = None

    # --- Detection on Embeddings ---

    def _fit_on_embeddings(self, embeddings: npt.NDArray[Any], y: Any = None) -> None:
        """Fit anomaly detector on BERT embeddings."""
        self._detector = build_detector(
            name=self.detector,
            caller="BertDetector",
            contamination=self.contamination,
            random_state=self.random_state,
            **(self.detector_kwargs or {}),
        )
        self._detector.fit(embeddings)

    def _score_embeddings(self, embeddings: npt.NDArray[Any]) -> npt.NDArray[Any]:
        """Score embeddings. OmniAD detector handles convention."""
        return cast(npt.NDArray[Any], self._detector.predict_score(embeddings))

    # --- Serialization ---

    def _save_detector(self, path: str) -> None:
        """Save the backend detector."""
        self._detector.save(os.path.join(path, "detector.zip"))

    def _load_detector(self, path: str) -> None:
        """Load the backend detector."""
        self._detector = build_detector(
            name=self.detector,
            caller="BertDetector",
            contamination=self.contamination,
        )
        self._detector.load(os.path.join(path, "detector.zip"))
