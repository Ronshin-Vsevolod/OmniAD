"""
Base adapter for HuggingFace Transformers-based anomaly detectors (Layer 1.5).

Manages the full lifecycle: tokenizer/model loading, embedding extraction,
batching, chunking for long texts, device management, and serialization.

Subclasses implement detection logic on top of embeddings via:
- _fit_on_embeddings()
- _score_embeddings()
- _save_detector()
- _load_detector()
"""

from __future__ import annotations

import json
import logging
import os
from abc import abstractmethod
from typing import Any, Callable, cast

import numpy as np
import numpy.typing as npt

from omniad.core.base import BaseDetector
from omniad.core.exceptions import ConfigError
from omniad.utils.text import (
    resolve_chunking_strategy,
    resolve_pooling,
    reverse_lookup_chunking,
    reverse_lookup_pooling,
)
from omniad.utils.validation import validate_text

try:
    import torch
except ImportError:
    torch = None

try:
    from transformers import AutoModel, AutoTokenizer
except ImportError:
    AutoModel = None
    AutoTokenizer = None

logger = logging.getLogger(__name__)


class BaseTransformersAdapter(BaseDetector):
    """
    Base adapter for HuggingFace Transformers-based detectors.

    Provides infrastructure for the pipeline:
    text → tokenize → embed (with optional chunking) → detect anomalies.

    The transformer model itself is NOT saved (potentially gigabytes).
    Only model_name is stored; on load the model is re-initialized
    from HuggingFace cache.

    Parameters
    ----------
    model_name : str, default="bert-base-uncased"
        HuggingFace model identifier.
    device : str, default="auto"
        Device for inference: "auto", "cpu", or "cuda".
    batch_size : int, default=32
        Batch size for embedding extraction.
    max_length : int, default=512
        Maximum token length per chunk.
    pooling : str, default="cls"
        Pooling strategy: "cls" (CLS token) or "mean"
        (mean of all tokens weighted by attention mask).
    chunking_strategy : str or None, default=None
        How to handle texts longer than max_length.

        - None : truncate (default).
        - "mean" : split into chunks, embed each, average embeddings.
        - "max" : split into chunks, take chunk with highest L2 norm.
    contamination : float, default=0.1
        Expected proportion of anomalies.
    save_weights : bool, default=False
        If True, save full model weights (state_dict) in addition to config.
    """

    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        device: str = "auto",
        batch_size: int = 32,
        max_length: int = 512,
        pooling: str = "cls",
        chunking_strategy: str | Callable[..., Any] | None = None,
        contamination: float = 0.1,
        save_weights: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(contamination=contamination, **kwargs)
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.max_length = max_length
        self.pooling = pooling
        self._pool_fn = resolve_pooling(pooling)
        self.chunking_strategy = chunking_strategy
        self._chunk_fn = resolve_chunking_strategy(chunking_strategy)
        self.save_weights = save_weights

        self._tokenizer: Any = None
        self._transformer: Any = None
        self._torch_device: Any = None

        # Embedding cache: used to avoid double inference during fit()
        self._cached_embeddings: npt.NDArray[Any] | None = None

    # --- Validation ---

    def _validate(self, X: Any) -> Any:
        """Text-specific: validate List[str]."""
        return validate_text(X)

    # --- Transformer Lifecycle ---

    def _check_transformers(self) -> None:
        """Verify that torch and transformers are available."""
        if torch is None:
            raise ImportError(
                "PyTorch is required for transformer-based detectors.\n"
                "  pip install omniad[deep]"
            )
        if AutoModel is None:
            raise ImportError(
                "HuggingFace Transformers is required.\n" "  pip install omniad[text]"
            )

    def _resolve_device(self) -> Any:
        """Resolve device string to torch.device."""
        if self.device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(self.device)

    def _init_transformer(self) -> None:
        """Lazy-load tokenizer and model. Idempotent."""
        if self._tokenizer is not None:
            return

        self._check_transformers()
        self._torch_device = self._resolve_device()

        logger.info("Loading '%s' on %s", self.model_name, self._torch_device)

        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._transformer = AutoModel.from_pretrained(self.model_name)
        self._transformer.eval()
        self._transformer.to(self._torch_device)

        self._backend_model = self._transformer

    # --- Embedding Extraction ---

    def _pool_output(self, last_hidden_state: Any, attention_mask: Any) -> Any:
        """Pool transformer output using configured strategy."""
        return self._pool_fn(last_hidden_state, attention_mask)

    def _embed_batch(self, texts: list[str]) -> npt.NDArray[Any]:
        """
        Embed a batch of texts with truncation.

        Returns
        -------
        embeddings : np.ndarray of shape (len(texts), hidden_dim)
        """
        inputs = self._tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        ).to(self._torch_device)

        with torch.no_grad():
            outputs = self._transformer(**inputs)

        pooled = self._pool_output(outputs.last_hidden_state, inputs["attention_mask"])
        return cast(npt.NDArray[Any], pooled.cpu().numpy())

    def _embed_single_chunked(self, text: str) -> npt.NDArray[Any]:
        """
        Embed a single long text by splitting into token-level chunks.

        Returns
        -------
        embedding : np.ndarray of shape (hidden_dim,)
        """
        encoding = self._tokenizer(
            text,
            truncation=False,
            return_attention_mask=True,
            return_tensors="pt",
        )
        input_ids = encoding["input_ids"][0]

        if len(input_ids) <= self.max_length:
            return cast(npt.NDArray[Any], self._embed_batch([text])[0])

        # Build chunks preserving special tokens ([CLS]...[SEP])
        cls_id = input_ids[0].item()
        sep_id = input_ids[-1].item()
        content = input_ids[1:-1]
        chunk_size = self.max_length - 2

        chunk_embeddings = []
        for start in range(0, len(content), chunk_size):
            chunk_content = content[start : start + chunk_size]
            chunk_ids = torch.tensor([[cls_id] + chunk_content.tolist() + [sep_id]]).to(
                self._torch_device
            )
            mask = torch.ones_like(chunk_ids)

            with torch.no_grad():
                outputs = self._transformer(input_ids=chunk_ids, attention_mask=mask)

            emb = self._pool_output(outputs.last_hidden_state, mask)
            chunk_embeddings.append(emb[0].cpu().numpy())

        chunks_arr = np.stack(chunk_embeddings)
        assert self._chunk_fn is not None
        return self._chunk_fn(chunks_arr)

    def _embed(self, texts: list[str]) -> npt.NDArray[Any]:
        """
        Extract embeddings for all texts with batching and optional chunking.

        Returns
        -------
        embeddings : np.ndarray of shape (len(texts), hidden_dim)
        """
        if self._chunk_fn is None:
            # Simple: batch embed with truncation
            parts: list[npt.NDArray[Any]] = []
            for i in range(0, len(texts), self.batch_size):
                batch = texts[i : i + self.batch_size]
                parts.append(self._embed_batch(batch))
            return np.concatenate(parts, axis=0)
        else:
            # Chunking: process each text individually
            return np.array([self._embed_single_chunked(t) for t in texts])

    # --- fit / predict_score ---
    def _fit_backend(self, X: Any, y: Any | None = None) -> None:
        """
        Embed texts, fit detector, cache embeddings for threshold calculation.

        The cache avoids a second BERT inference pass when
        BaseDetector.fit() calls predict_score(X) immediately after.
        """
        self._check_transformers()
        self._init_transformer()

        embeddings = self._embed(X)
        self._fit_on_embeddings(embeddings, y)

        # Cache for the upcoming predict_score() call from BaseDetector.fit()
        self._cached_embeddings = embeddings

    def predict_score(self, X: Any) -> npt.NDArray[Any]:
        """
        Compute anomaly scores for texts.

        Parameters
        ----------
        X : list[str]
            Texts to score.

        Returns
        -------
        scores : np.ndarray of shape (n_samples,)
            Higher values indicate more anomalous texts.
        """
        X = self._validate(X)

        # Use cached embeddings from _fit_backend if available
        if self._cached_embeddings is not None:
            embeddings = self._cached_embeddings
            self._cached_embeddings = None
        else:
            if self._transformer is None:
                self._init_transformer()
            embeddings = self._embed(X)

        return self._score_embeddings(embeddings)

    # --- Abstract interface for subclasses ---

    @abstractmethod
    def _fit_on_embeddings(self, embeddings: npt.NDArray[Any], y: Any = None) -> None:
        """
        Fit the anomaly detector on embedding vectors.

        Parameters
        ----------
        embeddings : np.ndarray of shape (n_samples, hidden_dim)
        y : ignored
        """

    @abstractmethod
    def _score_embeddings(self, embeddings: npt.NDArray[Any]) -> npt.NDArray[Any]:
        """
        Compute anomaly scores from embedding vectors.

        Parameters
        ----------
        embeddings : np.ndarray of shape (n_samples, hidden_dim)

        Returns
        -------
        scores : np.ndarray of shape (n_samples,)
            Higher = more anomalous.
        """

    @abstractmethod
    def _save_detector(self, path: str) -> None:
        """Save the detection component (not the transformer)."""

    @abstractmethod
    def _load_detector(self, path: str) -> None:
        """Load the detection component."""

    # --- Serialization ---

    def _save_backend(self, path: str) -> None:
        """Save transformer config + detector. Optionally save weights."""
        self._cached_embeddings = None

        chunking = self.chunking_strategy
        if callable(chunking):
            name = reverse_lookup_chunking(chunking)
            if name is None:
                raise ConfigError(
                    "Cannot save model with unregistered chunking strategy. "
                    "Register it first:\n"
                    "  from omniad.utils.text import register_chunking_strategy\n"
                    "  register_chunking_strategy('my_strategy', func)"
                )
            chunking = name

        pooling = self.pooling
        if callable(pooling):
            name = reverse_lookup_pooling(pooling)
            if name is None:
                raise ConfigError(
                    "Cannot save model with unregistered pooling strategy. "
                    "Register it first:\n"
                    "  from omniad.utils.text import register_pooling\n"
                    "  register_pooling('my_pooling', func)"
                )
            pooling = name

        config = {
            "model_name": self.model_name,
            "pooling": pooling,
            "max_length": self.max_length,
            "chunking_strategy": chunking,
            "weights_saved": self.save_weights,
        }
        with open(os.path.join(path, "transformer_config.json"), "w") as f:
            json.dump(config, f)

        if self.save_weights:
            torch.save(
                self._transformer.state_dict(),
                os.path.join(path, "transformer_weights.pt"),
            )

        self._save_detector(path)

    def _load_backend(self, path: str) -> None:
        """Restore config, re-initialize transformer, load detector."""
        config_path = os.path.join(path, "transformer_config.json")
        if os.path.exists(config_path):
            with open(config_path) as f:
                config = json.load(f)
            self.model_name = config.get("model_name", self.model_name)
            self.pooling = config.get("pooling", self.pooling)
            self.max_length = config.get("max_length", self.max_length)
            self.chunking_strategy = config.get(
                "chunking_strategy", self.chunking_strategy
            )
            self._pool_fn = resolve_pooling(self.pooling)
            self._chunk_fn = resolve_chunking_strategy(self.chunking_strategy)

        self._init_transformer()

        weights_path = os.path.join(path, "transformer_weights.pt")
        if os.path.exists(weights_path):
            state_dict = torch.load(weights_path, map_location=self._torch_device)
            self._transformer.load_state_dict(state_dict)
            self._transformer.eval()

        self._load_detector(path)

    # --- Seed ---

    def _set_seed(self) -> None:
        """Set seeds for numpy and torch."""
        random_state = getattr(self, "random_state", None)
        if random_state is not None:
            np.random.seed(random_state)
            if torch is not None:
                torch.manual_seed(random_state)
