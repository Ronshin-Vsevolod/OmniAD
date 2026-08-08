from __future__ import annotations

import json
import logging
import os
import shutil
import tempfile
from abc import ABC, abstractmethod
from typing import Any, Callable

import joblib
import numpy as np
import numpy.typing as npt

from omniad.core._logging import _ensure_verbose_handler, log_phase
from omniad.core.exceptions import CapabilityError, ConfigError, ModelNotFittedError
from omniad.core.metrics import reverse_lookup_metric
from omniad.core.mixins import (
    CAPABILITY_REGISTRY,
    IncrementalLearningMixin,
    describe_capability,
)
from omniad.utils.errors import backend_boundary
from omniad.utils.thresholds import (
    ThresholdName,
    ThresholdStrategy,
    resolve_threshold,
)
from omniad.utils.validation import validate_input

logger = logging.getLogger(__name__)


class BaseDetector(ABC):
    """
    Abstract base class for all anomaly detection algorithms.

    Parameters
    ----------
    contamination : float, default=0.1
        The amount of contamination of the data set, i.e. the proportion
        of outliers in the data set. Used when fitting to define the threshold
        on the scores.
    threshold_strategy : str, ThresholdStrategy, float, or None, default="quantile"
        Strategy used to calibrate `threshold_` during `fit()`, and
        (if the strategy supports it) to keep it updated during
        `partial_fit()`. See `omniad.utils.thresholds`.
    verbose : int, default=0
        Verbosity level of output.
    """

    def __init__(
        self,
        contamination: float = 0.1,
        threshold_strategy: ThresholdName
        | str
        | Callable[..., float]
        | float
        | None = "quantile",
        verbose: int = 0,
        **kwargs: Any,
    ) -> None:
        self.contamination = contamination
        self.threshold_strategy = threshold_strategy
        self.verbose = verbose
        self._backend_model: Any = None
        self.threshold_: float | None = None
        self._threshold_engine: ThresholdStrategy | None = None
        self._is_fitted = False
        self._cached_train_scores: npt.NDArray[Any] | None = None

    def fit(self, X: Any, y: Any | None = None) -> BaseDetector:
        """
        Fit the model using X as training data.

        Parameters
        ----------
        X : Any
            Training data. Format depends on the specific adapter
            (e.g., np.ndarray for tabular, List[str] for text).
        y : Any | None, optional
            Target values (ignored for unsupervised methods).

        Returns
        -------
        self : object
            Fitted estimator.
        """
        _ensure_verbose_handler(self.verbose)

        X = self._validate(X)

        self._set_seed()

        with log_phase(logger, "fit") as ctx:
            with backend_boundary(self.__class__.__name__, phase="fit"):
                self._fit_backend(X, y)

            self._calibrate_threshold(X)
            if self.threshold_ is not None:
                ctx["threshold"] = f"{self.threshold_:.6f}"

        self._is_fitted = True
        return self

    def _calibrate_threshold(self, X: Any) -> None:
        """
        Calculate threshold based on threshold_strategy.

        Builds and stores `self._threshold_engine`, a stateful
        ThresholdStrategy instance. For strategies that support
        `update()`, the same engine is later reused by `partial_fit()`,
        so a model can be warm-started on a batch and then continue
        calibrating online.
        """
        strategy = self.threshold_strategy

        if strategy is None:
            self.threshold_ = None
            self._threshold_engine = None
            return

        if isinstance(strategy, (int, float)):
            self.threshold_ = float(strategy)
            self._threshold_engine = None
            return

        if self._cached_train_scores is not None:
            train_scores = self._cached_train_scores
            self._cached_train_scores = None
        else:
            train_scores = self.predict_score(X)

        self._threshold_engine = resolve_threshold(strategy)
        self.threshold_ = float(
            self._threshold_engine.fit(train_scores, self.contamination)
        )

    @abstractmethod
    def _fit_backend(self, X: Any, y: Any | None = None) -> None:
        """
        Actual implementation of the fitting process for the backend model.
        """
        pass

    def predict_score(self, X: Any) -> npt.NDArray[Any]:
        """
        Predict the anomaly score of X of the input samples.

        Parameters
        ----------
        X : Any
            The input samples.

        Returns
        -------
        scores : np.ndarray of shape (n_samples,)
            Higher values indicate larger anomalies.
        """
        X = self._validate(X)
        with backend_boundary(self.__class__.__name__, phase="predict_score"):
            return self._predict_score_backend(X)

    @abstractmethod
    def _predict_score_backend(self, X: Any) -> npt.NDArray[Any]:
        """
        Compute raw anomaly scores from already-validated input.

        Parameters
        ----------
        X : Any
            Output of `self._validate(X)`.

        Returns
        -------
        scores : np.ndarray of shape (n_samples,)
            Higher values indicate larger anomalies.
        """

    def predict(self, X: Any, threshold: float | None = None) -> npt.NDArray[np.int_]:
        """
        Predict if a particular sample is an outlier or not.

        Parameters
        ----------
        X : Any
            The input samples.
        threshold : float | None, optional
            The threshold to use for binarization.
            If None, self.threshold_ is used.

        Returns
        -------
        is_outlier : np.ndarray of shape (n_samples,)
            For each observation, tells whether or not
            it should be considered as an anomaly (1) or not (0).

        Raises
        ------
        ModelNotFittedError
            If the model is not fitted.
        ValueError
            If the threshold is not fitted and not provided.
        """
        if not self._is_fitted:
            raise ModelNotFittedError("Model is not fitted. Call fit() first.")

        scores = self.predict_score(X)

        current_threshold = threshold if threshold is not None else self.threshold_

        if current_threshold is None:
            raise ValueError(
                "Threshold not fitted. Run fit() or pass explicit threshold."
            )

        return (scores > current_threshold).astype(np.int_)

    def partial_fit(self, x: Any, y: Any | None = None) -> BaseDetector:
        """
        Incrementally update the model with a single streaming sample.

        Requires the concrete class to explicitly implement
        `IncrementalLearningMixin` (e.g. `class LSTMAdapter(BaseTorchAdapter,
        IncrementalLearningMixin)`), and `fit()` to have been called at
        least once — cold-start (streaming before any fit()) is not
        supported.

        Parameters
        ----------
        x : Any
            A single sample.
        y : Any | None, optional
            Target value, ignored for unsupervised methods.

        Returns
        -------
        self : object

        Raises
        ------
        CapabilityError
            If the algorithm does not support incremental learning, the
            model has not been fitted yet, or the configured
            threshold_strategy has no online update rule.
        """
        if not isinstance(self, IncrementalLearningMixin):
            raise CapabilityError(
                f"{self.__class__.__name__} does not support incremental " f"learning."
            )

        if not self._is_fitted:
            raise CapabilityError(
                f"{self.__class__.__name__} must be fit() before "
                f"partial_fit() can be used."
            )

        x_valid = self._validate_single(x)

        score = self._score_for_partial_fit(x_valid)

        with backend_boundary(self.__class__.__name__, phase="partial_fit"):
            self._partial_fit_backend(x_valid, y)

        if self._threshold_engine is not None:
            self.threshold_ = self._threshold_engine.update(score, self.contamination)

        return self

    def _score_for_partial_fit(self, x_valid: Any) -> float:
        """
        Compute the anomaly score of an incoming sample for the
        test-then-train threshold update in partial_fit().

        Default: delegates to predict_score(x_valid) — correct for
        detectors whose score is a pure function of a single row
        (tabular streaming). Override for algorithms whose score
        depends on additional state beyond the row itself (e.g.
        windowed time-series models, where scoring requires the
        buffered context) — see LSTMAdapter._score_for_partial_fit.

        Parameters
        ----------
        x_valid : Any
            A single, already-validated sample.

        Returns
        -------
        score : float
        """
        return float(self.predict_score(x_valid)[0])

    def _validate_single(self, x: Any) -> Any:
        """Validate a single streaming sample using this detector's rules."""
        rules = (self.get_validation_rules() - {"require_2d"}) | {"require_single_row"}
        return validate_input(x, rules)

    @property
    def backend_model(self) -> Any:
        """
        Direct access to the underlying library object (Low-Level API).

        Returns
        -------
        backend_model : object
            The fitted inner model (e.g., sklearn estimator or PyTorch module).

        Raises
        ------
        RuntimeError
            If the model is not fitted yet.
        """
        if self._backend_model is None:
            raise ModelNotFittedError("The backend model is not initialized or fitted.")
        return self._backend_model

    @classmethod
    def get_validation_rules(cls) -> set[str]:
        """
        Declare validation requirements for this detector.

        Rules are applied by validate_input in the globally defined
        safe order (_VALIDATION_ORDER). Subclasses override this
        method using super() + set.add / set.discard.

        Returns
        -------
        rules : set[str]
            Rule names from _VALIDATION_REGISTRY.
        """
        return {"to_numpy", "reject_sparse", "require_2d", "reject_nan"}

    def _validate(self, X: Any) -> Any:
        """
        Validate and coerce input data for this detector.

        Called in fit() and may be called in predict_score()
        of concrete adapters. Override only when post-processing
        is needed after validation (e.g. dtype cast).
        Always call super()._validate(X) first in that case.
        """
        return validate_input(X, rules=self.get_validation_rules())

    # --- Discoverability ---

    @classmethod
    def get_capabilities(cls) -> set[str]:
        """
        Declare capability slugs for this class, without instantiation.

        Auto-detects Mixin-derived capabilities (those backed by a
        concrete public method, e.g. "reconstruction" -> predict_expected())
        via CAPABILITY_REGISTRY. This is the same declarative pattern as
        get_validation_rules(): a class-level method the core reads,
        rather than a manually maintained side registry.

        Returns
        -------
        capabilities : set[str]
            Capability slugs, e.g. {"reconstruction", "incremental_learning"}.
        """
        return {
            info.slug for info in CAPABILITY_REGISTRY if issubclass(cls, info.mixin)
        }

    @property
    def capabilities(self) -> dict[str, bool]:
        """
        Boolean lookup for well-known capabilities (see CAPABILITY_REGISTRY).

        Returns
        -------
        capabilities : dict[str, bool]
        """
        active = type(self).get_capabilities()
        return {info.slug: info.slug in active for info in CAPABILITY_REGISTRY}

    def __repr__(self) -> str:
        """Compact single-line representation, safe for logging and containers."""
        status = "fitted" if self._is_fitted else "unfitted"
        threshold = f"{self.threshold_:.4f}" if self.threshold_ is not None else "None"
        return f"<{self.__class__.__name__} {status} threshold={threshold}>"

    def __str__(self) -> str:
        """Multi-line, human-readable representation for print() / console use."""
        lines = [
            f"<{self.__class__.__name__}>",
            f"Fitted: {self._is_fitted}",
            "Capabilities:",
        ]
        slugs = sorted(type(self).get_capabilities())
        if slugs:
            for slug in slugs:
                label, usage = describe_capability(slug)
                lines.append(f"  - {label:<20} -> {usage}" if usage else f"  - {label}")
        else:
            lines.append("  (no extra capabilities)")
        return "\n".join(lines)

    def _repr_html_(self) -> str:
        """Rich HTML representation, auto-used by Jupyter/IPython."""
        slugs = sorted(type(self).get_capabilities())
        rows = "".join(
            f"<tr><td>{label}</td><td><code>{usage or '-'}</code></td></tr>"
            for label, usage in (describe_capability(s) for s in slugs)
        )
        if not rows:
            rows = "<tr><td colspan='2'>No extra capabilities</td></tr>"
        return (
            f"<b>{self.__class__.__name__}</b> (fitted={self._is_fitted})"
            f"<table>{rows}</table>"
        )

    # --- SERIALIZATION (ZIP Container) ---

    def save(self, filepath: str) -> None:
        """
        Save the model to a ZIP archive.

        This method creates a container holding:
        1. Metadata (JSON)
        2. wrapper attributes (Pickle)
        3. Backend model (Native format via _save_backend)

        Parameters
        ----------
        filepath : str
            Path where the model should be saved.
        """
        filepath = str(filepath)
        # Remove extension to allow shutil to add .zip correctly
        base_name = os.path.splitext(filepath)[0]

        with tempfile.TemporaryDirectory() as tmp_dir:
            # 1. Metadata (Class info, version, threshold)
            meta = {
                "class_name": self.__class__.__name__,
                "contamination": self.contamination,
                "threshold": self.threshold_,
                "version": "0.1.0",
            }
            with open(os.path.join(tmp_dir, "metadata.json"), "w") as f:
                json.dump(meta, f)

            # 2. Backend (Native save)
            backend_path = os.path.join(tmp_dir, "backend")
            os.makedirs(backend_path)
            with backend_boundary(self.__class__.__name__, phase="save"):
                self._save_backend(backend_path)

            # 3. Wrapper Attributes (Scalers, configs, etc.)
            # We make a copy and remove the heavy backend model to avoid pickling it
            state = self.__dict__.copy()
            state.pop("_backend_model", None)

            # If the model has a 'score_metric' attribute that is a function,
            # convert it to string name for pickling.
            metric = state.get("score_metric")
            if callable(metric):
                name = reverse_lookup_metric(metric)
                if name is None:
                    raise ConfigError(
                        "Cannot save model with unregistered custom metric. "
                        "Please register it:\n"
                        "  from omniad.core.metrics import register_metric\n"
                        "  register_metric('my_metric', func)"
                    )
                state["score_metric"] = name

            joblib.dump(state, os.path.join(tmp_dir, "attributes.pkl"))

            # 4. Pack into .zip
            shutil.make_archive(base_name, "zip", tmp_dir)

    def load(self, filepath: str) -> BaseDetector:
        """
        Load the model from a ZIP archive.

        Parameters
        ----------
        filepath : str
            Path to the saved model file.

        Returns
        -------
        self : object
            Loaded estimator.
        """
        if not os.path.exists(filepath) and os.path.exists(filepath + ".zip"):
            filepath += ".zip"

        with tempfile.TemporaryDirectory() as tmp_dir:
            shutil.unpack_archive(filepath, tmp_dir)

            # 1. Restore Attributes (Scalers, etc.)
            attributes = joblib.load(os.path.join(tmp_dir, "attributes.pkl"))
            self.__dict__.update(attributes)

            # 2. Restore Backend
            backend_path = os.path.join(tmp_dir, "backend")
            with backend_boundary(self.__class__.__name__, phase="load"):
                self._load_backend(backend_path)

            self._is_fitted = True
            return self

    @abstractmethod
    def _save_backend(self, path: str) -> None:
        """
        Save the backend model using its native mechanism.

        Parameters
        ----------
        path : str
            Directory path where the model files should be stored.
        """
        pass

    @abstractmethod
    def _load_backend(self, path: str) -> None:
        """
        Load the backend model from the specified directory.

        Parameters
        ----------
        path : str
            Directory path containing the model files.
        """
        pass

    def _set_seed(self) -> None:  # noqa: B027
        """
        Set random seed for reproducibility.

        This method should be overridden by adapters to set seeds
        for specific backends (numpy, torch, etc.).
        """
        pass
