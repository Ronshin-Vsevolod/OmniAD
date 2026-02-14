from __future__ import annotations

import json
import os
import shutil
import tempfile
from abc import ABC, abstractmethod
from typing import Any

import joblib
import numpy as np
import numpy.typing as npt

import omniad
from omniad.core.exceptions import ModelNotFittedError


class BaseDetector(ABC):
    """
    Abstract base class for all anomaly detection algorithms.

    Parameters
    ----------
    contamination : float, default=0.1
        The amount of contamination of the data set, i.e. the proportion
        of outliers in the data set. Used when fitting to define the threshold
        on the scores.
    """

    def __init__(self, contamination: float = 0.1, **kwargs: Any) -> None:
        self.contamination = contamination
        self._backend_model: Any = None
        self.threshold_: float | None = None
        self._is_fitted = False

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
        # 1. Validation (via utils)
        # X = validate_input(X)

        # 2. Seed fixing
        self._set_seed()

        # 3. Backend fitting
        self._fit_backend(X, y)

        # 4. Threshold calculation
        # We calculate scores on training data to determine the cut-off point
        train_scores = self.predict_score(X)
        self.threshold_ = float(np.quantile(train_scores, 1 - self.contamination))

        self._is_fitted = True
        return self

    @abstractmethod
    def _fit_backend(self, X: Any, y: Any | None = None) -> None:
        """
        Actual implementation of the fitting process for the backend model.
        """
        pass

    @abstractmethod
    def predict_score(self, X: Any) -> np.ndarray[Any, Any]:
        """
        Predict the anomaly score of X of the input samples.

        Parameters
        ----------
        X : Any
            The input samples.

        Returns
        -------
        scores : np.ndarray of shape (n_samples,)
            The anomaly score of the input samples.
            Higher values indicate larger anomalies.
        """
        pass

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

    # --- SERIALIZATION (ZIP Container) ---

    def save(self, filepath: str) -> None:  # *
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
                "version": omniad.__version__,
            }
            with open(os.path.join(tmp_dir, "metadata.json"), "w") as f:
                json.dump(meta, f)

            # 2. Backend (Native save)
            backend_path = os.path.join(tmp_dir, "backend")
            os.makedirs(backend_path)
            self._save_backend(backend_path)

            # 3. Wrapper Attributes (Scalers, configs, etc.)
            # We make a copy and remove the heavy backend model to avoid pickling it
            state = self.__dict__.copy()
            state.pop("_backend_model", None)
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
