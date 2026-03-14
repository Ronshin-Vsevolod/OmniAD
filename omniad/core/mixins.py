"""
The FeatureImportanceMixin can also be adapted for time series*
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, cast

import numpy as np
import numpy.typing as npt

from omniad.core.exceptions import ConfigError, DataFormatError
from omniad.utils.validation import validate_input

logger = logging.getLogger(__name__)


class ReconstructionMixin(ABC):
    """
    Mixin for models that reconstruct input data (Autoencoders, LSTMs).
    """

    @abstractmethod
    def predict_expected(self, X: Any) -> npt.NDArray[Any]:
        """
        Return the model's reconstruction or forecast for X.

        Parameters
        ----------
        X : Any
            Input data.

        Returns
        -------
        X_expected : np.ndarray
            The reconstructed/predicted values.
            Note: For time-series models with windowing, the output length
            may be shorter than the input length (N - window_size + 1).
        """
        pass


class FeatureImportanceMixin:
    """
    Mixin providing model-agnostic global feature importance
    via Permutation Importance.
    """

    def get_feature_importances(
        self,
        X: Any = None,
        method: str = "permutation",
        n_repeats: int = 3,
        random_state: int | None = None,
    ) -> npt.NDArray[Any]:
        """
        Calculate global feature importances.

        Parameters
        ----------
        X : Any, optional
            The input data used to evaluate importance. Required if
            method="permutation".
        method : str, default="permutation"
            The strategy to compute importance:
            - "native": Uses the backend model's built-in `feature_importances_`.
            - "permutation": Model-agnostic permutation importance.
        n_repeats : int, default=3
            Number of times to permute a feature (for "permutation" method).
        random_state : int | None, default=None
            Seed for random permutation. If None, tries to use the model's
            global `random_state` defined during __init__.

        Returns
        -------
        importances : np.ndarray of shape (n_features,)
            Normalized importance scores (sum to 1).
        """
        if method not in ("native", "permutation"):
            raise ConfigError(
                f"Unknown feature importance method: '{method}'. "
                "Supported: 'auto', 'native', 'permutation'."
            )

        # 1. Native Path
        if method == "native":
            if hasattr(self, "backend_model") and hasattr(
                self.backend_model, "feature_importances_"
            ):
                logger.debug("Using native feature importances from backend model.")
                imp = self.backend_model.feature_importances_
                return cast("npt.NDArray[Any]", np.asarray(imp))

            raise ConfigError(
                f"The backend model for {self.__class__.__name__} does not expose "
                "`feature_importances_` natively. "
                "If you want to compute it via permutations, explicitly pass "
                "method='permutation' and provide X."
            )

        # 2. Permutation Path
        if X is None:
            raise ValueError(
                "Input X is required for 'permutation' feature importance method."
            )

        if not hasattr(self, "predict_score"):
            raise NotImplementedError(
                "Permutation importance requires predict_score() to be implemented."
            )

        X_arr = validate_input(X, ensure_2d=False, allow_nd=True)
        if X_arr.ndim != 2:
            raise DataFormatError(
                f"Permutation importance supports only 2D tabular data. "
                f"Got ndim={X_arr.ndim}."
            )

        seed = (
            random_state
            if random_state is not None
            else getattr(self, "random_state", None)
        )
        rng = np.random.default_rng(seed)

        logger.debug(
            "Calculating permutation feature importance (repeats=%d, seed=%s).",
            n_repeats,
            seed,
        )

        base_scores = self.predict_score(X_arr)

        n_features = X_arr.shape[1]
        importances = np.zeros(n_features)

        for j in range(n_features):
            diffs = np.zeros(n_repeats)
            for i in range(n_repeats):
                X_permuted = X_arr.copy()
                rng.shuffle(X_permuted[:, j])

                permuted_scores = self.predict_score(X_permuted)
                diffs[i] = np.mean(np.abs(base_scores - permuted_scores))

            importances[j] = diffs.mean()

        total = importances.sum()
        if total > 0:
            importances = importances / total

        return importances
