"""
Mixins providing optional, discoverable capabilities for anomaly
detectors. See CAPABILITY_REGISTRY for the full list and
BaseDetector.capabilities / describe_capability() for how they surface
to users.
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, NamedTuple, cast

import numpy as np
import numpy.typing as npt

from omniad.core.exceptions import CapabilityError, ConfigError, DataFormatError
from omniad.utils.errors import backend_boundary_method
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

    @backend_boundary_method("feature_importances")
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

            raise CapabilityError(
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

        X_arr = validate_input(X, {"to_numpy", "require_2d", "reject_nan"})
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


class SegmentationMixin(ABC):
    """
    Mixin for models that can produce pixel/spatial-level anomaly maps.

    Typical use: CV defect detection, where the model localizes
    anomalous regions within an image.
    """

    @abstractmethod
    def predict_map(self, X: Any) -> npt.NDArray[Any]:
        """
        Predict spatial anomaly map.

        Parameters
        ----------
        X : Any
            Input data (e.g., images of shape (N, C, H, W)).

        Returns
        -------
        anomaly_map : np.ndarray
            Per-pixel anomaly scores. Shape depends on domain:
            - CV: (N, H, W) — higher values indicate defects.
        """


class IncrementalLearningMixin(ABC):
    """
    Mixin for detectors that support incremental (online) fitting,
    updating internal state one sample at a time.

    Deliberately as thin as ReconstructionMixin/SegmentationMixin: a
    single abstract method, no hidden lifecycle hooks. A backend is
    considered incremental only if a concrete algorithm explicitly
    inherits this mixin and implements `_partial_fit_backend` — there
    is no automatic detection based on the wrapped library. This keeps
    the decision visible in the algorithm's own class declaration
    (same idiom as `class LSTMAdapter(BaseTorchAdapter, ReconstructionMixin)`),
    instead of introducing implicit per-template inference rule
    """

    @abstractmethod
    def _partial_fit_backend(self, x: Any, y: Any | None = None) -> None:
        """
        Update model state using a single validated sample.

        Parameters
        ----------
        x : Any
            A single validated sample, shape (1, n_features).
        y : Any | None, optional
            Target value, ignored for unsupervised methods.
        """


class CapabilityInfo(NamedTuple):
    """
    Declarative description of a single discoverable capability.

    delegatable : bool, default=False
        Whether this capability's contract still holds after a
        domain-to-vector transform, and can therefore be forwarded by
        a composition adapter (see BaseCompositionAdapter) to whatever
        inner detector it wraps.
    """

    mixin: type
    slug: str
    label: str
    usage: str
    delegatable: bool = False


CAPABILITY_REGISTRY: list[CapabilityInfo] = [
    CapabilityInfo(
        FeatureImportanceMixin,
        "feature_importance",
        "Feature Importances",
        "model.get_feature_importances(X)",
        delegatable=True,
    ),
    CapabilityInfo(
        ReconstructionMixin,
        "reconstruction",
        "Data Reconstruction",
        "model.predict_expected(X)",
        delegatable=True,
    ),
    CapabilityInfo(
        SegmentationMixin,
        "segmentation",
        "Pixel-level Maps",
        "model.predict_map(X)",
        delegatable=False,  # spatial structure is destroyed by the
        # flatten-to-vector transform
    ),
    CapabilityInfo(
        IncrementalLearningMixin,
        "incremental_learning",
        "Online Learning",
        "model.partial_fit(x)",
        delegatable=False,  # would require merging two independent threshold_ states
    ),
]


def describe_capability(slug: str) -> tuple[str, str | None]:
    """
    Resolve a capability slug to (label, usage_hint) for display.

    Falls back to (slug, None) for adapter-declared capabilities that
    have no CAPABILITY_REGISTRY entry (flags with no dedicated public
    method, e.g. a hypothetical "gpu_accelerated").
    """
    for info in CAPABILITY_REGISTRY:
        if info.slug == slug:
            return info.label, info.usage
    return slug, None
