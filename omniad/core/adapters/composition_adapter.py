"""
Shared forwarding logic for adapters that embed raw domain input into
a numeric vector space and delegate anomaly detection to another,
separately registered OmniAD detector.
"""
from __future__ import annotations

from typing import Any, cast

import numpy.typing as npt

from omniad.core.exceptions import CapabilityError, ModelNotFittedError
from omniad.utils.errors import backend_boundary_method


class BaseCompositionAdapter:
    """
    Template for adapters that transform raw domain input (e.g. text)
    into numeric vectors and hand off anomaly detection to another
    OmniAD detector, stored as `self._detector`.

    Requires the host class to:
    - initialize `self._detector` to None before fitting;
    - expose the wrapped detector's registry name as `self.detector`;
    - implement `_to_vectors(X)`, converting validated domain input
      into the representation `self._detector` was fit on.
    """

    _serialization_exclude = frozenset({"_detector"})
    _detector: Any
    detector: str

    def _to_vectors(self, X: Any) -> Any:
        raise NotImplementedError("Concrete adapter must implement _to_vectors()")

    def _require_delegated(self, slug: str) -> None:
        """
        Raises
        ------
        ModelNotFittedError
            If the wrapped detector has not been created yet.
        CapabilityError
            If the configured `detector=` does not support `slug`.
        """
        if self._detector is None:
            raise ModelNotFittedError(f"{type(self).__name__} must be fit() first.")
        if slug not in type(self._detector).get_capabilities():
            raise CapabilityError(
                f"detector='{self.detector}' does not support '{slug}'. "
                f"Pick an inner detector that does."
            )

    @backend_boundary_method("feature_importances")
    def get_feature_importances(self, X: Any = None, **kwargs: Any) -> npt.NDArray[Any]:
        """
        Feature importance computed over the vector representation of X.

        Parameters
        ----------
        X : Any, optional
            Domain input. Required unless `method="native"`.
        **kwargs : Any
            Forwarded to the wrapped detector's `get_feature_importances`.

        Returns
        -------
        importances : np.ndarray

        Raises
        ------
        CapabilityError
            If the configured `detector=` does not support feature importance.
        """
        self._require_delegated("feature_importance")
        vectors = self._to_vectors(X) if X is not None else None
        return cast(
            "npt.NDArray[Any]",
            self._detector.get_feature_importances(vectors, **kwargs),
        )

    @backend_boundary_method("predict_expected")
    def predict_expected(self, X: Any) -> npt.NDArray[Any]:
        """
        Reconstruction/forecast computed over the vector representation of X.

        Parameters
        ----------
        X : Any
            Domain input.

        Returns
        -------
        expected : np.ndarray

        Raises
        ------
        CapabilityError
            If the configured `detector=` does not support reconstruction.
        """
        self._require_delegated("reconstruction")
        vectors = self._to_vectors(X)
        return cast(
            "npt.NDArray[Any]",
            self._detector.predict_expected(vectors),
        )
