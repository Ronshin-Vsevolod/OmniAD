from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy.typing as npt


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
            The reconstructed/predicted values. Same shape as X (or target subset).
        """
        pass
