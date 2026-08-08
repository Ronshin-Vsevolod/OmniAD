from __future__ import annotations

from typing import Any

from omniad.core.adapters.river_adapter import BaseRiverAdapter

try:
    import river.anomaly
except ImportError:
    river = None  # noqa: N816


class HalfSpaceTreesAdapter(BaseRiverAdapter):
    """
    Half-Space Trees anomaly detector for tabular data.

    A tree-ensemble method designed from the ground up for streaming
    data with bounded memory — unlike LSTM (batch-native, with
    partial_fit() as a secondary opt-in mode), this algorithm has no
    batch-only formulation to begin with; `fit()` here is simply a
    loop over the same incremental primitive used by `partial_fit()`.

    Operates on flat feature vectors, same input shape as
    IsolationForest — the difference from IsolationForest is entirely
    in processing mode (incremental vs batch), not in data domain;
    see IncrementalLearningMixin / model.capabilities for discovery.

    Parameters
    ----------
    n_trees : int, default=25
        Number of trees in the ensemble.
    window_size : int, default=250
        Number of observations per tree before a reset.
    contamination : float, default=0.1
        Expected proportion of anomalies.

    Examples
    --------
    >>> from omniad import get_detector
    >>> import numpy as np
    >>> X = np.random.randn(1000, 5)
    >>> model = get_detector("HalfSpaceTrees")
    >>> model.fit(X)
    >>> for row in X[:10]:
    ...     model.partial_fit(row)
    """

    def __init__(
        self,
        n_trees: int = 25,
        window_size: int = 250,
        contamination: float = 0.1,
        random_state: int | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(contamination=contamination, **kwargs)
        self.n_trees = n_trees
        self.window_size = window_size
        self.random_state = random_state

    def _build_backend(self) -> Any:
        self._check_river()
        return river.anomaly.HalfSpaceTrees(
            n_trees=self.n_trees, window_size=self.window_size, seed=self.random_state
        )
