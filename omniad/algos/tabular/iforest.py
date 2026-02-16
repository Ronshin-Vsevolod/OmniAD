from __future__ import annotations

from typing import Any, ClassVar

from sklearn.ensemble import IsolationForest

from omniad.core.adapters.sklearn_adapter import BaseSklearnAdapter


class IsolationForestAdapter(BaseSklearnAdapter):
    """
    Wrapper for Scikit-learn Isolation Forest.

    Parameters
    ----------
    n_estimators : int, default=100
        The number of base estimators in the ensemble.
    contamination : float, default=0.1
        The amount of contamination of the data set.
    n_jobs : int, default=-1
        The number of jobs to run in parallel. -1 means using all processors.
    random_state : int, optional
        Controls the pseudo-randomness of the selection of the feature
        and split values.
    """

    _backend_cls: ClassVar[type | None] = IsolationForest
    _param_mapping: ClassVar[dict[str, str]] = {}

    def __init__(
        self,
        n_estimators: int = 100,
        contamination: float = 0.1,
        n_jobs: int = -1,
        random_state: int | None = None,
        **kwargs: Any,
    ) -> None:
        self.n_estimators = n_estimators
        self.n_jobs = n_jobs
        self.random_state = random_state

        super().__init__(
            contamination=contamination,
            n_estimators=n_estimators,
            n_jobs=n_jobs,
            random_state=random_state,
            **kwargs,
        )
