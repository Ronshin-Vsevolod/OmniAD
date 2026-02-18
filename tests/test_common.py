from typing import Any

import numpy as np
import pytest

from omniad import get_detector
from omniad.core.base import BaseDetector
from omniad.registry import _REGISTRY

REGISTERED_ALGOS = list(_REGISTRY.keys())


@pytest.mark.parametrize("algo_name", REGISTERED_ALGOS)  # type: ignore[misc]
def test_api_contract(
    algo_name: str,
    random_xy_dataset: tuple[Any, Any, Any],
) -> None:
    """
    Smoke Test: Verifies that any algorithm from the registry:
    1. Initializes.
    2. Trains (fit).
    3. Predicts (predict_score, predict).
    """
    X_train, X_test, _ = random_xy_dataset

    model = get_detector(algo_name, contamination=0.1)

    assert isinstance(model, BaseDetector)
    assert not model._is_fitted

    model.fit(X_train)
    assert model._is_fitted
    assert model.threshold_ is not None

    scores = model.predict_score(X_test)
    assert isinstance(scores, np.ndarray)
    assert scores.shape == (len(X_test),)
    assert np.issubdtype(scores.dtype, np.number)

    labels = model.predict(X_test)
    assert labels.shape == (len(X_test),)
    assert set(np.unique(labels)).issubset({0, 1})


@pytest.mark.parametrize("algo_name", REGISTERED_ALGOS)  # type: ignore[misc]
def test_save_load_cycle(
    algo_name: str,
    random_xy_dataset: tuple[Any, Any, Any],
    tmp_path: Any,
) -> None:
    """
    Verifies that any algorithm supports serialization.
    """
    X_train, X_test, _ = random_xy_dataset

    model = get_detector(algo_name, contamination=0.1)
    model.fit(X_train)

    score_before = model.predict_score(X_test)

    save_path = tmp_path / "model.zip"
    model.save(str(save_path))

    loaded_model = get_detector(algo_name)
    loaded_model.load(str(save_path))

    score_after = loaded_model.predict_score(X_test)

    np.testing.assert_allclose(
        score_before,
        score_after,
        rtol=1e-5,
        err_msg=f"Scores mismatch after save/load for {algo_name}",
    )
    assert loaded_model.threshold_ == model.threshold_
