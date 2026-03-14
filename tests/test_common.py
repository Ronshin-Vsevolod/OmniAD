"""
A terrible decision. Remove this code and create a common logic at the domain level.
"""

from typing import Any

import numpy as np
import pytest

from omniad import get_detector
from omniad.core.base import BaseDetector
from omniad.core.mixins import FeatureImportanceMixin, ReconstructionMixin
from omniad.registry import _REGISTRY

REGISTERED_ALGOS = list(_REGISTRY.keys())


@pytest.mark.parametrize("algo_name", REGISTERED_ALGOS)  # type: ignore[misc]
def test_api_contract(
    algo_name: str,
    domain_dataset: tuple[Any, Any],
) -> None:
    """
    Smoke Test: Verifies that any algorithm from the registry:
    1. Initializes.
    2. Trains (fit).
    3. Predicts (predict_score, predict).
    """

    X_train, X_test = domain_dataset

    model = get_detector(algo_name, contamination=0.1)

    assert isinstance(model, BaseDetector)
    assert not model._is_fitted

    model.fit(X_train)
    assert model._is_fitted
    assert model.threshold_ is not None

    scores = model.predict_score(X_test)
    assert isinstance(scores, np.ndarray)

    expected_len = len(X_test)
    if hasattr(model, "window_size"):
        ws = int(model.window_size)
        expected_len = len(X_test) - ws + 1

    assert np.issubdtype(scores.dtype, np.number)

    labels = model.predict(X_test)
    assert labels.shape == (expected_len,)
    assert set(np.unique(labels)).issubset({0, 1})


@pytest.mark.parametrize("algo_name", REGISTERED_ALGOS)  # type: ignore[misc]
def test_save_load_cycle(
    algo_name: str,
    domain_dataset: tuple[Any, Any],
    tmp_path: Any,
) -> None:
    """
    Verifies that any algorithm supports serialization.
    """
    X_train, X_test = domain_dataset

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


@pytest.mark.parametrize("algo_name", REGISTERED_ALGOS)  # type: ignore[misc]
def test_mixins_contract(algo_name: str, domain_dataset: tuple[Any, Any]) -> None:
    """
    Automatic verification of all declared mixins.
    If an algorithm claims to support “feature importance,” we verify that it's
    telling the truth.
    """
    X_train, X_test = domain_dataset

    model = get_detector(algo_name, contamination=0.1)
    model.fit(X_train)

    # 1. Feature Importance Check
    if isinstance(model, FeatureImportanceMixin):
        if X_train.ndim == 2:
            importances = model.get_feature_importances(X_test)

            assert isinstance(importances, np.ndarray)
            assert importances.shape == (X_train.shape[1],)
            assert np.isfinite(importances).all()
        else:
            from omniad.core.exceptions import DataFormatError

            with pytest.raises(DataFormatError):
                model.get_feature_importances(X_test)

    # 2. Reconstruction Check
    if isinstance(model, ReconstructionMixin):
        recon = model.predict_expected(X_test)
        assert isinstance(recon, np.ndarray)
