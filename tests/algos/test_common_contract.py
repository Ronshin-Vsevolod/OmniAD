"""
Cross-domain contract tests: properties that must hold for every
registered algorithm.
"""

from __future__ import annotations

import numpy as np
import pytest

from omniad import get_detector, list_detectors
from tests.support import AllDomainDatasets, fast_kwargs

ALL_ALGOS = list_detectors()


@pytest.mark.parametrize("algo_name", ALL_ALGOS)
def test_predict_score_is_finite_1d_and_not_longer_than_input(
    algo_name: str, all_domain_datasets: AllDomainDatasets
) -> None:
    X_train, X_test = all_domain_datasets.for_algo(algo_name)
    model = get_detector(algo_name, **fast_kwargs(algo_name)).fit(X_train)
    scores = model.predict_score(X_test)

    assert scores.ndim == 1
    assert 0 < len(scores) <= len(X_test)
    assert np.isfinite(scores).all()


@pytest.mark.parametrize("algo_name", ALL_ALGOS)
def test_predict_labels_are_binary(
    algo_name: str, all_domain_datasets: AllDomainDatasets
) -> None:
    X_train, X_test = all_domain_datasets.for_algo(algo_name)
    model = get_detector(algo_name, **fast_kwargs(algo_name)).fit(X_train)
    labels = model.predict(X_test)
    assert set(np.unique(labels)).issubset({0, 1})


@pytest.mark.parametrize("algo_name", ALL_ALGOS)
def test_save_load_roundtrip_reproduces_predict_score(
    algo_name: str, all_domain_datasets: AllDomainDatasets, tmp_path
) -> None:
    X_train, X_test = all_domain_datasets.for_algo(algo_name)
    model = get_detector(algo_name, **fast_kwargs(algo_name)).fit(X_train)
    before = model.predict_score(X_test)

    path = str(tmp_path / "model.zip")
    model.save(path)

    loaded = get_detector(algo_name)
    loaded.load(path)
    after = loaded.predict_score(X_test)

    np.testing.assert_allclose(before, after, rtol=1e-5)
