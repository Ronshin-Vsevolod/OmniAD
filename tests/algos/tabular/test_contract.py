import numpy as np
import pytest

from omniad import get_detector, list_detectors
from tests.support import fast_kwargs

TABULAR_ALGOS = list_detectors(domain="tabular")


@pytest.mark.parametrize("algo_name", TABULAR_ALGOS)
def test_predict_score_shape_and_dtype(algo_name: str, random_xy_dataset) -> None:
    X_train, X_test, _ = random_xy_dataset
    scores = (
        get_detector(algo_name, **fast_kwargs(algo_name))
        .fit(X_train)
        .predict_score(X_test)
    )
    assert scores.shape == (len(X_test),)
    assert np.isfinite(scores).all()


@pytest.mark.parametrize("algo_name", TABULAR_ALGOS)
def test_predict_labels_are_binary(algo_name: str, random_xy_dataset) -> None:
    X_train, X_test, _ = random_xy_dataset
    labels = (
        get_detector(algo_name, **fast_kwargs(algo_name)).fit(X_train).predict(X_test)
    )
    assert set(np.unique(labels)).issubset({0, 1})


@pytest.mark.parametrize("algo_name", TABULAR_ALGOS)
def test_save_load_roundtrip(algo_name: str, random_xy_dataset, tmp_path) -> None:
    X_train, X_test, _ = random_xy_dataset
    model = get_detector(algo_name, **fast_kwargs(algo_name)).fit(X_train)
    before = model.predict_score(X_test)
    model.save(str(tmp_path / "model.zip"))
    loaded = get_detector(algo_name)
    loaded.load(str(tmp_path / "model.zip"))
    np.testing.assert_allclose(before, loaded.predict_score(X_test), rtol=1e-5)
