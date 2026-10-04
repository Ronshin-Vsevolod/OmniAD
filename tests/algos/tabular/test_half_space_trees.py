from typing import Any

import numpy as np
import numpy.typing as npt
import pytest

from omniad import get_detector

river_anomaly = pytest.importorskip("river.anomaly")


def test_hst_parity_with_raw_river(random_xy_dataset: tuple[Any, Any, Any]) -> None:
    """A. Parity: fit()'s learn/score loop must match a hand-rolled river loop."""
    X_train, _, _ = random_xy_dataset
    seed, n_trees, window_size = 7, 10, 50

    raw = river_anomaly.HalfSpaceTrees(
        n_trees=n_trees, window_size=window_size, seed=seed
    )
    raw_scores = []
    for row in X_train:
        x = {i: float(v) for i, v in enumerate(row)}
        raw_scores.append(raw.score_one(x))
        raw.learn_one(x)

    model = get_detector(
        "HalfSpaceTrees",
        n_trees=n_trees,
        window_size=window_size,
        random_state=seed,
        threshold_strategy=None,
    ).fit(X_train)

    assert model._cached_train_scores is not None
    np.testing.assert_allclose(model._cached_train_scores, raw_scores, rtol=1e-8)


def test_hst_param_injection() -> None:
    """B. Injection."""
    X = np.random.default_rng(0).normal(size=(50, 4))
    model = get_detector("HalfSpaceTrees", n_trees=13, window_size=77).fit(X)
    assert model.backend_model.n_trees == 13
    assert model.backend_model.window_size == 77


def test_hst_determinism(random_xy_dataset: tuple[Any, Any, Any]) -> None:
    """C. Determinism."""
    X_train, X_test, _ = random_xy_dataset

    def make_and_score(seed: int) -> npt.NDArray[Any]:
        model = get_detector("HalfSpaceTrees", window_size=50, random_state=seed)
        return model.fit(X_train).predict_score(X_test)

    np.testing.assert_allclose(make_and_score(42), make_and_score(42), rtol=1e-8)
    assert not np.allclose(make_and_score(42), make_and_score(99))


def test_hst_partial_fit_streams(random_xy_dataset: tuple[Any, Any, Any]) -> None:
    """D. Domain logic: fit() then partial_fit() is this adapter's flagship workflow."""
    X_train, X_test, _ = random_xy_dataset
    model = get_detector("HalfSpaceTrees", window_size=50, random_state=1).fit(X_train)
    before = model.threshold_
    for row in X_test[:20]:
        model.partial_fit(row)
    assert model.threshold_ != before


def test_hst_fit_uses_same_primitives_as_partial_fit(monkeypatch) -> None:
    """D. Domain logic: batch fit uses river's streaming primitives."""
    X = np.random.default_rng(0).normal(size=(20, 3))
    model = get_detector("HalfSpaceTrees")
    backend = model._build_backend()
    monkeypatch.setattr(model, "_build_backend", lambda: backend)

    calls = {"learn": 0, "score": 0}
    original_learn = backend.learn_one
    original_score = backend.score_one

    def counting_learn(x):
        calls["learn"] += 1
        return original_learn(x)

    def counting_score(x):
        calls["score"] += 1
        return original_score(x)

    monkeypatch.setattr(backend, "learn_one", counting_learn)
    monkeypatch.setattr(backend, "score_one", counting_score)

    model.fit(X)

    assert calls == {"learn": len(X), "score": len(X)}
