from typing import Any

import numpy as np
import numpy.typing as npt
from sklearn.ensemble import IsolationForest as SklearnIF

from omniad import get_detector
from omniad.core.mixins import FeatureImportanceMixin


def test_iforest_parity(random_xy_dataset: tuple[Any, Any, Any]) -> None:
    """
    A. Parity Test.

    Verifies that IsolationForestAdapter scores equal -decision_function
    of the equivalent sklearn model trained on the same data with the same seed.
    """
    X_train, X_test, _ = random_xy_dataset

    seed = 42
    n_estimators = 100

    sk_model = SklearnIF(n_estimators=n_estimators, random_state=seed, n_jobs=1)
    sk_model.fit(X_train)
    sk_scores = sk_model.decision_function(X_test)

    our_model = get_detector(
        "IsolationForest",
        n_estimators=n_estimators,
        random_state=seed,
        n_jobs=1,
    )
    our_model.fit(X_train)
    our_scores = our_model.predict_score(X_test)

    np.testing.assert_allclose(
        our_scores,
        -sk_scores,
        rtol=1e-5,
        err_msg="IsolationForestAdapter scores must equal -decision_function(X)",
    )


def test_iforest_param_injection() -> None:
    """
    B. Parameter Injection Test.

    Verifies that constructor parameters are correctly passed to the backend.
    """
    n_estimators = 17
    n_jobs = 2
    verbose = 1

    model = get_detector(
        "IsolationForest",
        n_estimators=n_estimators,
        n_jobs=n_jobs,
        verbose=verbose,
    )

    # Backend is created only inside fit — check adapter attrs before fit
    assert model.n_estimators == n_estimators  # type: ignore[attr-defined]
    assert model.n_jobs == n_jobs  # type: ignore[attr-defined]

    X = np.random.randn(50, 5).astype(np.float32)
    model.fit(X)

    # Verify parameters reached the backend
    assert model.backend_model.n_estimators == n_estimators
    assert model.backend_model.n_jobs == n_jobs
    assert model.backend_model.verbose == verbose


def test_iforest_determinism(random_xy_dataset: tuple[Any, Any, Any]) -> None:
    """
    C. Determinism Test.

    Verifies that two models with the same random_state produce identical scores.
    """
    X_train, X_test, _ = random_xy_dataset

    def make_and_score(seed: int) -> npt.NDArray[Any]:
        model = get_detector("IsolationForest", random_state=seed, n_jobs=1)
        model.fit(X_train)
        return model.predict_score(X_test)

    scores_a = make_and_score(seed=42)
    scores_b = make_and_score(seed=42)
    scores_c = make_and_score(seed=99)

    np.testing.assert_allclose(
        scores_a,
        scores_b,
        rtol=1e-8,
        err_msg="Same random_state must produce identical scores",
    )

    assert not np.allclose(
        scores_a, scores_c
    ), "Different random_state should produce different scores"


def test_iforest_feature_importance_logic() -> None:
    """
    D. Domain test
    Let's verify that Permutation Importance actually identifies important features.
    """
    np.random.seed(42)
    n_samples = 500

    # Feature 0: Informative (clearly distinguishes between normal values and anomalies)
    f0_normal = np.random.normal(0, 1, n_samples)
    f0_anom = np.random.normal(10, 1, 50)
    f0 = np.concatenate([f0_normal, f0_anom])

    # Feature 1: Noise (uniform distribution)
    f1 = np.random.uniform(-5, 5, n_samples + 50)

    X = np.column_stack([f0, f1])

    model = get_detector("IsolationForest", n_estimators=100, random_state=42)
    model.fit(X)

    assert isinstance(model, FeatureImportanceMixin)

    imp = model.get_feature_importances(X, n_repeats=5, random_state=42)

    assert (
        imp[0] > imp[1]
    ), f"Informative feature should have higher importance. Got {imp}"
