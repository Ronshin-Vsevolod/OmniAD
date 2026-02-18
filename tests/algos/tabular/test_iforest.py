from typing import Any

import numpy as np
from sklearn.ensemble import IsolationForest as SklearnIF

from omniad import get_detector


def test_iforest_parity(random_xy_dataset: tuple[Any, Any, Any]) -> None:
    """
    Verifies that wrapper behaves the same as the original Sklearn.
    """
    X_train, X_test, _ = random_xy_dataset

    seed = 3
    n_estimators = 50

    sk_model = SklearnIF(n_estimators=n_estimators, random_state=seed, n_jobs=1)
    sk_model.fit(X_train)

    sk_scores = sk_model.decision_function(X_test)

    our_model = get_detector(
        "IsolationForest", n_estimators=n_estimators, random_state=seed, n_jobs=1
    )
    our_model.fit(X_train)

    assert our_model.backend_model.n_estimators == n_estimators
    assert our_model.backend_model.random_state == seed

    our_model._backend_model = sk_model
    our_scores_from_injected = our_model.predict_score(X_test)

    np.testing.assert_allclose(
        our_scores_from_injected,
        -sk_scores,
        rtol=1e-5,
        err_msg="IsolationForestAdapter did not invert sklearn scores correctly",
    )


def test_iforest_params_mapping() -> None:
    """
    Verifies that specific parameters are mapped correctly.
    (Although the names match in IForest, this test is important for future changes).
    """
    model = get_detector("IsolationForest", n_estimators=10, verbose=1)
    # MyPy does not know that BaseDetector has n_estimators.
    # We know this because we created a specific adapter. We ignore the error.
    assert model.n_estimators == 10  # type: ignore[attr-defined]
    model.fit(np.array([[1, 2], [3, 4]]))
    # backend_model returns Any, so MyPy remains silent here
    assert model.backend_model.n_estimators == 10
    assert model.backend_model.verbose == 1
