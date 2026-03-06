from typing import Any

from omniad import get_detector


def test_verbose_does_not_break_fit(domain_dataset: tuple[Any, Any]) -> None:
    """Verify that verbose=1 does not affect model behavior."""
    X_train, X_test = domain_dataset
    model = get_detector("IsolationForest", verbose=1)
    model.fit(X_train)
    scores = model.predict_score(X_test)
    assert len(scores) == len(X_test)
