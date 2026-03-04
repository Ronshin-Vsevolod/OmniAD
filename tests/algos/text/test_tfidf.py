from typing import Any

import numpy as np
import pytest
from sklearn.ensemble import IsolationForest as SklearnIF
from sklearn.feature_extraction.text import TfidfVectorizer

from omniad import get_detector
from omniad.core.exceptions import DataFormatError

# --- A. Parity ---


def test_tfidf_parity(
    text_dataset: tuple[list[str], list[str], np.ndarray[Any, Any]],
) -> None:
    """
    A. Parity Test.

    Verifies that TfidfDetectorAdapter scores equal the scores
    of IsolationForest trained manually on the same TF-IDF matrix
    with the same parameters.
    """
    train_texts, _, _ = text_dataset
    seed = 42
    max_features = 50
    n_estimators = 100

    # Reference: manual pipeline
    vectorizer = TfidfVectorizer(max_features=max_features)
    X_matrix = vectorizer.fit_transform(train_texts).toarray()

    sk_model = SklearnIF(n_estimators=n_estimators, random_state=seed, n_jobs=1)
    sk_model.fit(X_matrix)
    sk_scores = -sk_model.decision_function(X_matrix)

    # Our adapter
    model = get_detector(
        "TfidfDetector",
        max_features=max_features,
        random_state=seed,
        detector_kwargs={"n_estimators": n_estimators, "n_jobs": 1},
    )
    model.fit(train_texts)
    our_scores = model.predict_score(train_texts)

    np.testing.assert_allclose(
        our_scores,
        sk_scores,
        rtol=1e-5,
        err_msg="TfidfDetectorAdapter scores must match manual\
         TF-IDF + IForest pipeline",
    )


# --- B. Parameter Injection ---


def test_tfidf_param_injection(
    text_dataset: tuple[list[str], list[str], np.ndarray[Any, Any]],
) -> None:
    """
    B. Parameter Injection Test.

    Verifies that vectorizer and detector parameters
    are correctly passed to internal components.
    """
    train_texts, _, _ = text_dataset
    max_features = 42
    ngram_range = (1, 2)
    n_estimators = 17

    model = get_detector(
        "TfidfDetector",
        max_features=max_features,
        ngram_range=ngram_range,
        detector_kwargs={"n_estimators": n_estimators},
    )
    model.fit(train_texts)

    assert model._vectorizer.max_features == max_features  # type: ignore[attr-defined]
    assert model._vectorizer.ngram_range == ngram_range  # type: ignore[attr-defined]
    assert model._detector.backend_model.n_estimators == n_estimators  # type: ignore[attr-defined]


# --- C. Determinism ---


def test_tfidf_determinism(
    text_dataset: tuple[list[str], list[str], np.ndarray[Any, Any]],
) -> None:
    """
    C. Determinism Test.

    Verifies that same random_state produces identical scores.
    Different random_state must produce different scores.
    """
    train_texts, _, _ = text_dataset

    def make_and_score(seed: int) -> np.ndarray[Any, Any]:
        model = get_detector("TfidfDetector", random_state=seed)
        model.fit(train_texts)
        return model.predict_score(train_texts)

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


# --- D. Domain Logic ---


def test_tfidf_rejects_numeric_input() -> None:
    """
    D. Domain Logic Test.

    Verifies that numeric input is rejected with a clear error.
    """
    model = get_detector("TfidfDetector")
    with pytest.raises(DataFormatError):
        model.fit(np.array([[1.0, 2.0], [3.0, 4.0]]))


def test_tfidf_rejects_empty_strings() -> None:
    """
    D. Domain Logic Test.

    Verifies that empty/whitespace-only strings are rejected.
    """
    model = get_detector("TfidfDetector")
    with pytest.raises(DataFormatError):
        model.fit(["valid text", "   ", "another valid text"])


def test_tfidf_unknown_detector_raises() -> None:
    """
    D. Domain Logic Test.

    Verifies that unknown detector name raises ConfigError at init time.
    """
    from omniad.core.exceptions import ConfigError

    with pytest.raises(ConfigError):
        get_detector("TfidfDetector", detector="NonExistentDetector")
