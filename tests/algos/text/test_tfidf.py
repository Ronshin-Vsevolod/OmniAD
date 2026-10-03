import zipfile
from collections import Counter

import joblib
import numpy as np
import pytest
from sklearn.feature_extraction.text import TfidfVectorizer

from omniad import get_detector
from omniad.core.exceptions import ConfigError


def test_tfidf_parity_with_manual_pipeline(text_dataset) -> None:
    """A. Parity: composition adapter must match manual vectorize + inner detector."""
    train, test, _ = text_dataset

    vectorizer = TfidfVectorizer(max_features=500)
    vectors_train = vectorizer.fit_transform(train)
    vectors_test = vectorizer.transform(test)

    inner = get_detector("IsolationForest", random_state=0, n_jobs=1).fit(vectors_train)
    manual_scores = inner.predict_score(vectors_test)

    model = get_detector(
        "TfidfDetector", max_features=500, random_state=0, detector_kwargs={"n_jobs": 1}
    ).fit(train)

    np.testing.assert_allclose(model.predict_score(test), manual_scores, rtol=1e-5)


def test_tfidf_param_injection(text_dataset) -> None:
    """B. Injection."""
    train, _, _ = text_dataset
    model = get_detector("TfidfDetector", max_features=50, ngram_range=(1, 2)).fit(
        train
    )
    assert model._vectorizer.max_features == 50
    assert model._vectorizer.ngram_range == (1, 2)


def test_tfidf_unknown_detector_raises_config_error() -> None:
    """B. Injection/config guard: `detector=` is validated eagerly,
    before any vectorizer/backend is touched."""
    with pytest.raises(ConfigError):
        get_detector("TfidfDetector", detector="NotARealDetector")


def test_tfidf_determinism(text_dataset) -> None:
    """C. Determinism."""
    train, test, _ = text_dataset

    def make_and_score(seed: int) -> np.ndarray:
        return (
            get_detector("TfidfDetector", random_state=seed)
            .fit(train)
            .predict_score(test)
        )

    np.testing.assert_allclose(make_and_score(0), make_and_score(0), rtol=1e-8)


def test_tfidf_separates_lexically_distinct_anomalies(text_dataset) -> None:
    """
    D. Domain logic: TF-IDF is a bag-of-words model bound to the
    training vocabulary — unlike BERT, it cannot recognize anomalies
    made of unseen words (they vanish at transform() as OOV). Its own
    anomaly signal is instead triggered by *rare-token repetition*:
    a low document-frequency word repeated many times gets a high
    IDF weight and, after L2-normalization, dominates the vector.
    We derive the anomaly directly from the shared fixture's own
    vocabulary instead of hand-writing a parallel corpus, so this
    test exercises the mechanism regardless of which words the
    fixture happens to contain.
    """
    train, normal_test, _ = text_dataset

    word_counts = Counter(w for line in train for w in line.split())
    rare_word = min(word_counts, key=lambda w: word_counts[w])

    anomalies = [f"{rare_word} " * n for n in (6, 10)]
    test = normal_test[:2] + anomalies
    y_test = np.array([0, 0, 1, 1])

    scores = get_detector("TfidfDetector").fit(train).predict_score(test)
    assert scores[y_test == 1].mean() > scores[y_test == 0].mean()


def test_tfidf_delegates_feature_importance_to_inner_detector(text_dataset) -> None:
    """D. Domain logic: regression test for the composition-delegation fix."""
    train, test, _ = text_dataset
    model = get_detector("TfidfDetector").fit(train)

    assert model.capabilities["feature_importance"] is True
    importances = model.get_feature_importances(test)
    assert isinstance(importances, np.ndarray)
    assert np.isfinite(importances).all()
    assert importances.shape == (len(model._vectorizer.get_feature_names_out()),)


def test_tfidf_reports_no_segmentation_or_incremental_learning() -> None:
    """D. Domain logic: capabilities that don't survive flattening to a vector."""
    model = get_detector("TfidfDetector")
    assert model.capabilities["segmentation"] is False
    assert model.capabilities["incremental_learning"] is False


def test_tfidf_load_restores_vectorizer(text_dataset, tmp_path) -> None:
    """D. Domain logic: serialization preserves the fitted vocabulary."""
    train, _, _ = text_dataset
    model = get_detector(
        "TfidfDetector",
        max_features=50,
        random_state=42,
    ).fit(train)

    path = str(tmp_path / "tfidf_model")
    model.save(path)
    loaded = get_detector("TfidfDetector").load(path)

    assert list(loaded._vectorizer.get_feature_names_out()) == list(
        model._vectorizer.get_feature_names_out()
    )


def test_tfidf_class_capabilities_match_default_detector() -> None:
    """D. Domain logic: class discovery uses the default inner detector."""
    from omniad.algos.text.tfidf import TfidfDetectorAdapter

    assert "feature_importance" in TfidfDetectorAdapter.get_capabilities()


def test_tfidf_backend_state_is_not_duplicated_in_wrapper_state(
    text_dataset, tmp_path
) -> None:
    """D. Domain logic: composed backend state is stored only once."""
    train, _, _ = text_dataset
    model = get_detector("TfidfDetector").fit(train)

    path = tmp_path / "model.zip"
    model.save(str(path))

    with zipfile.ZipFile(path) as zf:
        names = set(zf.namelist())
        zf.extract("attributes.pkl", tmp_path)

    state = joblib.load(tmp_path / "attributes.pkl")

    assert "backend/vectorizer.joblib" in names
    assert "backend/detector.zip" in names
    assert "_vectorizer" not in state
    assert "_detector" not in state
