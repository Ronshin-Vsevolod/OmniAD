import numpy as np
import pytest

from omniad import get_detector
from tests.support import require_algo

ALGO = "BertDetector"


def test_bert_output_shape_and_finiteness(text_dataset) -> None:
    """A. Parity substitute: shapes/finite values from the full pipeline."""
    require_algo(ALGO)
    train, test, _ = text_dataset
    scores = get_detector(ALGO, preset="debug").fit(train).predict_score(test)
    assert scores.shape == (len(test),)
    assert np.isfinite(scores).all()


def test_bert_param_injection(text_dataset) -> None:
    """B. Injection."""
    require_algo(ALGO)
    train, _, _ = text_dataset
    model = get_detector(ALGO, preset="debug", pooling="mean", max_length=32).fit(train)
    assert model.max_length == 32
    assert model.pooling == "mean"


def test_bert_determinism(text_dataset) -> None:
    """C. Determinism."""
    require_algo(ALGO)
    train, test, _ = text_dataset

    def make_and_score(seed: int) -> np.ndarray:
        return (
            get_detector(ALGO, preset="debug", random_state=seed)
            .fit(train)
            .predict_score(test)
        )

    np.testing.assert_allclose(make_and_score(0), make_and_score(0), rtol=1e-5)


def test_bert_chunking_handles_long_text_without_error() -> None:
    """D. Domain logic: chunking_strategy must accept arbitrarily long documents."""
    require_algo(ALGO)
    long_text = "anomaly detection " * 2000
    model = get_detector(ALGO, preset="debug", chunking_strategy="mean", max_length=32)
    model.fit(["short normal text", long_text, "another short text"])
    scores = model.predict_score([long_text])
    assert scores.shape == (1,)
    assert np.isfinite(scores).all()


def test_bert_reports_no_segmentation_or_incremental_learning() -> None:
    """D. Domain logic."""
    require_algo(ALGO)
    model = get_detector(ALGO, preset="debug")
    assert model.capabilities["segmentation"] is False
    assert model.capabilities["incremental_learning"] is False


@pytest.mark.slow
def test_bert_separates_known_anomalies_with_real_weights(text_dataset) -> None:
    """
    D. Domain logic (slow): unlike the 'debug' preset (random, untrained
    weights — no semantic signal), pretrained weights should separate
    lexically obvious anomalies.
    """
    require_algo(ALGO)
    train, test, y_test = text_dataset
    scores = get_detector(ALGO, preset="fast").fit(train).predict_score(test)
    assert scores[y_test == 1].mean() > scores[y_test == 0].mean()
