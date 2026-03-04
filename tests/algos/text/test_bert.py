from typing import Any

import numpy as np
import pytest

pytest.importorskip("torch", reason="torch not installed")
pytest.importorskip("transformers", reason="transformers not installed")

from omniad import get_detector  # noqa: E402
from omniad.core.exceptions import ConfigError, DataFormatError  # noqa: E402

TINY_MODEL = "hf-internal-testing/tiny-random-bert"


# --- A. Parity ---


def test_bert_parity(
    text_dataset: tuple[list[str], list[str], np.ndarray[Any, Any]],
) -> None:
    """
    A. Parity Test.

    Verifies that BertDetectorAdapter embedding pipeline matches
    raw HuggingFace AutoModel output with the same pooling strategy.
    Uses tiny-random-bert for speed (no large model download).
    """
    import torch
    from transformers import AutoModel, AutoTokenizer

    train_texts, _, _ = text_dataset
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Reference: raw HuggingFace pipeline
    tokenizer = AutoTokenizer.from_pretrained(TINY_MODEL)
    transformer = AutoModel.from_pretrained(TINY_MODEL).eval().to(device)

    inputs = tokenizer(
        train_texts,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        outputs = transformer(**inputs)

    # CLS pooling — default in BertDetectorAdapter
    reference_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()

    from sklearn.ensemble import IsolationForest as SklearnIF

    seed = 42
    sk_model = SklearnIF(n_estimators=100, random_state=seed, n_jobs=1)
    sk_model.fit(reference_embeddings)
    sk_scores = -sk_model.decision_function(reference_embeddings)

    our_model = get_detector(
        "BertDetector",
        model_name=TINY_MODEL,
        random_state=seed,
        detector_kwargs={"n_estimators": 100, "n_jobs": 1},
    )
    our_model.fit(train_texts)
    our_scores = our_model.predict_score(train_texts)

    np.testing.assert_allclose(
        our_scores,
        sk_scores,
        rtol=1e-5,
        err_msg=(
            "BertDetectorAdapter scores must match "
            "manual HuggingFace embedding + IForest pipeline"
        ),
    )


# --- B. Injection ---


def test_bert_param_injection(
    text_dataset: tuple[list[str], list[str], np.ndarray[Any, Any]],
) -> None:
    """
    B. Injection Test.

    Verifies that pooling, max_length and detector_kwargs
    are correctly passed to internal components.
    """
    train_texts, _, _ = text_dataset
    n_estimators = 17
    max_length = 128
    pooling = "mean"

    model = get_detector(
        "BertDetector",
        model_name=TINY_MODEL,
        pooling=pooling,
        max_length=max_length,
        detector_kwargs={"n_estimators": n_estimators},
    )

    assert model.pooling == pooling  # type: ignore[attr-defined]
    assert model.max_length == max_length  # type: ignore[attr-defined]

    model.fit(train_texts)

    assert model._detector.backend_model.n_estimators == n_estimators  # type: ignore[attr-defined]


# --- C. Determinism ---


def test_bert_determinism(
    text_dataset: tuple[list[str], list[str], np.ndarray[Any, Any]],
) -> None:
    """
    C. Determinism Test.

    Verifies that same random_state produces identical scores.
    BERT inference is deterministic — scores differ only due
    to the anomaly detector's random_state.
    Different random_state must produce different scores.
    """
    train_texts, _, _ = text_dataset

    def make_and_score(seed: int) -> np.ndarray[Any, Any]:
        model = get_detector(
            "BertDetector",
            model_name=TINY_MODEL,
            random_state=seed,
        )
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


def test_bert_pooling_affects_scores(
    text_dataset: tuple[list[str], list[str], np.ndarray[Any, Any]],
) -> None:
    """
    D. Domain Logic Test.

    Verifies that different pooling strategies produce different scores.
    CLS and mean pooling extract different information from transformer output.
    """
    train_texts, _, _ = text_dataset

    model_cls = get_detector(
        "BertDetector",
        model_name=TINY_MODEL,
        pooling="cls",
        random_state=42,
    )
    model_cls.fit(train_texts)

    model_mean = get_detector(
        "BertDetector",
        model_name=TINY_MODEL,
        pooling="mean",
        random_state=42,
    )
    model_mean.fit(train_texts)

    scores_cls = model_cls.predict_score(train_texts)
    scores_mean = model_mean.predict_score(train_texts)

    assert not np.allclose(
        scores_cls, scores_mean
    ), "CLS and mean pooling must produce different anomaly scores"


def test_bert_chunking_long_text(
    text_dataset: tuple[list[str], list[str], np.ndarray[Any, Any]],
) -> None:
    """
    D. Domain Logic Test.

    Verifies that chunking_strategy handles texts longer than max_length.
    Output shape must be (N_docs,) regardless of text length.
    """
    train_texts, _, _ = text_dataset

    model = get_detector(
        "BertDetector",
        model_name=TINY_MODEL,
        max_length=10,
        chunking_strategy="mean",
        random_state=42,
    )
    model.fit(train_texts)
    scores = model.predict_score(train_texts)

    assert scores.shape == (
        len(train_texts),
    ), f"Expected shape ({len(train_texts)},), got {scores.shape}"
    assert np.issubdtype(scores.dtype, np.number)


def test_bert_unknown_detector_raises() -> None:
    """
    D. Domain Logic Test.

    Verifies that unknown detector name raises ConfigError at init time.
    """
    with pytest.raises(ConfigError):
        get_detector(
            "BertDetector",
            model_name=TINY_MODEL,
            detector="NonExistentDetector",
        )


def test_bert_rejects_numeric_input(
    text_dataset: tuple[list[str], list[str], np.ndarray[Any, Any]],
) -> None:
    """
    D. Domain Logic Test.

    Verifies that numeric input is rejected with a clear error.
    """
    train_texts, _, _ = text_dataset
    model = get_detector("BertDetector", model_name=TINY_MODEL)
    model.fit(train_texts)

    with pytest.raises(DataFormatError):
        model.predict_score(np.array([[1.0, 2.0], [3.0, 4.0]]))


def test_bert_rejects_empty_strings() -> None:
    """
    D. Domain Logic Test.

    Verifies that empty/whitespace-only strings are rejected at fit time.
    """
    model = get_detector("BertDetector", model_name=TINY_MODEL)
    with pytest.raises(DataFormatError):
        model.fit(["valid text", "   ", "another valid text"])
