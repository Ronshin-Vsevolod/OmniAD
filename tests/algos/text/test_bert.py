import zipfile

import joblib
import numpy as np
import pytest

from omniad import get_detector
from omniad.core.exceptions import ConfigError
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


def test_bert_unknown_detector_raises_config_error() -> None:
    """B. Injection/config guard: fails before any tokenizer/model is
    touched, so this needs neither `transformers` nor `torch`."""
    require_algo(ALGO)

    with pytest.raises(ConfigError):
        get_detector("BertDetector", detector="NotARealDetector")


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


def test_bert_reports_no_segmentation_or_incremental_learning() -> None:
    """E."""
    require_algo(ALGO)
    model = get_detector(ALGO, preset="debug")
    assert model.capabilities["segmentation"] is False
    assert model.capabilities["incremental_learning"] is False


def test_bert_class_capabilities_match_default_detector() -> None:
    """E. class discovery uses the default inner detector."""
    from omniad.algos.text.bert import BertDetectorAdapter

    assert "feature_importance" in BertDetectorAdapter.get_capabilities()


def test_bert_save_weights_controls_transformer_persistence(
    text_dataset, tmp_path
) -> None:
    """E. save_weights controls transformer weight persistence."""
    require_algo(ALGO)
    train, _, _ = text_dataset

    for save_weights in (False, True):
        model = get_detector(
            ALGO,
            preset="debug",
            save_weights=save_weights,
        ).fit(train)

        path = tmp_path / f"bert_{save_weights}.zip"
        model.save(str(path))

        extract_dir = tmp_path / f"extracted_{save_weights}"
        with zipfile.ZipFile(path) as zf:
            names = set(zf.namelist())
            zf.extract("attributes.pkl", extract_dir)

        state = joblib.load(extract_dir / "attributes.pkl")

        assert ("backend/transformer_weights.pt" in names) is save_weights
        assert "_transformer" not in state
        assert "_tokenizer" not in state
        assert "_detector" not in state


def test_bert_rejects_unregistered_pooling_on_save(text_dataset, tmp_path) -> None:
    """E. custom pooling must be registered before saving."""
    require_algo(ALGO)
    train, _, _ = text_dataset

    def custom_pooling(hidden, mask):
        return hidden[:, 0, :]

    model = get_detector(
        ALGO,
        preset="debug",
        pooling=custom_pooling,
    ).fit(train)

    with pytest.raises(ConfigError, match="unregistered pooling"):
        model.save(str(tmp_path / "model.zip"))


def test_bert_rejects_unregistered_chunking_on_save(text_dataset, tmp_path) -> None:
    """E. custom chunking must be registered before saving."""
    require_algo(ALGO)
    train, _, _ = text_dataset

    def custom_chunking(chunks):
        return chunks.mean(axis=0)

    model = get_detector(
        ALGO,
        preset="debug",
        chunking_strategy=custom_chunking,
    ).fit(train)

    with pytest.raises(ConfigError, match="unregistered chunking"):
        model.save(str(tmp_path / "model.zip"))


def test_bert_registered_text_strategies_survive_roundtrip(
    text_dataset, tmp_path
) -> None:
    """E. registered text strategies are restored by name."""
    from omniad.utils.text import register_chunking_strategy, register_pooling

    require_algo(ALGO)
    train, test, _ = text_dataset

    def custom_pooling(hidden, mask):
        return hidden[:, 0, :]

    def custom_chunking(chunks):
        return chunks.mean(axis=0)

    register_pooling("serialization_pooling_test", custom_pooling)
    register_chunking_strategy(
        "serialization_chunking_test",
        custom_chunking,
    )

    model = get_detector(
        ALGO,
        preset="debug",
        pooling=custom_pooling,
        chunking_strategy=custom_chunking,
    ).fit(train)

    before = model.predict_score(test)
    path = tmp_path / "model.zip"
    model.save(str(path))

    loaded = get_detector(ALGO).load(str(path))

    assert loaded.pooling == "serialization_pooling_test"
    assert loaded.chunking_strategy == "serialization_chunking_test"
    np.testing.assert_allclose(
        loaded.predict_score(test),
        before,
        rtol=1e-5,
    )
