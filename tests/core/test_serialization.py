import json
import zipfile
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pytest

from omniad.core.exceptions import ConfigError
from omniad.core.metrics import register_metric
from tests.doubles import DummyDetector

X = np.random.default_rng(2).normal(size=(40, 3))


def _load_attributes(path: Path, tmp_path: Path) -> dict[str, Any]:
    with zipfile.ZipFile(path) as zf:
        zf.extract("attributes.pkl", tmp_path)
    return joblib.load(tmp_path / "attributes.pkl")


def test_transient_core_state_is_not_serialized(tmp_path) -> None:
    """Backend references and temporary fit state are not pickled."""
    model = DummyDetector().fit(X)
    model._cached_train_scores = np.ones(len(X))

    path = tmp_path / "model.zip"
    model.save(str(path))

    state = _load_attributes(path, tmp_path)

    assert "_backend_model" not in state
    assert "_cached_train_scores" not in state


def test_zip_contains_expected_entries(tmp_path) -> None:
    model = DummyDetector().fit(X)
    path = tmp_path / "model.zip"
    model.save(str(path))

    with zipfile.ZipFile(path) as zf:
        names = set(zf.namelist())
    assert "metadata.json" in names
    assert "attributes.pkl" in names
    assert any(n.startswith("backend/") for n in names)


def test_metadata_fields(tmp_path) -> None:
    model = DummyDetector(contamination=0.2).fit(X)
    path = tmp_path / "model.zip"
    model.save(str(path))

    with zipfile.ZipFile(path) as zf:
        meta = json.loads(zf.read("metadata.json"))
    assert meta["class_name"] == "DummyDetector"
    assert meta["contamination"] == 0.2
    assert meta["threshold"] == pytest.approx(model.threshold_)


def test_load_without_zip_extension(tmp_path) -> None:
    model = DummyDetector().fit(X)
    base = tmp_path / "model"
    model.save(str(base))

    loaded = DummyDetector().load(str(base))  # no ".zip" suffix passed
    np.testing.assert_allclose(loaded.predict_score(X), model.predict_score(X))


def test_scores_and_threshold_survive_roundtrip(tmp_path) -> None:
    model = DummyDetector().fit(X)
    path = tmp_path / "model.zip"
    model.save(str(path))

    loaded = DummyDetector().load(str(path))
    np.testing.assert_allclose(loaded.predict_score(X), model.predict_score(X))
    assert loaded.threshold_ == model.threshold_


def test_unregistered_custom_metric_raises_on_save(tmp_path) -> None:
    model = DummyDetector()
    model.score_metric = lambda a, b: 0.0  # never registered
    model.fit(X)
    with pytest.raises(ConfigError):
        model.save(str(tmp_path / "model.zip"))


def test_unregistered_custom_threshold_raises_on_save(tmp_path) -> None:
    def custom_threshold(scores, contamination):
        return float(np.median(scores))

    model = DummyDetector(threshold_strategy=custom_threshold).fit(X)

    with pytest.raises(ConfigError):
        model.save(str(tmp_path / "model.zip"))


def test_registered_custom_metric_survives_roundtrip(tmp_path) -> None:
    def my_metric(a, b):
        return float(np.mean(np.abs(a - b)))

    register_metric("my_metric_for_serialization_test", my_metric)
    model = DummyDetector()
    model.score_metric = my_metric
    model.fit(X)

    path = tmp_path / "model.zip"
    model.save(str(path))
    loaded = DummyDetector().load(str(path))
    assert loaded.score_metric == "my_metric_for_serialization_test"


def test_registered_custom_threshold_survives_roundtrip(tmp_path) -> None:
    from omniad.utils.thresholds import register_threshold

    def custom_threshold(scores, contamination):
        return float(np.median(scores))

    register_threshold("serialization_threshold_test", custom_threshold)

    model = DummyDetector(threshold_strategy=custom_threshold).fit(X)
    path = tmp_path / "model.zip"
    model.save(str(path))

    loaded = DummyDetector().load(str(path))

    assert loaded.threshold_strategy == "serialization_threshold_test"
    assert loaded.threshold_ == pytest.approx(model.threshold_)
