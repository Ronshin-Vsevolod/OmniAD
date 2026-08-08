"""
Confirms each Layer 1.5 template actually routes its backend call
through backend_boundary — test_backend_boundary.py only proves the
mechanism works in the abstract, via DummyDetector.
"""
import pytest

from omniad import get_detector
from omniad.core.exceptions import BackendError
from tests.support import require_algo


def test_sklearn_template_translates_backend_errors(
    random_xy_dataset, monkeypatch
) -> None:
    X_train, _, _ = random_xy_dataset
    model = get_detector("IsolationForest").fit(X_train)
    monkeypatch.setattr(
        model.backend_model,
        "decision_function",
        lambda X: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    with pytest.raises(BackendError, match="predict_score"):
        model.predict_score(X_train)


def test_torch_template_translates_backend_errors(
    timeseries_dataset, monkeypatch
) -> None:
    require_algo("LSTM")
    X_train, _ = timeseries_dataset
    model = get_detector("LSTM", epochs=1, window_size=5).fit(X_train)
    monkeypatch.setattr(
        model.model,
        "forward",
        lambda x: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    with pytest.raises(BackendError, match="predict_score"):
        model.predict_score(X_train)


def test_river_template_translates_backend_errors(
    random_xy_dataset, monkeypatch
) -> None:
    require_algo("HalfSpaceTrees")
    X_train, _, _ = random_xy_dataset
    model = get_detector("HalfSpaceTrees").fit(X_train)
    monkeypatch.setattr(
        model.backend_model,
        "score_one",
        lambda x: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    with pytest.raises(BackendError, match="predict_score"):
        model.predict_score(X_train)


def test_transformers_template_translates_backend_errors(
    text_dataset, monkeypatch
) -> None:
    require_algo("BertDetector")
    train, _, _ = text_dataset
    model = get_detector("BertDetector", preset="debug").fit(train)
    monkeypatch.setattr(
        model,
        "_score_embeddings",
        lambda emb: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    with pytest.raises(BackendError, match="predict_score"):
        model.predict_score(train)
