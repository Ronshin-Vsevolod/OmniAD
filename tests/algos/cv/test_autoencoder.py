import numpy as np
import pytest

pytest.importorskip("torch")

from omniad import get_detector  # noqa: E402
from tests.support import require_algo  # noqa: E402

ALGO = "ConvAutoencoder"


def test_autoencoder_loss_decreases_over_epochs(image_dataset) -> None:
    """A. Deep-learning parity substitute."""
    require_algo(ALGO)
    X_train, _ = image_dataset
    losses: list[float] = []

    model = get_detector(ALGO, epochs=5, hidden_dim=8, learning_rate=1e-2)
    original_step = model._train_step

    def recording_step(batch, m, criterion, optimizer):
        loss = original_step(batch, m, criterion, optimizer)
        losses.append(loss.item())
        return loss

    model._train_step = recording_step
    model.fit(X_train)

    chunk = max(len(losses) // 5, 1)
    assert np.mean(losses[-chunk:]) < np.mean(losses[:chunk])


def test_autoencoder_param_injection(image_dataset) -> None:
    """B. Injection."""
    require_algo(ALGO)
    X_train, _ = image_dataset
    model = get_detector(ALGO, hidden_dim=8, epochs=1).fit(X_train)
    assert model.model.encoder[2].out_channels == 8


def test_autoencoder_determinism(image_dataset) -> None:
    """C. Determinism."""
    require_algo(ALGO)
    X_train, X_test = image_dataset

    def make_and_score(seed: int) -> np.ndarray:
        model = get_detector(ALGO, epochs=1, hidden_dim=8, random_state=seed)
        return model.fit(X_train).predict_score(X_test)

    np.testing.assert_allclose(make_and_score(0), make_and_score(0), rtol=1e-5)


def test_autoencoder_predict_map_shape(image_dataset) -> None:
    """D. Domain logic: pixel-level map is (N, H, W), not (N, C, H, W)."""
    require_algo(ALGO)
    X_train, X_test = image_dataset
    model = get_detector(ALGO, epochs=1, hidden_dim=8).fit(X_train)
    anomaly_map = model.predict_map(X_test)
    assert anomaly_map.shape == (len(X_test), X_test.shape[2], X_test.shape[3])


def test_autoencoder_predict_expected_matches_input_shape(image_dataset) -> None:
    """D. Domain logic."""
    require_algo(ALGO)
    X_train, X_test = image_dataset
    model = get_detector(ALGO, epochs=1, hidden_dim=8).fit(X_train)
    assert model.predict_expected(X_test).shape == X_test.shape


def test_autoencoder_custom_model_fn_is_used(image_dataset) -> None:
    """D. Domain logic: model_fn overrides the default architecture."""
    require_algo(ALGO)
    import torch.nn as nn

    X_train, _ = image_dataset
    built: dict = {}

    def tiny_shape_preserving_model(channels: int) -> nn.Module:
        built["called_with"] = channels
        return nn.Conv2d(channels, channels, kernel_size=1)

    get_detector(ALGO, model_fn=tiny_shape_preserving_model, epochs=1).fit(X_train)
    assert built["called_with"] == X_train.shape[1]
