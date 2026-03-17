from typing import Any

import numpy as np
import numpy.typing as npt
import pytest

from omniad import get_detector
from omniad.core.mixins import ReconstructionMixin, SegmentationMixin

torch = pytest.importorskip("torch", reason="torch not installed")


# --- A. Parity ---


def test_conv_ae_shapes(
    image_dataset: tuple[npt.NDArray[Any], npt.NDArray[Any]],
) -> None:
    """
    A. Parity: Output shapes must match contract.

    predict_score → (N,)
    predict_map → (N, H, W)
    predict_expected → (N, C, H, W)
    """
    X_train, X_test = image_dataset
    N, C, H, W = X_test.shape

    model = get_detector("ConvAutoencoder", preset="debug")
    model.fit(X_train)

    scores = model.predict_score(X_test)
    assert scores.shape == (N,)
    assert np.all(scores >= 0), "MSE scores must be non-negative"

    assert isinstance(model, SegmentationMixin)
    heatmaps = model.predict_map(X_test)
    assert heatmaps.shape == (N, H, W)

    assert isinstance(model, ReconstructionMixin)
    reconstructed = model.predict_expected(X_test)
    assert reconstructed.shape == (N, C, H, W)


def test_conv_ae_loss_decreases(
    image_dataset: tuple[npt.NDArray[Any], npt.NDArray[Any]],
) -> None:
    """
    A. Parity: Training loss must decrease over epochs.

    Verifies that the model actually learns on synthetic data.
    """
    X_train, _ = image_dataset

    model = get_detector(
        "ConvAutoencoder",
        epochs=1,
        hidden_dim=8,
        verbose=0,
    )
    model.fit(X_train)
    score_1epoch = model.predict_score(X_train).mean()

    model2 = get_detector(
        "ConvAutoencoder",
        epochs=20,
        hidden_dim=8,
        random_state=42,
        verbose=0,
    )
    model2.fit(X_train)
    score_20epoch = model2.predict_score(X_train).mean()

    assert (
        score_20epoch < score_1epoch
    ), "Mean reconstruction error should decrease with more training"


# --- B. Injection (Parameter passthrough) ---


def test_conv_ae_param_injection(
    image_dataset: tuple[npt.NDArray[Any], npt.NDArray[Any]],
) -> None:
    """
    B. Injection: Constructor params must reach the backend model.
    """
    X_train, _ = image_dataset
    hidden_dim = 16

    model = get_detector(
        "ConvAutoencoder",
        hidden_dim=hidden_dim,
        epochs=1,
        verbose=0,
    )
    model.fit(X_train)

    assert model.hidden_dim == hidden_dim  # type: ignore[attr-defined]
    # Verify bottleneck dimension in the actual model
    backend = model.backend_model
    # Last encoder conv should output hidden_dim channels
    encoder_out_channels = list(backend.encoder.children())[-2].out_channels
    assert encoder_out_channels == hidden_dim


def test_conv_ae_custom_model_fn(
    image_dataset: tuple[npt.NDArray[Any], npt.NDArray[Any]],
) -> None:
    """
    B. Injection: model_fn must be used when provided.
    """
    X_train, _ = image_dataset

    class TinyAE(torch.nn.Module):  # type: ignore[misc, name-defined]
        def __init__(self, ch: int) -> None:
            super().__init__()
            self.conv = torch.nn.Conv2d(ch, ch, 1)

        def forward(self, x: Any) -> Any:
            return self.conv(x)

    model = get_detector(
        "ConvAutoencoder",
        model_fn=TinyAE,
        epochs=1,
        verbose=0,
    )
    model.fit(X_train)

    assert isinstance(model.backend_model, TinyAE)


# --- C. Determinism ---


@pytest.mark.deterministic  # type: ignore[misc]
def test_conv_ae_determinism(
    image_dataset: tuple[npt.NDArray[Any], npt.NDArray[Any]],
) -> None:
    """
    C. Determinism: Same random_state → identical scores.
    """
    X_train, _ = image_dataset

    def make_and_score(seed: int) -> npt.NDArray[Any]:
        model = get_detector(
            "ConvAutoencoder",
            epochs=2,
            hidden_dim=8,
            random_state=seed,
            verbose=0,
        )
        model.fit(X_train)
        return model.predict_score(X_train)

    scores_a = make_and_score(42)
    scores_b = make_and_score(42)
    scores_c = make_and_score(99)

    np.testing.assert_allclose(
        scores_a,
        scores_b,
        rtol=1e-5,
        err_msg="Same random_state must produce identical scores",
    )
    assert not np.allclose(
        scores_a, scores_c
    ), "Different random_state should produce different scores"


# --- D. Domain Logic ---


def test_conv_ae_grayscale() -> None:
    """
    D. Domain Logic: Must work with single-channel (grayscale) images.
    """
    X = np.random.rand(8, 1, 32, 32).astype(np.float32)

    model = get_detector("ConvAutoencoder", epochs=1, verbose=0)
    model.fit(X)

    scores = model.predict_score(X)
    assert scores.shape == (8,)

    heatmaps = model.predict_map(X)  # type: ignore[attr-defined]
    assert heatmaps.shape == (8, 32, 32)


def test_conv_ae_rejects_invalid_input() -> None:
    """
    D. Domain Logic: Non-image input must be rejected.
    """
    from omniad.core.exceptions import DataFormatError

    model = get_detector("ConvAutoencoder", epochs=1)

    # 2D tabular data
    with pytest.raises(DataFormatError):
        model.fit(np.random.rand(10, 5))

    # 3D timeseries
    with pytest.raises(DataFormatError):
        model.fit(np.random.rand(10, 5, 3))

    # Wrong channel count
    with pytest.raises(DataFormatError):
        model.fit(np.random.rand(10, 5, 32, 32).astype(np.float32))


def test_conv_ae_uint8_input() -> None:
    """
    D. Domain Logic: uint8 images [0, 255] must be auto-normalized to [0, 1].
    """
    X = (np.random.rand(8, 3, 32, 32) * 255).astype(np.uint8)

    model = get_detector("ConvAutoencoder", epochs=1, verbose=0)
    model.fit(X)  # should not crash

    scores = model.predict_score(X)
    assert scores.shape == (8,)
