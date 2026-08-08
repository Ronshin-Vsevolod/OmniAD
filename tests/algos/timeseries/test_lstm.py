from typing import Any

import numpy as np
import numpy.typing as npt
import pytest

pytest.importorskip("torch")

from omniad import get_detector  # noqa: E402


def test_lstm_loss_decreases_over_epochs(timeseries_dataset) -> None:
    """A. Deep-learning parity substitute: the model must actually learn."""
    X_train, _ = timeseries_dataset
    losses: list[float] = []

    model = get_detector("LSTM", window_size=10, epochs=5, learning_rate=1e-2)
    original_step = model._train_step

    def recording_step(batch, m, criterion, optimizer):
        loss = original_step(batch, m, criterion, optimizer)
        losses.append(loss.item())
        return loss

    model._train_step = recording_step
    model.fit(X_train)

    chunk = max(len(losses) // 5, 1)
    assert np.mean(losses[-chunk:]) < np.mean(losses[:chunk])


def test_lstm_param_injection(timeseries_dataset) -> None:
    """B. Injection."""
    X_train, _ = timeseries_dataset
    model = get_detector("LSTM", window_size=7, hidden_dim=16, epochs=1).fit(X_train)
    assert model.model.lstm.hidden_size == 16
    assert model.window_size == 7


def test_lstm_determinism(timeseries_dataset) -> None:
    """C. Determinism."""
    X_train, X_test = timeseries_dataset

    def make_and_score(seed: int) -> npt.NDArray[Any]:
        model = get_detector(
            "LSTM", window_size=10, epochs=2, random_state=seed, device="cpu"
        )
        return model.fit(X_train).predict_score(X_test)

    np.testing.assert_allclose(make_and_score(0), make_and_score(0), rtol=1e-5)


def test_lstm_window_size_shrinks_output_length(timeseries_dataset) -> None:
    """D. Domain logic."""
    X_train, X_test = timeseries_dataset
    model = get_detector("LSTM", window_size=10, epochs=1).fit(X_train)
    scores = model.predict_score(X_test)
    assert len(scores) == len(X_test) - 10 + 1 < len(X_test)


def test_lstm_forecasting_mode_restricts_output_dim(timeseries_dataset) -> None:
    """D. Domain logic: target_cols switches from reconstruction to forecasting."""
    X_train, X_test = timeseries_dataset
    model = get_detector("LSTM", window_size=10, epochs=1, target_cols=[0]).fit(X_train)
    assert model.model.linear.out_features == 1
    scores = model.predict_score(X_test)
    assert scores.shape == (len(X_test) - 10 + 1,)
