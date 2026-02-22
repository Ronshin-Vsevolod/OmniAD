from typing import Any

import numpy as np
import pytest

from omniad import get_detector

torch = pytest.importorskip("torch", reason="torch not installed")


def test_lstm_reconstruction_shape(
    timeseries_dataset: tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]],
) -> None:
    """
    A. Domain Logic:
                     Reconstruction mode (target_cols=None).

    Output length must equal N - window_size + 1.
    predict_expected shape must equal (N - window_size + 1, n_features).
    """
    X_train, _ = timeseries_dataset
    N, n_features = X_train.shape  # 100, 2
    window_size = 10
    expected_len = N - window_size + 1  # 91

    model = get_detector(
        "LSTM",
        window_size=window_size,
        hidden_dim=4,
        epochs=1,
        random_state=42,
        verbose=False,
    )
    model.fit(X_train)

    scores = model.predict_score(X_train)
    assert scores.shape == (
        expected_len,
    ), f"Expected scores shape ({expected_len},), got {scores.shape}"

    reconstruction = model.predict_expected(X_train)  # type: ignore[attr-defined]
    assert reconstruction.shape == (expected_len, n_features), (
        f"Expected reconstruction shape ({expected_len}, {n_features}), "
        f"got {reconstruction.shape}"
    )


def test_lstm_forecasting_shape(
    timeseries_dataset: tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]],
) -> None:
    """
    A. Domain Logic:
                     Forecasting mode (target_cols specified).

    Output layer size must equal len(target_cols).
    predict_expected shape must equal (N - window_size + 1, len(target_cols)).
    """
    X_train, _ = timeseries_dataset
    N = X_train.shape[0]  # 100
    window_size = 10
    target_cols = [0, 1]
    expected_len = N - window_size + 1  # 91
    n_targets = len(target_cols)  # 2

    model = get_detector(
        "LSTM",
        window_size=window_size,
        hidden_dim=8,
        target_cols=target_cols,
        epochs=1,
        random_state=42,
        verbose=False,
    )
    model.fit(X_train)

    assert model.model.linear.out_features == n_targets, (  # type: ignore[attr-defined]
        f"Expected linear.out_features={n_targets}, "
        f"got {model.model.linear.out_features}"  # type: ignore[attr-defined]
    )

    forecast = model.predict_expected(X_train)  # type: ignore[attr-defined]
    assert forecast.shape == (expected_len, n_targets), (
        f"Expected forecast shape ({expected_len}, {n_targets}), "
        f"got {forecast.shape}"
    )


@pytest.mark.deterministic  # type: ignore[misc]
def test_lstm_determinism(
    timeseries_dataset: tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]],
) -> None:
    """
    B. Determinism:
                    Same random_state must produce identical scores.
    Different random_state must produce different scores.
    """
    X_train, _ = timeseries_dataset

    def make_and_score(seed: int) -> np.ndarray[Any, Any]:
        model = get_detector(
            "LSTM",
            window_size=5,
            hidden_dim=4,
            epochs=2,
            random_state=seed,
            verbose=False,
        )
        model.fit(X_train)
        return model.predict_score(X_train)

    scores_a = make_and_score(seed=42)
    scores_b = make_and_score(seed=42)
    scores_c = make_and_score(seed=99)

    np.testing.assert_allclose(
        scores_a,
        scores_b,
        rtol=1e-6,
        err_msg="Same random_state must produce identical scores",
    )
    assert not np.allclose(
        scores_a, scores_c
    ), "Different random_state should produce different scores"


@pytest.mark.deterministic  # type: ignore[misc]
def test_lstm_metric_affects_scores(
    timeseries_dataset: tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]],
) -> None:
    """
    C. Metric:
                Switching score_metric must change anomaly scores.

    Both models share the same trained weights to isolate
    the effect of the metric from weight randomness.
    """
    X_train, _ = timeseries_dataset

    model_mse = get_detector(
        "LSTM",
        score_metric="mse",
        window_size=5,
        hidden_dim=4,
        epochs=1,
        random_state=42,
        verbose=False,
    )
    model_mse.fit(X_train)

    model_mae = get_detector(
        "LSTM",
        score_metric="mae",
        window_size=5,
        hidden_dim=4,
        epochs=1,
        random_state=42,
        verbose=False,
    )
    model_mae.fit(X_train)

    # Share trained weights to isolate metric effect from weight randomness
    model_mae._backend_model = model_mse._backend_model
    model_mae.model = model_mse.model  # type: ignore[attr-defined]

    scores_mse = model_mse.predict_score(X_train)
    scores_mae = model_mae.predict_score(X_train)

    assert not np.allclose(
        scores_mse, scores_mae
    ), "MSE and MAE metrics must produce different anomaly scores"
