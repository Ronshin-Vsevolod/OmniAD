from __future__ import annotations

import logging
from typing import Any, Callable, cast

import numpy as np
import numpy.typing as npt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from omniad.core.adapters.torch_adapter import BaseTorchAdapter
from omniad.core.exceptions import ConfigError
from omniad.core.mixins import ReconstructionMixin
from omniad.utils.timeseries import create_windows

logger = logging.getLogger(__name__)


class LSTMModel(nn.Module):  # type: ignore[misc]
    """
    Simple LSTM architecture.

    Parameters
    ----------
    input_dim : int
        Dimensionality of the input features per time step.
    hidden_dim : int
        Number of features in the hidden state of the LSTM.
    output_dim : int
        Dimensionality of the output layer.
    """

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        # batch_first=True means input is (batch, seq, feature)
        self.lstm = nn.LSTM(
            input_size=input_dim, hidden_size=hidden_dim, batch_first=True
        )
        self.linear = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # lstm_out: (batch_size, sequence_length, hidden_dim)
        # hidden state is ignored
        lstm_out, _ = self.lstm(x)
        # We take the output of the last time step for prediction
        # last_step: (batch_size, hidden_dim)
        last_step = lstm_out[:, -1, :]
        # Returns: (batch_size, output_dim)
        return self.linear(last_step)


class LSTMAdapter(BaseTorchAdapter, ReconstructionMixin):
    """
    LSTM-based Anomaly Detector for Time Series.

    Supports two modes:
    1. Reconstruction (default): Learns to reconstruct the input window.
    2. Forecasting: Uses input window to predict specific target columns.

     Parameters
    ----------
    window_size : int, default=10
        Length of the input sequence (look-back window).
    hidden_dim : int, default=64
        Hidden dimension of LSTM layer.
    target_cols : list[int] | None, default=None
        Indices of columns to predict. If None, reconstructs all columns.
    score_metric : str, default='mse'
        Metric used to calculate anomaly score.
    **kwargs : Any
        Standard arguments for BaseTorchAdapter (batch_size, epochs, lr, device).
    """

    def __init__(
        self,
        window_size: int = 10,
        hidden_dim: int = 64,
        target_cols: list[int] | None = None,
        score_metric: str | Callable[[Any, Any], Any] = "mse",
        **kwargs: Any,
    ) -> None:
        super().__init__(score_metric=score_metric, **kwargs)
        self.window_size = window_size
        self.hidden_dim = hidden_dim
        self.target_cols = target_cols

    def _build_model(self, input_dim: int) -> nn.Module:
        """Builds the LSTM network."""
        output_dim: int
        if self.target_cols is not None:
            output_dim = len(self.target_cols)
        else:
            output_dim = input_dim

        return LSTMModel(input_dim, self.hidden_dim, output_dim)

    def _prepare_data(self, X: npt.NDArray[Any]) -> npt.NDArray[Any]:
        """Convert input to 3D windows if needed."""
        # If already 3D (batch_size, sequence_length, features)
        if X.ndim == 3:
            return X

        # If 2D (time_steps, features), create windows
        return create_windows(X, self.window_size)

    # --- Overrides to handle Time-Series Logic ---

    def _fit_backend(self, X: Any, y: Any | None = None) -> None:
        """Override to handle window creation before training."""
        # We manually prepare windows here because BaseTorchAdapter expects
        # data ready for TensorDataset.
        X_windows = self._prepare_data(X)
        logger.debug("Windows: %s -> %s", X.shape, X_windows.shape)
        super()._fit_backend(X_windows, y)

    def predict_score(self, X: Any) -> npt.NDArray[Any]:
        """Override to handle window creation before prediction."""
        self._check_torch()
        if self.model is None or self.device is None:
            raise ConfigError("Model not initialized. Call fit() first.")

        model = self.model
        device = self.device

        X = self._validate(X)
        X_windows = self._prepare_data(X).astype(np.float32)

        logger.debug("predict_score: windows=%d", len(X_windows))

        dataset = TensorDataset(torch.from_numpy(X_windows))
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

        model.eval()
        scores_list: list[npt.NDArray[Any]] = []

        with torch.no_grad():
            for (batch_x,) in loader:
                batch_x = batch_x.to(device)
                output = model(batch_x)

                if self.target_cols is not None:
                    target = batch_x[:, -1, self.target_cols]
                else:
                    target = batch_x[:, -1, :]

                batch_scores = self.score_func(target, output)
                scores_list.append(batch_scores.cpu().numpy())

        return cast("npt.NDArray[Any]", np.concatenate(scores_list))

    def _train_step(
        self,
        batch: tuple[torch.Tensor, ...],
        model: nn.Module,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
    ) -> torch.Tensor:
        """Custom training step dealing with Targets."""
        x = batch[0].to(self.device)

        output = model(x)

        target: torch.Tensor
        if self.target_cols is not None:
            target = x[:, -1, self.target_cols]
        else:
            target = x[:, -1, :]

        loss = criterion(output, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return loss

    # --- Mixin Implementation ---

    def predict_expected(self, X: Any) -> npt.NDArray[Any]:
        """
        Return the model's prediction/reconstruction.
        """
        self._check_torch()
        if self.model is None or self.device is None:
            raise ConfigError("Model not initialized. Call fit() first.")

        # Local variables for MyPy narrowing
        model = self.model
        device = self.device

        X_windows = self._prepare_data(X).astype(np.float32)

        dataset = TensorDataset(torch.from_numpy(X_windows))
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

        model.eval()
        results: list[npt.NDArray[Any]] = []

        with torch.no_grad():
            for (batch_x,) in loader:
                batch_x = batch_x.to(device)
                output = model(batch_x)
                results.append(output.cpu().numpy())

        return cast("npt.NDArray[Any]", np.concatenate(results))
