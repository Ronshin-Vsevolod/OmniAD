"""
Native (non-omniad) baseline callables for overhead comparison.

Each function receives a **fitted** omniad model and input data *X*,
extracts the underlying backend, and returns a zero-argument callable
that produces anomaly scores by calling the backend directly —
bypassing omniad's validation and wrapper layer.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
import numpy.typing as npt

from omniad.utils.timeseries import create_windows

#  Tabular


def iforest_native(
    model: Any,
) -> Callable[[npt.NDArray[Any]], npt.NDArray[Any]]:
    backend = model.backend_model

    def predict(
        X: npt.NDArray[Any],
    ) -> npt.NDArray[Any]:
        return np.asarray(
            -backend.decision_function(X),
            dtype=np.float32,
        )

    return predict


#  Text


def tfidf_native(
    model: Any,
) -> Callable[[list[str]], npt.NDArray[Any]]:
    vec = model._vectorizer
    backend = model.backend_model

    def predict(
        X: list[str],
    ) -> npt.NDArray[Any]:
        X_tfidf = vec.transform(X)

        return np.asarray(
            -backend.decision_function(X_tfidf),
            dtype=np.float32,
        )

    return predict


#  Time-series


def lstm_native(
    model: Any,
) -> Callable[[npt.NDArray[Any]], npt.NDArray[Any]]:
    """Direct PyTorch LSTM inference using omniad backend."""
    import torch

    backend = model.backend_model
    window_size = getattr(model, "window_size", 50)
    batch_size = getattr(model, "batch_size", 32)
    target_cols = getattr(model, "target_cols", None)
    device = next(backend.parameters()).device

    def predict(X: npt.NDArray[Any]) -> npt.NDArray[Any]:
        X = np.asarray(X, dtype=np.float32)
        windows = create_windows(X, window_size).astype(np.float32)
        X_tensor = torch.from_numpy(windows)

        parts: list[npt.NDArray[Any]] = []

        with torch.inference_mode():
            for i in range(0, len(windows), batch_size):
                batch_x = X_tensor[i : i + batch_size].to(device)
                output = backend(batch_x)

                if target_cols is not None:
                    target = batch_x[:, -1, target_cols]
                else:
                    target = batch_x[:, -1, :]

                mse = ((target - output) ** 2).mean(dim=1)
                parts.append(mse.cpu().numpy())

        return np.concatenate(parts)

    return predict


#  Computer vision


def conv_ae_native(
    model: Any,
) -> Callable[[npt.NDArray[Any]], npt.NDArray[Any]]:
    """Direct PyTorch ConvAutoencoder inference using omniad's own backend."""
    import torch

    backend = model.backend_model
    batch_size: int = getattr(model, "batch_size", 32)
    device = next(backend.parameters()).device

    def predict(X: npt.NDArray[Any]) -> npt.NDArray[Any]:
        parts: list[npt.NDArray[Any]] = []
        with torch.no_grad():
            for i in range(0, len(X), batch_size):
                b = torch.from_numpy(X[i : i + batch_size]).to(device)
                mse = ((b - backend(b)) ** 2).mean(dim=(1, 2, 3))
                parts.append(mse.cpu().numpy())
        return np.concatenate(parts)

    return predict
