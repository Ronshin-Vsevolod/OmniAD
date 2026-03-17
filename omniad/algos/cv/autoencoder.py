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
from omniad.core.mixins import ReconstructionMixin, SegmentationMixin
from omniad.utils.validation import validate_image

logger = logging.getLogger(__name__)


class ConvAutoencoderModel(nn.Module):  # type: ignore[misc]
    """
    Symmetric convolutional autoencoder.

    Uses strided convolutions (no pooling) for downsampling
    and transposed convolutions for upsampling.
    Fully convolutional — works with any H, W divisible by 4.

    Parameters
    ----------
    in_channels : int
        Number of input channels (1 for grayscale, 3 for RGB).
    hidden_dim : int
        Number of filters in the bottleneck layer.
    """

    def __init__(self, in_channels: int, hidden_dim: int = 32) -> None:
        super().__init__()

        # Encoder: C -> hidden_dim/2 -> hidden_dim
        mid = max(hidden_dim // 2, 4)

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, mid, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, hidden_dim, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )

        # Decoder: hidden_dim -> hidden_dim/2 -> C
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(
                hidden_dim,
                mid,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1,
            ),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(
                mid,
                in_channels,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1,
            ),
            nn.Sigmoid(),  # output in [0, 1]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: encode then decode."""
        z = self.encoder(x)
        return self.decoder(z)


class ConvAutoencoderAdapter(BaseTorchAdapter, ReconstructionMixin, SegmentationMixin):
    """
    Convolutional Autoencoder for visual anomaly detection.

    Learns to reconstruct normal images. Anomalies are detected
    via high reconstruction error.

    Provides three output modes:
    - ''predict_score(X)'' -> image-level anomaly scores (N,)
    - ''predict_map(X)''   -> pixel-level anomaly maps (N, H, W)
    - ''predict_expected(X)'' -> reconstructed images (N, C, H, W)

    Parameters
    ----------
    hidden_dim : int, default=32
        Number of filters in the bottleneck layer.
    epochs : int, default=50
        Number of training epochs.
    batch_size : int, default=32
        Batch size for training and inference.
    learning_rate : float, default=1e-3
        Learning rate for Adam optimizer.
    device : str, default="auto"
        Device: "auto", "cpu", or "cuda".
    contamination : float, default=0.1
        Expected proportion of anomalies.
    verbose : int, default=0
        Verbosity level.

    Examples
    --------
    >>> from omniad import get_detector
    >>> import numpy as np
    >>> X = np.random.rand(100, 3, 64, 64).astype(np.float32)
    >>> model = get_detector("ConvAutoencoder", epochs=10)
    >>> model.fit(X)
    >>> scores = model.predict_score(X)       # (100,)
    >>> heatmaps = model.predict_map(X)       # (100, 64, 64)
    >>> reconstructed = model.predict_expected(X)  # (100, 3, 64, 64)

    >>> def my_unet(channels):
    >>> return MyUNet(in_ch=channels, depth=4)=
    >>> model = get_detector("ConvAutoencoder", model_fn=my_unet)
    """

    def __init__(
        self,
        model_fn: Callable[[int], nn.Module] | None = None,
        hidden_dim: int = 32,
        epochs: int = 50,
        batch_size: int = 32,
        learning_rate: float = 1e-3,
        device: str = "auto",
        contamination: float = 0.1,
        verbose: int = 0,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            device=device,
            contamination=contamination,
            verbose=verbose,
            **kwargs,
        )
        self.model_fn = model_fn
        self.hidden_dim = hidden_dim

    # --- Domain-specific overrides ---

    def _validate(self, X: Any) -> Any:
        """CV-specific: validate (N, C, H, W) images."""
        return validate_image(X)

    def _extract_input_dim(self, X: Any) -> int:
        """For CV: input dimension = number of channels."""
        return int(X.shape[1])

    def _build_model(self, input_dim: int) -> nn.Module:
        """
        Build model. input_dim = in_channels for CV.

        Uses model_fn if provided, otherwise default ConvAutoencoderModel.
        Contract: forward(x) -> x_reconstructed, output.shape == input.shape.
        """
        if self.model_fn is not None:
            logger.debug("Using custom model_fn: %s", self.model_fn)
            return self.model_fn(input_dim)

        return ConvAutoencoderModel(
            in_channels=input_dim,
            hidden_dim=self.hidden_dim,
        )

    def _compute_anomaly_score(
        self, x: torch.Tensor, output: torch.Tensor
    ) -> torch.Tensor:
        """Image-level: error averaged over (C, H, W) -> (B,)."""
        batch_size = x.shape[0]
        return ((x - output) ** 2).reshape(batch_size, -1).mean(dim=1)

    def _compute_pixel_error(
        self, x: torch.Tensor, output: torch.Tensor
    ) -> torch.Tensor:
        """Pixel-level: error averaged over C only -> (B, H, W)."""
        return ((x - output) ** 2).mean(dim=1)

    # --- SegmentationMixin ---

    def predict_map(self, X: Any) -> npt.NDArray[Any]:
        """
        Predict pixel-level anomaly map.

        Parameters
        ----------
        X : np.ndarray of shape (N, C, H, W)
            Input images.

        Returns
        -------
        anomaly_map : np.ndarray of shape (N, H, W)
            Per-pixel reconstruction error (MSE over channels).
        """
        self._check_torch()
        if self.model is None or self.device is None:
            raise ConfigError("Model not initialized. Call fit() first.")

        X = self._validate(X)
        dataset = TensorDataset(torch.from_numpy(X))
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

        self.model.eval()
        maps: list[npt.NDArray[Any]] = []

        with torch.no_grad():
            for (batch_x,) in loader:
                batch_x = batch_x.to(self.device)
                output = self.model(batch_x)

                # MSE over channels only -> (B, H, W)
                pixel_error = self._compute_pixel_error(batch_x, output)
                maps.append(pixel_error.cpu().numpy())

        return cast("npt.NDArray[Any]", np.concatenate(maps, axis=0))

    # --- ReconstructionMixin ---

    def predict_expected(self, X: Any) -> npt.NDArray[Any]:
        """
        Return reconstructed images.

        Parameters
        ----------
        X : np.ndarray of shape (N, C, H, W)

        Returns
        -------
        reconstructed : np.ndarray of shape (N, C, H, W)
        """
        self._check_torch()
        if self.model is None or self.device is None:
            raise ConfigError("Model not initialized. Call fit() first.")

        X = self._validate(X)
        dataset = TensorDataset(torch.from_numpy(X))
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

        self.model.eval()
        results: list[npt.NDArray[Any]] = []

        with torch.no_grad():
            for (batch_x,) in loader:
                batch_x = batch_x.to(self.device)
                output = self.model(batch_x)
                results.append(output.cpu().numpy())

        return cast("npt.NDArray[Any]", np.concatenate(results, axis=0))
