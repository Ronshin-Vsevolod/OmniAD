from __future__ import annotations

import logging
import os
from typing import Any, Callable, cast

import numpy as np
import numpy.typing as npt

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
except ImportError:
    torch = None
    nn = None
    DataLoader = None
    TensorDataset = None

from omniad.core.base import BaseDetector
from omniad.core.exceptions import ConfigError
from omniad.core.metrics import resolve_metric
from omniad.utils.validation import validate_input

logger = logging.getLogger(__name__)


class BaseTorchAdapter(BaseDetector):
    """
    Base template for PyTorch-based anomaly detectors.

    Handles device management, data loading, training loops, and serialization.
    Implements the Template Method pattern for training steps and scoring.

    Parameters
    ----------
    contamination : float, default=0.1
        The proportion of outliers in the data set.
    device : str, default="auto"
        Device to use for computation ("cpu", "cuda", "auto").
    batch_size : int, default=32
        Number of samples per gradient update.
    epochs : int, default=10
        Number of epochs to train the model.
    learning_rate : float, default=1e-3
        Optimizer learning rate.
    verbose : bool, default=False
        If True, logging loss during training.
    score_metric : str or callable, default="mse"
        Metric used to compute anomaly scores.
    **kwargs : Any
        Additional arguments passed to BaseDetector or stored for backend configuration.
    """

    def __init__(
        self,
        contamination: float = 0.1,
        device: str = "auto",
        batch_size: int = 32,
        epochs: int = 10,
        learning_rate: float = 1e-3,
        verbose: int | bool = False,
        score_metric: str | Callable[..., Any] = "mse",
        **kwargs: Any,
    ) -> None:
        super().__init__(contamination=contamination, **kwargs)
        self.device_name = device
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.verbose = verbose
        self.score_metric = score_metric
        self.score_func = resolve_metric(score_metric)
        self.random_state = kwargs.get("random_state")
        # Options for optimizer/scheduler passed via get_detector backend_options
        self.backend_options = kwargs.get("backend_options", {})

        self.model: nn.Module | None = None
        self.device: torch.device | None = None
        # Critical for loading: remember input dimension
        self.n_features_in_: int | None = None
        # For DataLoader shuffle
        self._generator: torch.Generator | None = None

    def _check_torch(self) -> None:
        """Check if PyTorch is installed."""
        if torch is None:
            raise ImportError(
                "PyTorch is not installed. Run 'pip install omniad[deep]'."
            )

    def _setup_device(self) -> torch.device:
        """
        Configure the compute device based on availability and user preference.

        Returns
        -------
        device : torch.device
            The target device object.
        """
        self._check_torch()
        if self.device_name == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(self.device_name)

    #  --- Hook Methods (Override these for custom behavior) ---

    def _build_model(self, input_dim: int) -> nn.Module:
        """
        Construct the PyTorch model architecture.

        Parameters
        ----------
        input_dim : int
            Number of input features.

        Returns
        -------
        model : nn.Module
            The initialized PyTorch model.
        """
        raise NotImplementedError("Concrete adapter must implement _build_model")

    def _configure_optimizer(self, model: nn.Module) -> torch.optim.Optimizer:
        """
        Create the optimizer.

        Default is Adam. Override to use SGD, RMSprop, etc.
        Parameters passed via `backend_options['optimizer_params']` are applied here.
        """
        optimizer_params = self.backend_options.get("optimizer_params", {})
        return torch.optim.Adam(
            model.parameters(), lr=self.learning_rate, **optimizer_params
        )

    def _configure_criterion(self) -> nn.Module:
        """
        Create the loss function for training.

        Default is MSELoss (Reconstruction Error).
        Override for BinaryCrossEntropy, KLD, etc.
        """
        return nn.MSELoss()

    def _train_step(
        self,
        batch: tuple[torch.Tensor, ...],
        model: nn.Module,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
    ) -> torch.Tensor:
        """
        Perform a single training step: Forward -> Loss -> Backward.

        Parameters
        ----------
        batch : tuple
            A single batch from DataLoader (X, [y]).
        model : nn.Module
            The model being trained.
        criterion : nn.Module
            Loss function.
        optimizer : Optimizer
            Optimization algorithm.

        Returns
        -------
        loss : torch.Tensor
            Scalar loss value for the batch.
        """
        X = batch[0].to(self.device)

        # Default Autoencoder logic: Target is Input (Unsupervised)
        output = model(X)
        loss = criterion(output, X)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return loss

    def _compute_anomaly_score(
        self, X: torch.Tensor, output: torch.Tensor
    ) -> torch.Tensor:
        """
        Calculate anomaly score using the configured metric.
        Target is assumed to be 'X' (Autoencoder mode) by default.
        Override this method if Target != Input (e.g. Forecasting).
        """
        return self.score_func(X, output)

    def _validate(self, X: Any) -> Any:
        """Torch-specific: validate + convert to float32."""
        return validate_input(X, ensure_2d=False, allow_nd=True).astype(np.float32)

    def _extract_input_dim(self, X: Any) -> int:
        """
        Extract model input dimension from data.

        Default: last dimension (works for 2D tabular and 3D timeseries).
        Override for CV (channels at dim=1).
        """
        if X.ndim > 1:
            return int(X.shape[-1])
        return 1

    # --- Main Logic ---

    def _fit_backend(self, X: Any, y: Any | None = None) -> None:
        """
        Generic PyTorch training loop implementation.
        """
        self._check_torch()

        # 1. Setup metadata
        input_dim: int
        input_dim = self._extract_input_dim(X)

        self.n_features_in_ = input_dim
        self.device = self._setup_device()
        self._set_seed()

        # 2. Build Model
        self.model = self._build_model(self.n_features_in_).to(self.device)
        self._backend_model = self.model

        # 3. Setup Training Infrastructure
        dataset = TensorDataset(torch.from_numpy(X))
        loader = DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True, generator=self._generator
        )

        optimizer = self._configure_optimizer(self.model)
        criterion = self._configure_criterion()

        logger.debug("Device: %s", self.device)
        logger.debug(
            "Model: %s | params=%d",
            self.model.__class__.__name__,
            sum(p.numel() for p in self.model.parameters()),
        )

        # 4. Training Loop
        self.model.train()
        for epoch in range(self.epochs):
            total_loss = 0.0
            for batch in loader:
                loss = self._train_step(batch, self.model, criterion, optimizer)
                total_loss += loss.item()

            logger.info("Epoch %d/%d | loss=%.6f", epoch + 1, self.epochs, total_loss)

        self._cached_train_scores = self.predict_score(X)

    def predict_score(self, X: Any) -> npt.NDArray[Any]:
        """
        Inference loop. Returns anomaly scores using _compute_anomaly_score.
        """
        self._check_torch()
        if self.model is None or self.device is None:
            raise ConfigError("Model not initialized.")

        X = self._validate(X)

        logger.debug("predict_score: n_samples=%d", len(X))

        dataset = TensorDataset(torch.from_numpy(X))
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

        self.model.eval()
        scores = []

        with torch.no_grad():
            for (batch_x,) in loader:
                batch_x = batch_x.to(self.device)
                output = self.model(batch_x)

                batch_scores = self._compute_anomaly_score(batch_x, output)
                scores.append(batch_scores.cpu().numpy())

        return cast("npt.NDArray[Any]", np.concatenate(scores))

    # --- Serialization ---

    def _save_backend(self, path: str) -> None:
        """Save PyTorch model weights (state_dict)."""
        if self.model is None:
            raise ConfigError("Model is empty.")
        torch.save(self.model.state_dict(), os.path.join(path, "model.pt"))

    def _load_backend(self, path: str) -> None:
        """Load PyTorch model weights."""
        self._check_torch()
        model_path = os.path.join(path, "model.pt")

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        # Safety check: ensure input dimension was restored by BaseDetector.load()
        if self.n_features_in_ is None:
            raise ConfigError(
                "Input dimension (n_features_in_) is missing. "
                "The model might be corrupted."
            )

        self.device = self._setup_device()
        self.model = self._build_model(self.n_features_in_).to(self.device)
        self._backend_model = self.model

        # Load weights with map_location (handles GPU->CPU loading)
        state_dict = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.eval()

    def _set_seed(self) -> None:
        """Set random seeds for PyTorch and NumPy."""
        seed = getattr(self, "random_state", None)
        if seed is not None:
            self._check_torch()
            # Note: The sequence below is critically important
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            np.random.seed(seed)

            # For DataLoader shuffle
            self._generator = torch.Generator()
            self._generator.manual_seed(seed)
        else:
            self._generator = None
