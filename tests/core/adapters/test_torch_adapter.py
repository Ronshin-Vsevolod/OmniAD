"""
Layer 1.5 template tests for BaseTorchAdapter's own glue logic —
independent of any specific network architecture (see
tests/algos/timeseries/test_lstm.py and tests/algos/cv/test_autoencoder.py
for architecture-specific parity/injection checks).
"""
from __future__ import annotations

import zipfile

import joblib
import numpy as np
import pytest

torch = pytest.importorskip("torch")

import omniad.core.adapters.torch_adapter as torch_adapter_module  # noqa: E402
from omniad.core.adapters.torch_adapter import BaseTorchAdapter  # noqa: E402
from omniad.core.exceptions import ConfigError  # noqa: E402

X = np.random.default_rng(0).normal(size=(20, 3)).astype(np.float32)


class _LinearAdapter(BaseTorchAdapter):
    """Minimal concrete adapter: a single linear autoencoder layer."""

    def _build_model(self, input_dim: int):
        return torch.nn.Linear(input_dim, input_dim)


def test_save_backend_before_fit_raises_config_error(tmp_path: str) -> None:
    with pytest.raises(ConfigError, match="empty"):
        _LinearAdapter()._save_backend(str(tmp_path))


def test_check_torch_raises_import_error_when_torch_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch_adapter_module, "torch", None)
    with pytest.raises(ImportError, match="pip install omniad"):
        _LinearAdapter()._check_torch()


def test_torch_runtime_state_is_not_duplicated_in_wrapper_state(tmp_path) -> None:
    """Torch runtime objects are serialized only through the backend."""
    model = _LinearAdapter(epochs=1).fit(X)
    path = tmp_path / "model.zip"
    model.save(str(path))

    with zipfile.ZipFile(path) as zf:
        assert "backend/model.pt" in zf.namelist()
        zf.extract("attributes.pkl", tmp_path)

    state = joblib.load(tmp_path / "attributes.pkl")

    assert "model" not in state
    assert "device" not in state
    assert "_generator" not in state
