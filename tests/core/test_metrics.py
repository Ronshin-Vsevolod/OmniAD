from typing import Any

import pytest

from omniad.core.exceptions import ConfigError
from omniad.core.metrics import (
    register_metric,
    resolve_metric,
    reverse_lookup_metric,
)

torch = pytest.importorskip("torch", reason="torch not installed")


def test_resolve_builtin_metrics() -> None:
    """Checking built-in metrics (mse, mae, etc.)."""
    target = torch.tensor([[1.0, 1.0], [2.0, 2.0]])
    output = torch.tensor([[1.5, 1.0], [2.0, 4.0]])
    # Diff: [[-0.5, 0], [0, -2]]

    # MSE: (0.25 + 0) / 2 = 0.125; (0 + 4) / 2 = 2.0
    func = resolve_metric("mse")
    scores = func(target, output)
    expected = torch.tensor([0.125, 2.0])
    assert torch.allclose(scores, expected)

    # MAE: (0.5 + 0) / 2 = 0.25; (0 + 2) / 2 = 1.0
    func = resolve_metric("mae")
    scores = func(target, output)
    expected = torch.tensor([0.25, 1.0])
    assert torch.allclose(scores, expected)


def test_register_custom_metric() -> None:
    """Verification of user metric registration."""

    def my_crazy_metric(target: Any, output: Any) -> Any:
        return torch.mean(target * output, dim=1)

    # 1. Register
    metric_name = "crazy_dot"
    register_metric(metric_name, my_crazy_metric)

    # 2. Resolve by name
    func = resolve_metric(metric_name)
    assert func is my_crazy_metric

    # 3. Checking reverse lookup (required for save/load)
    name = reverse_lookup_metric(my_crazy_metric)
    assert name == metric_name


def test_resolve_errors() -> None:
    """Checking for configuration errors."""
    with pytest.raises(ConfigError, match="Unknown metric"):
        resolve_metric("non_existent_metric")

    with pytest.raises(ConfigError):
        resolve_metric(123)  # type: ignore


def test_register_errors() -> None:
    """Checking registration errors."""
    with pytest.raises(TypeError):
        register_metric("bad", "not_a_function")  # type: ignore
