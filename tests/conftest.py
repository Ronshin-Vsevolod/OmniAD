from collections.abc import Generator
from typing import Any

import numpy as np
import pytest

from omniad.registry import _DEPENDENCY_CHECKS, _REGISTRY
from omniad.utils.deps import is_available


@pytest.fixture(scope="session")  # type: ignore[misc]
def random_xy_dataset() -> tuple[Any, Any, Any]:
    """
    Generates a synthetic dataset for testing.
    Returns (X_train, X_test, y_test).
    """
    np.random.seed(3)
    X_normal = np.random.randn(200, 5)
    X_outliers = np.random.uniform(low=5, high=10, size=(20, 5))

    X = np.vstack([X_normal, X_outliers])
    y = np.hstack([np.zeros(200), np.ones(20)])

    indices = np.arange(len(X))
    np.random.shuffle(indices)

    X = X[indices]
    y = y[indices]

    split = int(len(X) * 0.8)
    return X[:split], X[split:], y[split:]


@pytest.fixture(autouse=True)  # type: ignore[misc]
def skip_if_dependency_missing(request: pytest.FixtureRequest) -> None:
    """
    If the test is parameterized with the 'algo_name' argument, it checks
    the dependencies for that algorithm and skips the test if necessary.
    """
    if "algo_name" in request.fixturenames:
        algo_name = request.getfixturevalue("algo_name")

        entry = _REGISTRY.get(algo_name)
        if not entry:
            return

        group = entry.get("requires")
        if group:
            pkg = _DEPENDENCY_CHECKS.get(group)
            if pkg and not is_available(pkg):
                pytest.skip(
                    f"Skipping {algo_name}: requires '{pkg}' "
                    f"(pip install omniad[{group}])"
                )


@pytest.fixture(scope="session")  # type: ignore[misc]
def timeseries_dataset() -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    """
    Synthetic time-series dataset for all time-series algorithm tests.

    Returns (X_train, X_test), each of shape (N, 2), dtype float32.
    """
    rng = np.random.default_rng(42)
    t = np.linspace(0, 4 * np.pi, 150)
    channel_1 = np.sin(t) + rng.normal(0, 0.05, size=t.shape)
    channel_2 = np.cos(t) + rng.normal(0, 0.05, size=t.shape)
    X = np.column_stack([channel_1, channel_2]).astype(np.float32)
    return X[:100], X[100:]


@pytest.fixture  # type: ignore[misc]
def deterministic_mode(request: pytest.FixtureRequest) -> Generator[None, None, None]:
    if "deterministic" not in request.keywords:
        yield
        return

    try:
        import torch
    except ImportError:
        pytest.skip("torch not available → cannot enable deterministic mode")

    prev_benchmark = torch.backends.cudnn.benchmark
    prev_deterministic = torch.backends.cudnn.deterministic
    prev_allow_tf32 = torch.backends.cudnn.allow_tf32
    prev_strict = torch.are_deterministic_algorithms_enabled()

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True, warn_only=True)

    yield

    torch.backends.cudnn.benchmark = prev_benchmark
    torch.backends.cudnn.deterministic = prev_deterministic
    torch.backends.cudnn.allow_tf32 = prev_allow_tf32
    torch.use_deterministic_algorithms(prev_strict, warn_only=True)
