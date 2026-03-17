from collections.abc import Generator
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest

from omniad.registry import _DEPENDENCY_CHECKS, _REGISTRY
from omniad.utils.deps import is_available


@pytest.fixture(autouse=True)  # type: ignore[misc]
def skip_if_dependency_missing(request: pytest.FixtureRequest) -> None:
    """Skip parametrized tests with algo_name if dependency is missing."""
    if "algo_name" not in request.fixturenames:
        return
    algo_name = request.getfixturevalue("algo_name")
    entry = _REGISTRY.get(algo_name)
    if not entry:
        return
    groups = entry.get("requires")
    if not groups:
        return
    for group in groups:
        pkg = _DEPENDENCY_CHECKS.get(group)
        if pkg and not is_available(pkg):
            pytest.skip(
                f"Skipping '{algo_name}': requires '{pkg}' "
                f"(pip install omniad[{group}])"
            )


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


@pytest.fixture(scope="session")  # type: ignore[misc]
def text_dataset() -> tuple[list[str], list[str], np.ndarray[Any, Any]]:
    """
    Synthetic text dataset.
    Returns (train_texts, test_texts, y_test).
    """
    train = [
        "user login successful",
        "user logout session ended",
        "file opened successfully",
        "connection established",
        "normal operation completed",
        "system reboot initiated",
        "user login successful",
        "file closed successfully",
        "connection closed normally",
        "backup completed successfully",
    ]
    test = [
        "kernel panic segfault critical error",
        "out of memory oom killer activated",
        "disk full write failed immediately",
        "user login successful",
        "normal operation completed",
    ]
    y_test = np.array([1, 1, 1, 0, 0])
    return train, test, y_test


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


@pytest.fixture(scope="session")  # type: ignore[misc]
def image_dataset() -> tuple[npt.NDArray[Any], npt.NDArray[Any]]:
    """Synthetic image dataset: 10 train + 5 test, 3-channel 32x32."""
    rng = np.random.default_rng(42)
    X_train = rng.random((10, 3, 32, 32)).astype(np.float32)
    X_test = rng.random((5, 3, 32, 32)).astype(np.float32)
    return X_train, X_test


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


def get_algo_domain(algo_name: str) -> str:
    from omniad.registry import _REGISTRY

    module_path = _REGISTRY[algo_name]["module"]
    # omniad.algos.text.bert -> "text"
    # omniad.algos.tabular.iforest -> "tabular"
    parts = module_path.split(".")
    if "algos" in parts:
        return parts[parts.index("algos") + 1]
    return "tabular"


@pytest.fixture  # type: ignore[misc]
def domain_dataset(
    request: pytest.FixtureRequest,
    random_xy_dataset: tuple[Any, Any, Any],
    timeseries_dataset: tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]],
    text_dataset: tuple[list[str], list[str], np.ndarray[Any, Any]],
    image_dataset: tuple[Any, Any],
) -> tuple[Any, Any]:
    """
    Returns (X_train, X_test) appropriate for the algorithm's domain.
    Falls back to tabular if algo_name is unknown.
    """
    callspec = getattr(request.node, "callspec", None)
    algo_name = callspec.params.get("algo_name") if callspec else None
    domain = get_algo_domain(algo_name) if algo_name else "tabular"

    if domain == "text":
        train, test, _ = text_dataset
        return train, test
    if domain == "timeseries":
        return timeseries_dataset
    if domain == "cv":
        return image_dataset
    X_train, X_test, _ = random_xy_dataset
    return X_train, X_test
