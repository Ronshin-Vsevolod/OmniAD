from collections.abc import Generator
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest

from tests.support import AllDomainDatasets, require_algo


@pytest.fixture(autouse=True)
def skip_if_dependency_missing(request: pytest.FixtureRequest) -> None:
    """Skip parametrized tests with algo_name if dependency is missing."""
    if "algo_name" not in request.fixturenames:
        return
    require_algo(request.getfixturevalue("algo_name"))


@pytest.fixture(scope="session")
def random_xy_dataset() -> tuple[Any, Any, Any]:
    """(X_train, X_test, y_test) synthetic tabular dataset."""
    np.random.seed(3)
    X_normal = np.random.randn(200, 5)
    X_outliers = np.random.uniform(low=5, high=10, size=(20, 5))
    X = np.vstack([X_normal, X_outliers])
    y = np.hstack([np.zeros(200), np.ones(20)])
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    X, y = X[indices], y[indices]
    split = int(len(X) * 0.8)
    return X[:split], X[split:], y[split:]


@pytest.fixture(scope="session")
def text_dataset() -> tuple[list[str], list[str], np.ndarray[Any, Any]]:
    """(train_texts, test_texts, y_test) synthetic text dataset."""
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


@pytest.fixture(scope="session")
def timeseries_dataset() -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    """(X_train, X_test) synthetic time-series, shape (N, 2), float32."""
    rng = np.random.default_rng(42)
    t = np.linspace(0, 4 * np.pi, 150)
    channel_1 = np.sin(t) + rng.normal(0, 0.05, size=t.shape)
    channel_2 = np.cos(t) + rng.normal(0, 0.05, size=t.shape)
    X = np.column_stack([channel_1, channel_2]).astype(np.float32)
    return X[:100], X[100:]


@pytest.fixture(scope="session")
def image_dataset() -> tuple[npt.NDArray[Any], npt.NDArray[Any]]:
    """(X_train, X_test) synthetic images: 100 train + 50 test, 3x32x32."""
    rng = np.random.default_rng(42)
    X_train = rng.random((100, 3, 32, 32)).astype(np.float32)
    X_test = rng.random((50, 3, 32, 32)).astype(np.float32)
    return X_train, X_test


@pytest.fixture
def deterministic_mode(request: pytest.FixtureRequest) -> Generator[None, None, None]:
    if "deterministic" not in request.keywords:
        yield
        return
    try:
        import torch
    except ImportError:
        pytest.skip("torch not available → cannot enable deterministic mode")

    prev = (
        torch.backends.cudnn.benchmark,
        torch.backends.cudnn.deterministic,
        torch.backends.cudnn.allow_tf32,
        torch.are_deterministic_algorithms_enabled(),
    )
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    yield
    (
        torch.backends.cudnn.benchmark,
        torch.backends.cudnn.deterministic,
        torch.backends.cudnn.allow_tf32,
        _,
    ) = prev
    torch.use_deterministic_algorithms(prev[3], warn_only=True)


@pytest.fixture(scope="session")
def all_domain_datasets(
    random_xy_dataset: tuple[Any, Any, Any],
    timeseries_dataset: tuple[Any, Any],
    text_dataset: tuple[list[str], list[str], Any],
    image_dataset: tuple[Any, Any],
) -> AllDomainDatasets:
    X_train, X_test, _ = random_xy_dataset
    train_texts, test_texts, _ = text_dataset
    return AllDomainDatasets(
        tabular=(X_train, X_test),
        timeseries=timeseries_dataset,
        text=(train_texts, test_texts),
        cv=image_dataset,
    )
