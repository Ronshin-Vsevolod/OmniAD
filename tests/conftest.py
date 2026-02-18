from typing import Any

import numpy as np
import pytest


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
