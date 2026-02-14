# Developer Guide

## 1. Code Style & Linting
We use strict automated checks. Before committing, ensure your code passes:
- **Ruff** for formatting and linting.
- **MyPy** for static type checking (strict mode).

Run checks manually:
```bash
pre-commit run --all-files
```

## 2. Docstrings Standard
We follow the **NumPy Docstring Standard.**
Every public class and method must have a docstring describing parameters, returns, and raised exceptions.

**Example:**
```python
def fit(self, X, y=None):
    """
    Fit the model.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        The input data.
    y : ignored
        Not used, present for API consistency.

    Returns
    -------
    self : object
        Fitted estimator.
    """
```
## 3. Naming Conventions
- **Classes:** CamelCase (e.g., IsolationForestAdapter).
- **Functions/Methods:** snake_case (e.g., predict_score).
- **Internal/Private:** Prefix with _ (e.g., _fit_backend).
- **Mathematical variables:**
- - Matrix: X (uppercase).
- - Vector: y (lowercase).
- - Count: n_samples, n_features.

## 4. Exceptions
Never raise generic Exception or ValueError in core logic.
Use custom exceptions from OmniAD.core.exceptions.
