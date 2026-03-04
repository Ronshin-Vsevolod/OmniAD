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
## 3. General Naming Conventions
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

## 5. Standard Parameters
We adhere to a single list of parameters, compiled according to the principle of necessity and sufficiency for the majority.

### Approved Parameter Names

| **Parameter**     | **Type**       | **Description**                                      |
|-------------------|----------------|------------------------------------------------------|
| **n_estimators**  | `int`          | Number of base estimators (trees)                    |
| **contamination** | `float`        | Proportion of outliers / anomalies expected          |
| **n_jobs**        | `int`          | Number of jobs to run in parallel (-1 = all cores)   |
| **random_state**  | `int`          | Controls randomness for reproducibility              |
| **verbose**       | `int` or `bool`| Verbosity level of output                            |
| **window_size**   | `int`          | Size of the sliding window for time series data      |
| **batch_size**    | `int`             | Number of samples per gradient update (Deep Learning)        |
| **epochs**        | `int`             | Number of epochs to train the model                          |
| **learning_rate** | `float`           | Optimization step size (default usually 1e-3 or 1e-4)        |
| **device**        | `str`             | Computation device: "cpu", "cuda", or "auto"                 |
| **hidden_dim**    | `int`             | Size of the hidden layer / state representation              |
| **target_cols**   | `list[int]`       | Indices of columns to predict (Forecasting mode)             |
| **score_metric**  | `str`             | Name of the metric for anomaly scoring ('mse', 'mae', etc.)  |
| **detector**      | `str`             | Backend detector name for embedding pipelines     |
| **detector_kwargs** | `dict`          | Additional parameters for the backend detector    |
| **model_name**    | `str`             | HuggingFace model identifier                      |
| **max_length**    | `int`             | Maximum token length for transformer models       |
| **pooling**       | `str`             | Pooling strategy for transformer output           |
| **chunking_strategy** | `str`         | Strategy for handling long texts                  |
| **save_weights**  | `bool`            | Whether to save model weights in archive          |
| **max_features**  | `int`             | Maximum vocabulary size for text vectorizers      |
| **ngram_range**   | `tuple[int,int]`  | Range of n-grams for text vectorizers             |

## 6. Naming & Registry Conventions

To keep the factory (`get_detector`) working, we follow strict naming rules:

1.  **Algorithm Name:** The string key in `registry.py` (e.g., `"IsolationForest"`). This is what the user types.
2.  **Class Name:** Must be `{AlgorithmName}Adapter` (e.g., `IsolationForestAdapter`).
3.  **File Name:** Should be concise snake_case.
    - Preferred: `iforest.py`, `lstm.py`.
    - Allowed: `lstm_torch.py` (if disambiguation is needed).

**Registration Process:**
1.  Create `algos/domain/my_algo.py`.
2.  Define `class MyAlgoAdapter(Base...)`.
3.  Add `"MyAlgo": "omniad.algos.domain.my_algo"` to `omniad/registry.py`.

## 7. Testing Strategy

We use `pytest`. Tests are located in `tests/`.

1.  **Common Tests (`tests/test_common.py`):**
    - Automatically checks API contract (fit/predict/save/load) for ALL algorithms in registry.
    - No need to write basic smoke tests for new algorithms manually.

2.  **Parity/Logic Tests (`tests/algos/...`):**
    - Created specifically for each algorithm (e.g., `test_iforest.py`).
    - **Goal:** Verify that our wrapper matches the backend's math (e.g. score inversion).
    - **Docstrings:** It is acceptable to use similar docstrings ("Verifies parity with sklearn...") across these tests as they perform similar functions for different algorithms.
