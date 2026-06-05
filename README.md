# OmniAD: Universal Anomaly Detection Library

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![CI](https://github.com/yourusername/omniad/actions/workflows/ci.yml/badge.svg)](https://github.com/yourusername/omniad/actions)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![Checked with mypy](https://www.mypy-lang.org/static/mypy_badge.svg)](https://mypy-lang.org/)

**One API to rule them all.** OmniAD is a high-level, domain-agnostic meta-framework for Anomaly Detection. It wraps algorithms from `scikit-learn`, `PyTorch`, `HuggingFace Transformers`, and more, providing a strictly unified interface with **near-zero overhead**.

## 💡 Why OmniAD?

The Anomaly Detection ecosystem is fragmented:
- `sklearn` returns negative scores for anomalies. PyTorch returns positive losses.
- Saving a hybrid pipeline (e.g., TF-IDF Vectorizer + PyTorch Model) is a deployment nightmare.
- Logging and tracking experiments (e.g., via MLflow) require writing custom wrappers for every single algorithm.

**OmniAD solves this by providing:**
1. **Unified Interface:** `fit(X)` and `predict_score(X)`.
2. **Strict Score Convention:** Higher score ALWAYS means more anomalous.
3. **Smart Thresholding:** Auto-calibration via the `contamination` parameter.
4. **Isolated Serialization:** A unified `.zip` container format that safely packages Python objects, Scikit-learn models, and PyTorch weights together.
5. **Multi-Domain:** Seamlessly handle Tabular, Time-Series (sliding windows), Text (LLMs/TF-IDF), and Computer Vision data.

## 🚀 Installation

OmniAD's core is lightweight. Heavy frameworks are loaded lazily only when requested.

'''bash
# Base installation (Tabular models, Scikit-learn backend)
pip install omniad

# Install with domain-specific backends
pip install "omniad[deep]"    # PyTorch for Time-Series & CV
pip install "omniad[text]"    # HuggingFace Transformers for NLP
pip install "omniad[viz]"     # Matplotlib/Seaborn for visualization
pip install "omniad[all]"     # Everything
'''

*Note on NLP Models:* Some specific HuggingFace models may require additional tokenizer backends (like `sentencepiece`). HuggingFace will prompt you to install them if needed.

## 📖 Quickstart

### The Unified API
No matter the domain or backend, the workflow remains identical.

**1. Tabular Data (Scikit-Learn Backend)**
'''python
import numpy as np
from omniad import get_detector

X_train = np.random.randn(1000, 20)

model = get_detector("IsolationForest", n_estimators=100, contamination=0.05)
model.fit(X_train)

scores = model.predict_score(X_train)  # Raw anomaly scores
labels = model.predict(X_train)        # Binary labels (0=Normal, 1=Anomaly)
'''

**2. Time-Series (PyTorch Backend)**
OmniAD automatically handles 2D to 3D sliding window transformations.
'''python
# 'window_size' is automatically processed by the adapter
model = get_detector("LSTM", window_size=50, epochs=10, device="cuda")
model.fit(ts_data)
'''

**3. NLP / Text (HuggingFace Backend)**
OmniAD manages tokenization, embeddings, and chunking under the hood.
'''python
logs = ["User login successful", "FATAL: Null pointer dereference"]

# Use presets ('fast') to automatically load lightweight models like bert-tiny
model = get_detector("BertDetector", preset="fast", chunking_strategy="max")
model.fit(logs)
'''

### Serialization & MLOps Readiness
Save complex, multi-backend models into a single deployable file. Perfect for MLflow.
'''python
# Saves metadata, scalers, and PyTorch/Sklearn weights into one container
model.save("my_detector.zip")

loaded_model = get_detector("LSTM").load("my_detector.zip")
'''

## 🔬 Explainability & Visualization (Mixins)

OmniAD uses Mixins to expose advanced features cleanly.

'''python
from omniad.core.mixins import FeatureImportanceMixin, ReconstructionMixin
from omniad.viz import plot_timeseries_anomalies

# 1. Feature Importance (Global Explainability via Permutations)
if isinstance(model, FeatureImportanceMixin):
    importances = model.get_feature_importances(X)

# 2. Reconstruction (e.g., Autoencoders, LSTM)
if isinstance(model, ReconstructionMixin):
    expected_signal = model.predict_expected(X)
    plot_timeseries_anomalies(X, scores, model.threshold_, reconstruction=expected_signal)
'''

## ⚡ Benchmarks & Zero-Overhead Guarantee

OmniAD is built for production. It utilizes **Zero-Copy validation** (preserving sparse matrices) and direct backend delegation.

Our isolated benchmark suite profiles algorithms across three dimensions:
- **Quality:** Validates PR-AUC on real-world datasets (ODDS).
- **Overhead:** Proves mathematical parity and measures absolute latency overhead (predicting single rows) vs native libraries.
- **Performance:** Stress-tests throughput and VRAM/RAM consumption on millions of rows.

*Example Overhead:* For fast tabular models, the absolute latency overhead is strictly kept under `2ms`. For heavy deep-learning pipelines, throughput overhead is strictly within `2-5%`.

To run benchmarks yourself:
'''bash
python -m benchmarks.overhead.run --preset fast
'''

## 🏗 Architecture Overview

OmniAD is built using a layered "Onion" architecture to strictly separate mathematical logic, data validation, and backend integrations.

'''text
omniad/
├── core/         # L1: Strict contracts (BaseDetector, Mixins) and L1.5 Library Templates
├── algos/        # L2: Concrete algorithm wrappers grouped by domain
├── utils/        # L3: Declarative validation, dependency checks, and domain helpers
└── viz/          # L4: Optional plotting module (lazy-loaded)
'''

For a deep dive into how algorithms are implemented and how to add your own, see [CONTRIBUTING.md](CONTRIBUTING.md).

---
**License:** MIT License (or your chosen license)
