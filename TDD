# OmniAD: Architecture & Design Manifesto

**Version:** 1.0 (Core Stabilized)
**Purpose:** This document is the ultimate "source of truth" for OmniAD`s architecture. It captures the design patterns, data flows, and philosophies governing the library. It acts as the primary context for developers and LLMs joining the project.

---

## 1. Core Philosophy & Design Principles

OmniAD is a **domain-agnostic meta-framework**. We do not rewrite machine learning mathematics; we manage the lifecycle, compatibility, and edge cases of algorithms from disparate ecosystems (`scikit-learn`, `PyTorch`, `HuggingFace`, `PyGOD`).

1. **Strict Core, Flexible Input (Duck Typing):**
   The public API (`fit`, `predict_score`, `predict`) is rigid. However, the input `X` is polymorphic (`np.ndarray`, `torch.Tensor`, `scipy.sparse`, `List[str]`, `DataObject`).
2. **Declarative Validation:**
   Adapters do not write validation logic. They declare their capabilities (e.g., `{"require_2d", "reject_sparse"}`). The core executes validation in a globally safe order.
3. **Thin Gateway & Exception Translation:**
   We do not pre-validate math (like NaNs) if the backend can handle it natively. If a backend fails, OmniAD catches the backend-specific error and translates it into a domain error (`DataFormatError`, `ConfigError`), preserving the traceback.
4. **Universal Score Convention:**
   Across all algorithms, **Higher Score = More Anomalous**. Backends that violate this (e.g., Sklearn`s negative scores) are inverted at the adapter level.
5. **Zero-Copy & Performance:**
   Transformations (like Dense to Sparse, or `float64` to `float32`) only happen if explicitly required by the backend. Benchmarking overhead must strictly distinguish between *Latency* (per-row API calls) and *Throughput* (batch processing).

---

## 2. Global Architecture Topology

OmniAD uses a strict Layered (Onion) architecture. Dependencies point inwards.

```mermaid
graph TD
    classDef l4 fill:#ffcdd2,stroke:#333,stroke-width:1px;
    classDef l3 fill:#ffe0b2,stroke:#333,stroke-width:1px;
    classDef l2 fill:#c8e6c9,stroke:#333,stroke-width:1px;
    classDef l15 fill:#b3e5fc,stroke:#333,stroke-width:1px;
    classDef l1 fill:#f0f4f8,stroke:#333,stroke-width:2px;
    classDef ext fill:#eceff1,stroke:#90a4ae,stroke-dasharray: 5 5;

    User((User Application)) --> Facade

    subgraph Layer 4: Facade & UX
        Facade[omniad.__init__.py<br>get_detector]:::l4
        Presets[presets.py<br>Aliases: `fast`, `accurate`]:::l4
    end

    subgraph Layer 3: Infrastructure
        Registry[registry.py<br>Lazy Loading & Deps]:::l3
        Validation[utils.validation.py<br>Rules Registry]:::l3
        TSUtils[utils.timeseries.py<br>Sliding Windows]:::l3
        TextUtils[utils.text.py<br>Pooling & Chunking]:::l3
    end

    subgraph Layer 2: Concrete Adapters
        IForest[tabular.iforest.py<br>IsolationForestAdapter]:::l2
        LSTM[timeseries.lstm.py<br>LSTMAdapter]:::l2
        BERT[text.bert.py<br>BertDetectorAdapter]:::l2
        ConvAE[cv.autoencoder.py<br>ConvAutoencoderAdapter]:::l2
    end

    subgraph Layer 1.5: Library Templates
        SklearnBase[BaseSklearnAdapter]:::l15
        TorchBase[BaseTorchAdapter]:::l15
        TransformersBase[BaseTransformersAdapter]:::l15
    end

    subgraph Layer 1: Core
        Base[BaseDetector<br>Lifecycle & Serialization]:::l1
        Mixins[mixins.py<br>FeatureImp, Recon, Segm]:::l1
        Metrics[metrics.py<br>Backend-agnostic math]:::l1
        Exceptions[exceptions.py]:::l1
    end

    %% Flow connections
    Facade -->|1. Resolves Preset| Presets
    Facade -->|2. Checks Extra Deps| Registry
    Facade -.->|3. Lazy Imports| Adapter

    Adapter -->|Inherits| Template
    Template -->|Inherits| Base
    Adapter -->|Inherits| Mixins

    Base -->|Executes Rules| Validation
    Mixins -.-> Metrics

    %% Backend connections
    SklearnBase -->|Wraps| Sklearn[(Scikit-Learn)]:::ext
    TorchBase -->|Wraps| PyTorch[(PyTorch)]:::ext
    TransformersBase -->|Wraps| HF[(HuggingFace)]:::ext
```

---

## 3. Data Flow & Declarative Validation

We abandoned "God Object" validation functions with 20 boolean flags. Instead, we use a **Rule-based execution pipeline**.

1. **Declaration:** The adapter exposes a `set[str]` of rules via `@classmethod get_validation_rules()`.
2. **Execution:** `validate_input` executes these rules following a strictly hardcoded `_VALIDATION_ORDER` (e.g., must cast to numpy *before* checking shapes).

```mermaid
sequenceDiagram
    participant User
    participant Adapter
    participant Validator as utils.validation
    participant Backend

    User->>Adapter: fit(X)
    Note over Adapter: Adapter gets its rules:<br>{"to_numpy", "require_2d", "reject_nan"}
    Adapter->>Validator: validate_input(X, rules)

    rect rgb(240, 248, 255)
    Note over Validator: Following _VALIDATION_ORDER
    Validator->>Validator: 1. _rule_domain_text (Skipped)
    Validator->>Validator: 2. _rule_to_numpy (Executed)
    Validator->>Validator: 3. _rule_require_2d (Executed)
    Validator->>Validator: 4. _rule_reject_nan (Executed)
    end

    Validator-->>Adapter: X_valid
    Adapter->>Backend: Train on X_valid
```

---

## 4. The Detector Lifecycle & Core Mechanics

The `BaseDetector` enforces a strict lifecycle to ensure consistency.

### A. Initialization & Presets (`get_detector`)
Users interact with the factory. The factory intercepts `preset="fast"`, retrieves the `kwargs` from `presets.py`, merges them with explicitly provided `kwargs`, and performs pre-import dependency checks using `registry.py` (e.g., skipping `torch` imports if the `deep` extra is missing).

### B. Fitting & Threshold Calibration
1. **Seed:** `self._set_seed()` is called to guarantee reproducibility (handling `numpy`, `torch.manual_seed`, and `DataLoader` generators).
2. **Backend Fit:** `self._fit_backend(X)` is delegated to the adapter.
3. **Thresholding:** Raw scores lack physical meaning. `BaseDetector` automatically calls `self._calibrate_threshold(X)`. By default, it computes the quantile based on the `contamination` parameter.
    * *Optimization:* Heavy models (like BERT or LSTM) populate `self._cached_train_scores` during `_fit_backend` to avoid a second inference pass during calibration.

### C. Backend-Agnostic Metrics (`omniad.core.metrics`)
To support customizable scoring (e.g., changing MSE to MAE or Huber), we use a unified metric registry.
* **The `_ops` Trick:** Metric functions inspect the input type. If `x.__module__` starts with `"torch"`, they import and use `torch` math; otherwise, they use `numpy`. This keeps the module 0-dependency at import time while serving multiple backends.
* **Serialization Safety:** Custom callables are reverse-looked-up by name in the registry to allow safe `.pkl` serialization.

---

## 5. Domain-Specific Implementations

Each domain overrides `BaseDetector` or Layer 1.5 templates differently.

### 5.1 Time-Series (LSTM Example)
Time-Series introduces **Sliding Windows** and **Exogenous Variables**.
* `omniad.utils.timeseries.create_windows` uses ultra-fast `np.lib.stride_tricks` to convert 2D `(Time, Feats)` to 3D `(Batch, Window, Feats)`.
* **Modes:**
  * *Reconstruction:* Learns to recreate the whole window.
  * *Forecasting:* If `target_cols` is passed, the model uses all features (including exogenous) to predict only the `target_cols` of the final step.

### 5.2 NLP Pipeline (BERT Example)
NLP adapters don't just wrap models; they wrap **Pipelines** (Text → Tokenizer → Embedding → Chunking → Detector).

```mermaid
flowchart LR
    A[List of Strings] --> B[Tokenizer]
    B --> C[Transformer Forward]
    C --> D{Pooling Strategy<br>cls / mean}
    D --> E[Embeddings Matrix]

    subgraph Long Document Chunking
    E --> F{Chunking Strategy}
    F -->|max| G[Highest L2 Norm]
    F -->|mean| H[Average Vector]
    F -->|None| I[Truncation]
    end

    G --> J[Inner Detector<br>e.g., IsolationForest]
    H --> J
    I --> J
    J --> K[Anomaly Score]
```

### 5.3 Computer Vision
* Handles `(N, C, H, W)` tensors.
* Incorporates `SegmentationMixin` to return pixel-level anomaly heatmaps `(N, H, W)` without altering the global `predict_score` behavior.

---

## 6. Serialization Protocol (ModelIO)

Standard `pickle` fails on hybrid objects (e.g., Python dictionaries + PyTorch C++ tensors). OmniAD implements a **Container Storage Protocol (.zip)**.

```mermaid
block-beta
columns 1
  block:Archive["Model Archive (.zip)"]
    block:Meta["metadata.json"]
       MetaText["Class Name, OmniAD Version, Threshold, Contamination"]
    end
    block:Attr["attributes.pkl"]
       AttrText["Vectorizers, Scalers, Metrics (as strings), State Dictionaries (minus backend)"]
    end
    block:Folder["/backend (Directory)"]
       block:Files["Native Backend Files (Generated by Adapter)"]
         File1["model.pt (PyTorch)"]
         File2["model.joblib (Sklearn)"]
       end
    end
  end
  style Archive fill:#f3e5f5,stroke:#333,stroke-width:2px
  style Meta fill:#fff,stroke:#333
  style Attr fill:#e3f2fd,stroke:#333
  style Folder fill:#fff9c4,stroke:#333
```

* **Responsibility Split:** `BaseDetector` handles the zip archive, `metadata.json`, and safe pickling of attributes. The adapter only implements `_save_backend(path)`, utilizing its native save mechanism (e.g., `torch.save`).

---

## 7. Explainability & Mixins

OmniAD heavily uses Mixins to expose advanced features predictably.

1. **`FeatureImportanceMixin`**:
   * Provides `get_feature_importances()`.
   * Has a `"native"` mode (if the backend supports it, e.g., Sklearn Trees) and a fallback `"permutation"` mode (Model-Agnostic, shuffles columns and measures score degradation).
2. **`ReconstructionMixin`**:
   * Provides `predict_expected(X)`. Crucial for visualizing Time-Series forecasts or Autoencoder outputs.
3. **`SegmentationMixin`**:
   * Provides `predict_map(X)` for spatial/pixel-level anomaly localization.

---

## 8. Benchmarking & QA Principles

Benchmarking in OmniAD is treated as a first-class citizen, distinct from unit testing. It ensures the library remains viable for enterprise production.

1. **Process Isolation:** Memory leaks are real. Benchmark workers run in isolated `subprocess` environments. Peak RAM/VRAM are measured via `psutil` and `torch.cuda` by subtracting the base footprint.
2. **Throughput vs Latency:**
   * **Throughput:** Evaluates batch processing speed (`overhead_ratio`).
   * **Latency:** Evaluates real-time API response by running inference in a `for` loop, extracting the absolute `overhead_ms` per row.
3. **Native Parity:** Mathematical equivalence is non-negotiable. Overhead benchmarks execute a "pure" native implementation side-by-side with OmniAD to assert `parity_max_diff` ≈ 0.
