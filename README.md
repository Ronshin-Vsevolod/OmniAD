# OmniAD

## High-Level Architecture
Purpose: To show the user how the library is structured globally—from its call to the core.
Meaning: To show layer isolation. The user only touches the Facade, while the Core is isolated from external libraries.

```mermaid
---
config:
  layout: fixed
---
flowchart TB
 subgraph subGraph0["Library Boundary"]
        Registry["Layer 3: Registry & Infra"]
        Facade["Layer 4: High-Level API"]
        Adapter["Layer 2: Concrete Wrappers"]
        Template["Layer 1.5: Library Templates"]
        Core["Layer 1: Core API"]
  end
    User(("User")) -- get_detector --> Facade
    Facade -- Lookup --> Registry
    Registry -- Instantiate --> Adapter
    Adapter -- Inherits --> Template
    Template -- Inherits --> Core
    Adapter -- Wraps --> Sklearn["External: Sklearn"] & Torch["External: PyTorch"] & PyGOD["External: PyGOD"]
    Core -- Defines Contract --- FitPredict["fit/predict_score"]
    Template -- Implements --- BoilerplateValidation["Boilerplate & Validation"]
    Adapter -- Implements --- ParamMapping["Param Mapping & Data Translation"]

     User:::user
     Facade:::facade
     Registry:::registry
     Adapter:::adapter
     Template:::template
     Core:::core
     Sklearn:::external
     Torch:::external
     PyGOD:::external
    classDef user fill:#f9f,stroke:#333,stroke-width:2px
    classDef facade fill:#ff9,stroke:#333,stroke-width:2px
    classDef registry fill:#aff,stroke:#333,stroke-width:1px
    classDef adapter fill:#9f9,stroke:#333,stroke-width:1px
    classDef template fill:#fc9,stroke:#333,stroke-width:1px
    classDef core fill:#ccf,stroke:#333,stroke-width:2px
    classDef external fill:#ddd,stroke:#333,stroke-dasharray:5 5
```

## Class Hierarchy
Purpose: For developers adding new algorithms.
Meaning: To show who to inherit from and how Mixins work.
```mermaid
classDiagram
    %% Core Layer
    class BaseDetector {
        +contamination: float
        +threshold_: float
        +fit(X, y)
        +predict_score(X)
        +predict(X)
        +save(path)
        +load(path)
        #_fit_backend(X)
    }

    %% Layer 1.5
    class BaseSklearnDetector {
        #_fit_backend(X)
        #_param_mapping
    }
    class BaseDeepLearningAdapter {
        +device: str
        #_fit_backend(X)
    }

    %% Mixins
    class FeatureImportanceMixin {
        +get_feature_importances()
    }
    class ReconstructionMixin {
        +predict_expected(X)
    }

    %% Inheritance
    BaseDetector <|-- BaseSklearnDetector
    BaseDetector <|-- BaseDeepLearningAdapter

    %% Realizations
    BaseSklearnDetector <|-- IsolationForestAdapter
    BaseDeepLearningAdapter <|-- LSTMAdapter

    %% Mixin Usage
    IsolationForestAdapter --|> FeatureImportanceMixin
    LSTMAdapter --|> ReconstructionMixin

    %% Notes
    note for BaseDetector "Strict Interface"
    note for BaseSklearnDetector "DRY: Implements generic sklearn logic"
    note for LSTMAdapter "Specific logic + Reconstruction capability"
```

## Data Flow: The fit() Method
Purpose: To explain what happens “under the hood” when a user calls fit.
Meaning: To show centralized validation, threshold calculation, and backend operations.
```mermaid
sequenceDiagram
    participant User
    participant Wrapper as Adapter (Your Lib)
    participant Validator as Utils.Validation
    participant Core as BaseDetector
    participant Backend as External Lib (e.g. Torch)

    User->>Wrapper: fit(X)

    rect rgb(240, 248, 255)
    note right of Wrapper: 1. Валидация
    Wrapper->>Validator: validate_input(X, task)
    Validator-->>Wrapper: Clean X (Numpy/Tensor)
    end

    rect rgb(255, 240, 240)
    note right of Wrapper: 2. Инфраструктура
    Wrapper->>Core: _set_seed()
    end

    rect rgb(240, 255, 240)
    note right of Wrapper: 3. Обучение Бэкенда
    Wrapper->>Backend: Init & Fit (with mapped params)
    Backend-->>Wrapper: Trained Model
    end

    rect rgb(255, 255, 240)
    note right of Wrapper: 4. Авто-порог
    Wrapper->>Wrapper: predict_score(X) -> train_scores
    Wrapper->>Core: quantile(train_scores, 1-contamination)
    Core-->>Wrapper: set self.threshold_
    end

    Wrapper->>Wrapper: set _is_fitted = True
    Wrapper-->>User: self
```

## Storage Structure (ZIP Container)
Purpose: To explain to administrators and engineers what a .zip model file is.
Meaning: To show that the heavy model and metadata are stored separately.
```mermaid
block-beta
columns 1
  block:Archive["Model Archive (.zip / .adl)"]
    block:Meta["metadata.json"]
       MetaText["Class Name<br/>Version<br/>Threshold"]
    end
    block:Attr["attributes.pkl"]
       AttrText["Scalers<br/>Init Params<br/>Feature Names"]
    end
    block:Folder["/backend (Folder)"]
       block:Files["Native Files"]
         File1["model.pt (Torch)"]
         File2["model.joblib (Sklearn)"]
         File3["config.json (HF)"]
       end
    end
  end
  style Archive fill:#f9f,stroke:#333,stroke-width:2px
  style Meta fill:#fff,stroke:#333
  style Attr fill:#e1f5fe,stroke:#333
  style Folder fill:#fff9c4,stroke:#333
```
