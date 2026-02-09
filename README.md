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
