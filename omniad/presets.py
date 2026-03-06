"""
Configuration presets (aliases) for algorithms.
Provides shortcuts for common use-cases like 'fast' or 'accurate'.
"""
from typing import Any

# Structure: AlgorithmName -> AliasName -> ParamDict
PRESETS: dict[str, dict[str, dict[str, Any]]] = {
    "BertDetector": {
        "fast": {
            "model_name": "prajjwal1/bert-tiny",
            "batch_size": 64,
        },
        "accurate": {
            "model_name": "roberta-base",
            "batch_size": 16,
        },
        "debug": {
            "model_name": "hf-internal-testing/tiny-random-bert",
            "batch_size": 2,
        },
    },
    "IsolationForest": {
        "fast": {
            "n_estimators": 50,
            "n_jobs": -1,
        },
        "accurate": {
            "n_estimators": 300,
            "n_jobs": -1,
        },
    },
    "LSTM": {
        "debug": {
            "epochs": 1,
            "hidden_dim": 4,
            "window_size": 5,
        }
    },
}
