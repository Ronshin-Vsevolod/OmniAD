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
    "ConvAutoencoder": {
        "debug": {
            "hidden_dim": 8,
            "epochs": 1,
            "batch_size": 4,
        },
        "fast": {
            "hidden_dim": 16,
            "epochs": 20,
            "batch_size": 64,
        },
        "accurate": {
            "hidden_dim": 64,
            "epochs": 100,
            "batch_size": 32,
            "learning_rate": 1e-4,
        },
    },
}
