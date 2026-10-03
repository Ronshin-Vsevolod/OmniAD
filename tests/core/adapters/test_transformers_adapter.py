"""
Layer 1.5 tests for BaseTransformersAdapter infrastructure.
"""
from __future__ import annotations

import pytest

import omniad.core.adapters.transformers_adapter as transformers_adapter_module
from omniad.algos.text.bert import BertDetectorAdapter


def test_check_transformers_requires_torch(monkeypatch) -> None:
    """Transformer adapters require PyTorch."""
    model = BertDetectorAdapter()

    monkeypatch.setattr(transformers_adapter_module, "torch", None)

    with pytest.raises(ImportError, match="omniad\\[deep\\]"):
        model._check_transformers()


def test_check_transformers_requires_transformers(monkeypatch) -> None:
    """Transformer adapters require HuggingFace Transformers."""
    model = BertDetectorAdapter()

    monkeypatch.setattr(transformers_adapter_module, "AutoModel", None)

    with pytest.raises(ImportError, match="omniad\\[text\\]"):
        model._check_transformers()
