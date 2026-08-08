"""
Layer 1.5 template tests for BaseRiverAdapter's own glue logic —
independent of any specific streaming algorithm (see
tests/algos/tabular/test_half_space_trees.py for HalfSpaceTrees-
specific parity/streaming checks).
"""
from __future__ import annotations

import pytest

import omniad.core.adapters.river_adapter as river_adapter_module
from omniad.core.adapters.river_adapter import BaseRiverAdapter


class _FakeRiverAdapter(BaseRiverAdapter):
    """Adapter that never needs a real backend for this ImportError check."""

    def _build_backend(self) -> None:
        raise NotImplementedError

    def _save_backend(self, path: str) -> None:
        pass

    def _load_backend(self, path: str) -> None:
        pass


def test_check_river_raises_import_error_when_river_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(river_adapter_module, "river", None)
    with pytest.raises(ImportError, match="pip install omniad"):
        _FakeRiverAdapter()._check_river()
