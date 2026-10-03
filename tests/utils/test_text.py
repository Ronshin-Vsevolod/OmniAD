"""
Layer 3 tests for text chunking and pooling strategy registries.

Chunking is pure numpy logic — tested directly. Pooling operates on
torch tensors (attention_mask math) — gated behind pytest.importorskip.
"""
from __future__ import annotations

import numpy as np
import pytest

from omniad.core.exceptions import ConfigError
from omniad.utils.text import (
    ChunkAggregator,
    get_available_chunking_strategies,
    get_available_poolings,
    register_chunking_strategy,
    register_pooling,
    resolve_chunking_strategy,
    resolve_pooling,
    reverse_lookup_chunking,
    reverse_lookup_pooling,
)

CHUNKS = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])


def _chunking(name: str) -> ChunkAggregator:
    strategy = resolve_chunking_strategy(name)
    assert strategy is not None
    return strategy


# --- Built-in chunking strategies ---


def test_mean_chunking_averages_all_chunks() -> None:
    np.testing.assert_allclose(_chunking("mean")(CHUNKS), [3.0, 4.0])


def test_max_chunking_selects_highest_l2_norm_chunk() -> None:
    np.testing.assert_allclose(_chunking("max")(CHUNKS), [5.0, 6.0])


def test_first_chunking_selects_first_chunk() -> None:
    np.testing.assert_allclose(_chunking("first")(CHUNKS), [1.0, 2.0])


def test_last_chunking_selects_last_chunk() -> None:
    np.testing.assert_allclose(_chunking("last")(CHUNKS), [5.0, 6.0])


def test_resolve_chunking_strategy_none_means_no_chunking() -> None:
    assert resolve_chunking_strategy(None) is None


def test_resolve_chunking_strategy_passes_through_callable() -> None:
    def custom(chunks: np.ndarray) -> np.ndarray:
        return chunks[0]

    assert resolve_chunking_strategy(custom) is custom


def test_resolve_chunking_strategy_unknown_name_raises_config_error() -> None:
    with pytest.raises(ConfigError):
        resolve_chunking_strategy("not_a_real_strategy")


def test_register_chunking_strategy_roundtrip() -> None:
    def weighted_first_double(chunks: np.ndarray) -> np.ndarray:
        return chunks[0] * 2

    register_chunking_strategy("weighted_first_double_test", weighted_first_double)

    assert "weighted_first_double_test" in get_available_chunking_strategies()
    resolved = resolve_chunking_strategy("weighted_first_double_test")
    assert resolved is not None
    np.testing.assert_allclose(resolved(CHUNKS), [2.0, 4.0])
    assert (
        reverse_lookup_chunking(weighted_first_double) == "weighted_first_double_test"
    )


def test_reverse_lookup_chunking_returns_none_for_unregistered() -> None:
    def never_registered(chunks: np.ndarray) -> np.ndarray:
        return chunks[0]

    assert reverse_lookup_chunking(never_registered) is None


def test_register_chunking_strategy_rejects_non_callable() -> None:
    with pytest.raises(TypeError):
        register_chunking_strategy("bad_chunking_test", "not_callable")  # type: ignore[arg-type]


# --- Built-in pooling strategies ---


def test_cls_pooling_selects_first_token() -> None:
    torch = pytest.importorskip("torch")
    hidden = torch.tensor([[[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]]])
    mask = torch.tensor([[1, 1, 0]])

    result = resolve_pooling("cls")(hidden, mask)
    np.testing.assert_allclose(result.numpy(), [[1.0, 1.0]])


def test_mean_pooling_averages_only_attended_tokens() -> None:
    torch = pytest.importorskip("torch")
    hidden = torch.tensor([[[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]]])
    mask = torch.tensor([[1, 1, 0]])  # third token masked out

    result = resolve_pooling("mean")(hidden, mask)
    np.testing.assert_allclose(result.numpy(), [[1.5, 1.5]])  # avg of tokens 0,1 only


def test_resolve_pooling_passes_through_callable() -> None:
    def custom(hidden: object, mask: object) -> object:
        return hidden

    assert resolve_pooling(custom) is custom


def test_resolve_pooling_unknown_name_raises_config_error() -> None:
    with pytest.raises(ConfigError):
        resolve_pooling("not_a_real_pooling")


def test_register_pooling_roundtrip() -> None:
    torch = pytest.importorskip("torch")

    def last_token_pooling(hidden, mask):
        return hidden[:, -1, :]

    register_pooling("last_token_pooling_test", last_token_pooling)

    assert "last_token_pooling_test" in get_available_poolings()
    resolved = resolve_pooling("last_token_pooling_test")
    hidden = torch.tensor([[[1.0, 1.0], [9.0, 9.0]]])
    np.testing.assert_allclose(resolved(hidden, None).numpy(), [[9.0, 9.0]])
    assert reverse_lookup_pooling(last_token_pooling) == "last_token_pooling_test"


def test_reverse_lookup_pooling_returns_none_for_unregistered() -> None:
    def never_registered(hidden: object, mask: object) -> object:
        return hidden

    assert reverse_lookup_pooling(never_registered) is None


def test_register_pooling_rejects_non_callable() -> None:
    with pytest.raises(TypeError):
        register_pooling("bad_pooling_test", "not_callable")  # type: ignore[arg-type]
