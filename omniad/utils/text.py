"""
Provides chunking strategies for long texts and extensible
registry for adding custom aggregation methods.


"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
import numpy.typing as npt

from omniad.core.exceptions import ConfigError

# --- CHUNKING STRATEGIES ---

# Signature: (chunks: npt.NDArray of shape (n_chunks, hidden_dim)) ->
# npt.NDArray of shape (hidden_dim,)
ChunkAggregator = Callable[[npt.NDArray[Any]], npt.NDArray[Any]]


# --- Built-in strategies ---


def _mean_chunks(chunks: npt.NDArray[Any]) -> npt.NDArray[Any]:
    """Average all chunk embeddings."""
    return chunks.mean(axis=0)


def _max_norm_chunks(chunks: npt.NDArray[Any]) -> npt.NDArray[Any]:
    """Select chunk with highest L2 norm."""
    norms = np.linalg.norm(chunks, axis=1)
    return chunks[norms.argmax()]


def _first_chunk(chunks: npt.NDArray[Any]) -> npt.NDArray[Any]:
    """Use only the first chunk (beginning of text)."""
    return chunks[0]


def _last_chunk(chunks: npt.NDArray[Any]) -> npt.NDArray[Any]:
    """Use only the last chunk (end of text)."""
    return chunks[-1]


# --- Registry ---

_CHUNKING_REGISTRY: dict[str, ChunkAggregator] = {
    "mean": _mean_chunks,
    "max": _max_norm_chunks,
    "first": _first_chunk,
    "last": _last_chunk,
}


def register_chunking_strategy(name: str, func: ChunkAggregator) -> None:
    """
    Register a custom chunking aggregation strategy.

    Parameters
    ----------
    name : str
        Name to reference the strategy by.
    func : callable
        Function with signature (chunks: npt.NDArray) -> npt.NDArray.
        Input shape: (n_chunks, hidden_dim).
        Output shape: (hidden_dim,).

    Examples
    --------
    >>> def weighted_mean(chunks):
    ...     weights = np.arange(1, len(chunks) + 1, dtype=float)
    ...     weights /= weights.sum()
    ...     return (chunks * weights[:, None]).sum(axis=0)
    >>> register_chunking_strategy("weighted_mean", weighted_mean)
    """
    if not callable(func):
        raise TypeError(f"Expected callable, got {type(func)}")
    _CHUNKING_REGISTRY[name] = func


def get_available_chunking_strategies() -> list[str]:
    """Return names of all registered chunking strategies."""
    return sorted(_CHUNKING_REGISTRY.keys())


def resolve_chunking_strategy(
    strategy: str | ChunkAggregator | None,
) -> ChunkAggregator | None:
    """
    Resolve a chunking strategy from string name, callable, or None.

    Parameters
    ----------
    strategy : str, callable, or None
        - None: no chunking (truncate to max_length).
        - str: registered strategy name.
        - callable: custom aggregation function.

    Returns
    -------
    func : callable or None
        Resolved aggregation function, or None for truncation.
    """
    if strategy is None:
        return None

    if callable(strategy):
        return strategy

    if isinstance(strategy, str):
        if strategy not in _CHUNKING_REGISTRY:
            available = get_available_chunking_strategies()
            raise ConfigError(
                f"Unknown chunking strategy '{strategy}'. "
                f"Available strategies: {available}. "
                f"Or pass a callable(chunks_array) -> embedding_vector."
            )
        return _CHUNKING_REGISTRY[strategy]

    raise ConfigError(
        f"chunking_strategy must be str, callable, or None, got {type(strategy)}"
    )


def reverse_lookup_chunking(func: ChunkAggregator) -> str | None:
    """Find registry name for a chunking callable, or None."""
    for name, registered in _CHUNKING_REGISTRY.items():
        if registered is func:
            return name
    return None

    # --- POOLING ---


# Signature: (last_hidden_state, attention_mask) to pooled_tensor
PoolingFunction = Callable[[Any, Any], Any]


def _cls_pooling(last_hidden_state: Any, attention_mask: Any) -> Any:
    """Use CLS token embedding."""
    return last_hidden_state[:, 0, :]


def _mean_pooling(last_hidden_state: Any, attention_mask: Any) -> Any:
    """Mean of all tokens weighted by attention mask."""
    mask = attention_mask.unsqueeze(-1).float()
    summed = (last_hidden_state * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts


_POOLING_REGISTRY: dict[str, PoolingFunction] = {
    "cls": _cls_pooling,
    "mean": _mean_pooling,
}


def register_pooling(name: str, func: PoolingFunction) -> None:
    """
    Register a custom pooling strategy.

    Parameters
    ----------
    name : str
        Name to reference the strategy by.
    func : callable
        Function with signature (last_hidden_state, attention_mask) -> pooled.

    Examples
    --------
    >>> def max_pooling(hidden, mask):
    ...     hidden[mask.unsqueeze(-1).expand_as(hidden) == 0] = -1e9
    ...     return hidden.max(dim=1).values
    >>> register_pooling("max", max_pooling)
    """
    if not callable(func):
        raise TypeError(f"Expected callable, got {type(func)}")
    _POOLING_REGISTRY[name] = func


def get_available_poolings() -> list[str]:
    """Return names of all registered pooling strategies."""
    return sorted(_POOLING_REGISTRY.keys())


def resolve_pooling(pooling: str | PoolingFunction) -> PoolingFunction:
    """
    Resolve pooling from string name or callable.

    Parameters
    ----------
    pooling : str or callable

    Returns
    -------
    func : callable
    """
    if callable(pooling):
        return pooling

    if isinstance(pooling, str):
        if pooling not in _POOLING_REGISTRY:
            available = get_available_poolings()
            raise ConfigError(
                f"Unknown pooling '{pooling}'. "
                f"Available pooling: {available}. "
                f"Or pass a callable(last_hidden_state, attention_mask) -> pooled."
            )
        return _POOLING_REGISTRY[pooling]

    raise ConfigError(f"pooling must be str or callable, got {type(pooling)}")


def reverse_lookup_pooling(func: PoolingFunction) -> str | None:
    """Find registry name for a pooling callable, or None."""
    for name, registered in _POOLING_REGISTRY.items():
        if registered is func:
            return name
    return None
