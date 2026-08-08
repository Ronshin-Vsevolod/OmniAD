from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable, Literal

import numpy as np
import numpy.typing as npt

from omniad.core.exceptions import CapabilityError, ConfigError

# Built-in strategy names, kept for IDE autocomplete (Literal | str pattern)
ThresholdName = Literal["quantile", "sigma3", "iqr", "ewma", "streaming_quantile"]

ThresholdFactory = Callable[[], "ThresholdStrategy"]


class ThresholdStrategy(ABC):
    """
    Unified threshold calibration contract.

    A single abstraction serves both batch (`fit`) and streaming
    (`update`) calibration. This mirrors `core.metrics`: one interface,
    multiple possible implementations.

    Subclasses fall into two groups, detected structurally (no separate
    registration) by whether `update()` is overridden:
    - Batch-only (quantile, sigma3, iqr): need the full score
      distribution to recompute; implement only `fit()`.
    - Streaming-capable (ewma, streaming_quantile): maintain running
      state; override `update()` too.

    See get_available_thresholds(streaming=True) / _supports_streaming().
    """

    @abstractmethod
    def fit(self, scores: npt.NDArray[Any], contamination: float) -> float:
        """
        Compute a threshold from a full batch of training scores.

        Parameters
        ----------
        scores : np.ndarray of shape (n_samples,)
        contamination : float
            Expected proportion of anomalies.

        Returns
        -------
        threshold : float
        """

    def update(self, score: float, contamination: float) -> float:
        """
        Incrementally refine the threshold with a single new score.

        Not supported by default. Override only in strategies with a
        genuine online update rule (e.g. EWMAThreshold).

        Parameters
        ----------
        score : float
            Anomaly score of the newly observed sample.
        contamination : float
            Expected proportion of anomalies.

        Returns
        -------
        threshold : float

        Raises
        ------
        CapabilityError
            If this strategy has no online update rule.
        """
        raise CapabilityError(
            f"'{self.__class__.__name__}' is a batch-only threshold strategy "
            f"and has no online update rule. Use one of the streaming-capable "
            f"strategies instead: {get_available_thresholds(streaming=True)}. "
            f"You can register your own via "
            f"omniad.utils.thresholds.register_threshold() with an update() "
            f"override — see ThresholdStrategy for the contract."
        )


class FunctionThresholdStrategy(ThresholdStrategy):
    """
    Adapter wrapping a plain `fn(scores, contamination) -> float`
    callable as a ThresholdStrategy.

    Preserves the lightweight ergonomics of simple custom thresholds
    (most users only need a formula, not a stateful class). Streaming
    is unavailable for such thresholds by construction; users who need
    `update()` should subclass ThresholdStrategy directly.
    """

    def __init__(self, fn: Callable[[npt.NDArray[Any], float], float]) -> None:
        self.fn = fn

    def fit(self, scores: npt.NDArray[Any], contamination: float) -> float:
        return float(self.fn(scores, contamination))


class QuantileThreshold(ThresholdStrategy):
    """Percentile-based threshold (batch-only)."""

    def fit(self, scores: npt.NDArray[Any], contamination: float) -> float:
        return float(np.quantile(scores, 1 - contamination))


class Sigma3Threshold(ThresholdStrategy):
    """Mean + 3 standard deviations (batch-only)."""

    def fit(self, scores: npt.NDArray[Any], contamination: float) -> float:
        return float(scores.mean() + 3 * scores.std())


class IQRThreshold(ThresholdStrategy):
    """Interquartile range: Q3 + 1.5 * IQR (batch-only)."""

    def fit(self, scores: npt.NDArray[Any], contamination: float) -> float:
        q1, q3 = np.quantile(scores, [0.25, 0.75])
        return float(q3 + 1.5 * (q3 - q1))


class EWMAThreshold(ThresholdStrategy):
    """
    Exponentially weighted mean + k * std. No extra dependencies;
    supports both `fit()` and `update()` on the same running state.

    Parameters
    ----------
    alpha : float, default=0.05
        Smoothing factor for the running mean/variance.
    k : float, default=3.0
        Number of standard deviations above the mean.
    """

    def __init__(self, alpha: float = 0.05, k: float = 3.0) -> None:
        self.alpha = alpha
        self.k = k
        self._mean: float | None = None
        self._var: float = 0.0

    def fit(self, scores: npt.NDArray[Any], contamination: float) -> float:
        self._mean = float(scores.mean())
        self._var = float(scores.var())
        return float(self._mean + self.k * (self._var**0.5))

    def update(self, score: float, contamination: float) -> float:
        if self._mean is None:
            self._mean = score
            return float(self._mean + self.k * (self._var**0.5))
        delta = score - self._mean
        self._mean += self.alpha * delta
        self._var = (1 - self.alpha) * (self._var + self.alpha * delta**2)
        return float(self._mean + self.k * (self._var**0.5))


class StreamingQuantileThreshold(ThresholdStrategy):
    """
    Streaming analogue of QuantileThreshold
    """

    def __init__(self) -> None:
        self._q: Any = None
        self._contamination: float | None = None

    def _ensure(self, contamination: float) -> None:
        if self._q is None or self._contamination != contamination:
            try:
                import river.stats as river_stats
            except ImportError as e:
                raise ImportError(
                    "threshold_strategy='streaming_quantile' requires river.\n"
                    "  pip install river"
                ) from e
            self._q = river_stats.Quantile(q=1 - contamination)
            self._contamination = contamination

    def fit(self, scores: npt.NDArray[Any], contamination: float) -> float:
        self._ensure(contamination)
        for s in scores:
            self._q.update(float(s))
        return float(self._q.get())

    def update(self, score: float, contamination: float) -> float:
        self._ensure(contamination)
        self._q.update(score)
        return float(self._q.get())


_THRESHOLD_REGISTRY: dict[str, ThresholdFactory] = {
    "quantile": QuantileThreshold,
    "sigma3": Sigma3Threshold,
    "iqr": IQRThreshold,
    "ewma": EWMAThreshold,
    "streaming_quantile": StreamingQuantileThreshold,
}
_RAW_FUNCTION_NAMES: dict[Callable[..., Any], str] = {}


def register_threshold(
    name: str,
    strategy: Callable[[npt.NDArray[Any], float], float] | type[ThresholdStrategy],
) -> None:
    """
    Register a custom threshold strategy: either a plain batch formula
    or a full ThresholdStrategy subclass (required for streaming support).

    Parameters
    ----------
    name : str
        Strategy identifier for use as `threshold_strategy=name`.
    strategy : callable or type[ThresholdStrategy]
        Either `fn(scores, contamination) -> float`, or a
        ThresholdStrategy subclass implementing `fit()` (and,
        optionally, `update()` for streaming).

    Examples
    --------
    >>> def double_median(scores, contamination):
    ...     return float(np.median(scores) * 2)
    >>> register_threshold("double_median", double_median)
    """
    if isinstance(strategy, type) and issubclass(strategy, ThresholdStrategy):
        _THRESHOLD_REGISTRY[name] = strategy
    elif callable(strategy):
        _THRESHOLD_REGISTRY[name] = lambda: FunctionThresholdStrategy(strategy)
        _RAW_FUNCTION_NAMES[strategy] = name
    else:
        raise TypeError(
            f"Expected callable or ThresholdStrategy subclass, got {type(strategy)}"
        )


def get_available_thresholds(streaming: bool | None = None) -> list[str]:
    """
    List registered threshold strategy names.

    Parameters
    ----------
    streaming : bool | None, default=None
        If True, only strategies overriding `update()` (usable with
        `partial_fit()`). If False, only batch-only strategies. If
        None, all strategies.
    """
    names = sorted(_THRESHOLD_REGISTRY)
    if streaming is None:
        return names
    return [n for n in names if _supports_streaming(n) == streaming]


def _supports_streaming(name: str) -> bool:
    """
    Class-level capability check: does this strategy override update()?
    """
    factory = _THRESHOLD_REGISTRY[name]
    if isinstance(factory, type) and issubclass(factory, ThresholdStrategy):
        return factory.update is not ThresholdStrategy.update
    return False  # raw functions wrapped via FunctionThresholdStrategy
    # are never streaming-capable


def resolve_threshold(
    strategy: ThresholdName
    | str
    | ThresholdStrategy
    | type[ThresholdStrategy]
    | Callable[..., float],
) -> ThresholdStrategy:
    """
    Resolve a threshold_strategy value into a ThresholdStrategy instance.

    Raises
    ------
    ConfigError
        If `strategy` is an unknown name or an unsupported type.
    """
    if isinstance(strategy, ThresholdStrategy):
        return strategy
    if isinstance(strategy, type) and issubclass(strategy, ThresholdStrategy):
        return strategy()
    if isinstance(strategy, str):
        if strategy not in _THRESHOLD_REGISTRY:
            raise ConfigError(
                f"Unknown threshold strategy '{strategy}'. "
                f"Available: {get_available_thresholds()}"
            )
        return _THRESHOLD_REGISTRY[strategy]()
    if callable(strategy):
        return FunctionThresholdStrategy(strategy)
    raise ConfigError(
        f"threshold_strategy must be str, ThresholdStrategy, callable, or None. "
        f"Got {type(strategy)}"
    )


def reverse_lookup_threshold(fn: Callable[..., Any]) -> str | None:
    """Look up the registered name of a raw threshold function, for safe pickling."""
    return _RAW_FUNCTION_NAMES.get(fn)
