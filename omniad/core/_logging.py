"""
Centralized logging configuration.

Design principles:
- Silent by default (NullHandler)
- User opts in via set_verbosity() or verbose= parameter
- All logging through standard Python logging module
- No print() anywhere in the codebase

See Configuration Aliases in CONTRIBUTING to modify the file.
"""

from __future__ import annotations

import logging
import sys
import time
from collections.abc import Generator
from contextlib import contextmanager
from typing import Any

ROOT_LOGGER_NAME = "omniad"
_root_logger = logging.getLogger(ROOT_LOGGER_NAME)
_root_logger.addHandler(logging.NullHandler())

_stream_handler: logging.StreamHandler | None = None  # type: ignore[type-arg]

_LEVEL_MAP = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warning": logging.WARNING,
    "error": logging.ERROR,
    0: logging.WARNING,  # verbose=0
    1: logging.INFO,  # verbose=1
    2: logging.DEBUG,  # verbose=2
    False: logging.WARNING,
    True: logging.INFO,
}

_FORMAT = "%(asctime)s [%(name)s] %(levelname)s: %(message)s"
_FORMAT_SHORT = "[%(name)s] %(levelname)s: %(message)s"


def set_verbosity(level: str | int = "info", fmt: str | None = None) -> None:
    """
    Configure OmniAD logging output.

    Parameters
    ----------
    level : str or int
        Logging level. Accepts: "debug", "info", "warning", "error",
        or int (0=warning, 1=info, 2=debug).
    fmt : str or None
        Custom format string. If None, uses default short format.

    Examples
    --------
    >>> import omniad
    >>> omniad.set_verbosity("info")       # see fit/predict lifecycle
    >>> omniad.set_verbosity("debug")      # see everything
    >>> omniad.set_verbosity("warning")    # quiet (default)
    """
    global _stream_handler

    log_level = _LEVEL_MAP.get(level)
    if log_level is None:
        if isinstance(level, int) and level in (
            logging.DEBUG,
            logging.INFO,
            logging.WARNING,
            logging.ERROR,
        ):
            log_level = level
        else:
            raise ValueError(
                f"Unknown verbosity level: {level!r}. "
                f"Use 'debug', 'info', 'warning', or 0/1/2."
            )

    if _stream_handler is None:
        _stream_handler = logging.StreamHandler(sys.stderr)
        _root_logger.addHandler(_stream_handler)

    _stream_handler.setFormatter(logging.Formatter(fmt or _FORMAT_SHORT))
    _stream_handler.setLevel(log_level)
    _root_logger.setLevel(log_level)


def _ensure_verbose_handler(verbose: int | bool) -> None:
    """
    Called by BaseDetector when verbose > 0.
    Ensures a StreamHandler exists without overriding user config.
    """
    if not verbose:
        return

    # If user already configured handlers (beyond NullHandler), respect that
    real_handlers = [
        h for h in _root_logger.handlers if not isinstance(h, logging.NullHandler)
    ]
    if real_handlers:
        return

    # Auto-configure for convenience
    level = _LEVEL_MAP.get(verbose, logging.INFO)
    set_verbosity(level)


@contextmanager
def log_phase(
    logger: logging.Logger,
    phase: str,
    **details: Any,
) -> Generator[dict[str, Any], None, None]:
    """
    Context manager that logs start/end of a phase with timing.

    Parameters
    ----------
    logger : logging.Logger
        Logger instance.
    phase : str
        Phase name (e.g., "fit", "predict_score").
    **details
        Additional key=value pairs logged at start.

    Yields
    ------
    context : dict
        Mutable dict. Add keys during the phase; they'll be logged at end.

    Examples
    --------
    >>> with log_phase(logger, "fit", n_samples=200, n_features=5) as ctx:
    ...     # ... do work ...
    ...     ctx["threshold"] = 0.342
    # Logs:
    # [omniad.algos.tabular.iforest] INFO: fit started | n_samples=200 n_features=5
    # [omniad.algos.tabular.iforest] INFO: fit completed | threshold=0.342 elapsed=2.1s
    """
    detail_str = " | ".join(f"{k}={v}" for k, v in details.items())
    logger.info("%s started | %s", phase, detail_str)

    context: dict[str, Any] = {}
    start = time.perf_counter()

    yield context

    elapsed = time.perf_counter() - start
    context["elapsed"] = f"{elapsed:.3f}s"
    result_str = " | ".join(f"{k}={v}" for k, v in context.items())
    logger.info("%s completed | %s", phase, result_str)
