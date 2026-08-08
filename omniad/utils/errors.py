from __future__ import annotations

import contextlib
import functools
from collections.abc import Iterator
from typing import Any, Callable, TypeVar, cast

from omniad.core.exceptions import (
    AnomalyLibError,
    BackendError,
)

F = TypeVar("F", bound=Callable[..., Any])


@contextlib.contextmanager
def backend_boundary(
    detector_name: str,
    phase: str,
) -> Iterator[None]:
    """
    Translate backend exceptions into OmniAD exceptions.

    This context manager defines the boundary between OmniAD and
    third-party libraries such as sklearn, PyTorch, HuggingFace,
    PyGOD, etc.

    OmniAD exceptions pass through unchanged.

    Any other exception is wrapped into BackendError while
    preserving the original traceback via ``raise ... from e``.
    """
    try:
        yield

    except AnomalyLibError:
        raise

    except Exception as e:
        raise BackendError(
            f"[{detector_name}] Backend failure during '{phase}'.\n"
            f"Original error ({type(e).__name__}): {e}"
        ) from e


def backend_boundary_method(phase: str) -> Callable[[F], F]:
    """
    Method decorator variant of `backend_boundary`, for public API
    methods hitting a backend outside the fit()/predict_score()
    template already covered by BaseDetector (e.g. mixin entry points
    `predict_map`, `predict_expected`).

    Parameters
    ----------
    phase : str
        Label used in the translated BackendError message.

    Examples
    --------
    >>> class MyAdapter(BaseTorchAdapter, SegmentationMixin):
    ...     @backend_boundary_method("predict_map")
    ...     def predict_map(self, X):
    ...         ...
    """

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
            with backend_boundary(self.__class__.__name__, phase=phase):
                return func(self, *args, **kwargs)

        return cast(F, wrapper)

    return decorator
