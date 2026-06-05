from __future__ import annotations

import contextlib
from collections.abc import Iterator

from omniad.core.exceptions import (
    AnomalyLibError,
    BackendError,
)


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
