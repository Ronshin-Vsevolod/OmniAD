class AnomalyLibError(Exception):
    """Base class for all exceptions in the library."""

    pass


class ModelNotFittedError(AnomalyLibError):
    """Called when attempting to predict/transform without prior fit."""

    pass


class DataFormatError(AnomalyLibError):
    """Called when the input data format is incorrect."""

    pass


class ConfigError(AnomalyLibError):
    """Called when parameters are configured incorrectly."""

    pass


class BackendError(AnomalyLibError):
    """
    Failure inside a third-party backend.

    Raised when sklearn, PyTorch, HuggingFace, or another backend
    throws an exception that is not already represented by an
    OmniAD-specific exception.
    """

    pass


class CapabilityError(AnomalyLibError):
    """
    Raised when a valid, well-formed request cannot be fulfilled because
    the algorithm/backend does not structurally support the operation.

    Distinct from ConfigError: ConfigError signals invalid parameters,
    CapabilityError signals a valid request that this class cannot
    honor (e.g. partial_fit() on a batch-only model, a batch-only
    threshold strategy used in a streaming context, native feature
    importance on a model that doesn't expose it).
    """

    pass
