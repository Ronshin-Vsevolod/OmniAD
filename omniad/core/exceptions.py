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
