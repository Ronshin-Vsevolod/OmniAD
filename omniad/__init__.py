import importlib
from typing import Any, cast

from omniad.core.base import BaseDetector
from omniad.core.exceptions import ConfigError
from omniad.registry import _REGISTRY

__version__ = "0.1.0"


def get_detector(name: str, **kwargs: Any) -> BaseDetector:
    """
    Factory method to instantiate a detector by name.

    Parameters
    ----------
    name : str
        The name of the algorithm (e.g., "IsolationForest").
        Must be registered in omniad.registry.
    **kwargs : Any
        Parameters passed to the detector's __init__ method.

    Returns
    -------
    model : BaseDetector
        An instance of the requested anomaly detector.

    Raises
    ------
    ConfigError
        If the algorithm name is not found in the registry.
    ImportError
        If the module cannot be loaded (e.g., missing dependency).
    AttributeError
        If the class (Name + 'Adapter') is not found in the module.
    """
    if name not in _REGISTRY:
        available = list(_REGISTRY.keys())
        raise ConfigError(
            f"Unknown algorithm: '{name}'. Available algorithms: {available}"
        )

    module_path = _REGISTRY[name]

    try:
        module = importlib.import_module(module_path)
    except ImportError as e:
        raise ImportError(
            f"Failed to import module '{module_path}' for algorithm '{name}'. "
            f"Ensure all dependencies are installed. Error: {e}"
        ) from e

    # Convention: The class name must be {AlgorithmName}Adapter
    class_name = f"{name}Adapter"

    try:
        model_class = getattr(module, class_name)
    except AttributeError as e:
        raise AttributeError(
            f"Module '{module_path}' does not have class '{class_name}'. "
            "Check naming conventions."
        ) from e

    return cast(BaseDetector, model_class(**kwargs))
