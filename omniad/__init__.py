import importlib
from typing import Any, cast

from omniad.core.base import BaseDetector
from omniad.core.exceptions import ConfigError
from omniad.presets import PRESETS
from omniad.registry import _DEPENDENCY_CHECKS, _REGISTRY
from omniad.utils.deps import check_dependency

__version__ = "0.1.0"


def _apply_presets(algo_name: str, kwargs: dict[str, Any]) -> dict[str, Any]:
    """
    Apply configuration presets if 'preset' argument is present.
    """
    if "preset" not in kwargs:
        return kwargs

    preset_name = kwargs.pop("preset")

    if algo_name not in PRESETS or preset_name not in PRESETS[algo_name]:
        available = list(PRESETS.get(algo_name, {}).keys())
        raise ConfigError(
            f"Unknown preset '{preset_name}' for {algo_name}. Available: {available}"
        )

    preset_params = PRESETS[algo_name][preset_name].copy()
    preset_params.update(kwargs)

    return preset_params


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

    final_kwargs = _apply_presets(name, kwargs)

    entry = _REGISTRY[name]

    check_dependency(
        group=entry["requires"],
        algo_name=name,
        checks=_DEPENDENCY_CHECKS,
    )

    module_path = entry["module"]

    try:
        module = importlib.import_module(module_path)
    except ImportError as e:
        raise ImportError(
            f"Failed to import module '{module_path}' for algorithm '{name}'. "
            f"Ensure all dependencies are installed. Error: {e}"
        ) from e

    # Convention: The class name must be {AlgorithmName}Adapter
    class_name = f"{name}Adapter"

    model_class = getattr(module, class_name, None)
    if model_class is None:
        raise AttributeError(
            f"Module '{module_path}' has no class '{class_name}'. "
            "Check naming conventions."
        )

    return cast(BaseDetector, model_class(**final_kwargs))
