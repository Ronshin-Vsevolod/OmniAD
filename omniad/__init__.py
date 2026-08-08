from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any, cast

from omniad.core.base import BaseDetector
from omniad.core.exceptions import ConfigError
from omniad.core.mixins import describe_capability
from omniad.presets import PRESETS
from omniad.registry import _DEPENDENCY_CHECKS, _REGISTRY, filter_by_domain
from omniad.utils.deps import check_dependency
from omniad.utils.introspection import extract_init_params

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


def _resolve_adapter_class(name: str) -> type[BaseDetector]:
    """
    Resolve and import the adapter class for a registered algorithm name,
    without instantiating it. Shared by get_detector() and describe().

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

    class_name = f"{name}Adapter"
    model_class = getattr(module, class_name, None)
    if model_class is None:
        raise AttributeError(
            f"Module '{module_path}' has no class '{class_name}'. "
            "Check naming conventions."
        )

    return cast(type[BaseDetector], model_class)


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
    final_kwargs = _apply_presets(name, kwargs)
    model_class = _resolve_adapter_class(name)

    return model_class(**final_kwargs)


@dataclass
class DetectorInfo:
    """Structured description of a registered algorithm."""

    name: str
    class_name: str
    domain: str
    capabilities: list[str]
    presets: list[str]
    params: dict[str, Any]


def _format_detector_info(info: DetectorInfo) -> str:
    lines = [
        f"<{info.name}> ({info.class_name})",
        f"Domain: {info.domain}",
        f"Capabilities: {', '.join(info.capabilities) or 'none'}",
        f"Presets: {', '.join(info.presets) or 'none'}",
        "Parameters:",
    ]
    lines += [f"  {k} = {v!r}" for k, v in info.params.items()]
    return "\n".join(lines)


def describe(name: str, verbose: bool = True) -> DetectorInfo:
    """
    Describe a registered algorithm without instantiating or fitting it.

    Imports only the single requested adapter module (not the whole
    registry), so it remains safe to call even if extras for other
    algorithms are not installed.

    Parameters
    ----------
    name : str
        Registered algorithm name.
    verbose : bool, default=True
        If True, also print a human-readable summary (for console/
        notebook use).

    Returns
    -------
    info : DetectorInfo
        Structured description: domain, capabilities, presets, and
        constructor parameters with their defaults.
    """
    model_class = _resolve_adapter_class(name)
    entry = _REGISTRY[name]

    info = DetectorInfo(
        name=name,
        class_name=model_class.__name__,
        domain=entry["domain"],
        capabilities=[
            describe_capability(slug)[0]
            for slug in sorted(model_class.get_capabilities())
        ],
        presets=list(PRESETS.get(name, {}).keys()),
        params=extract_init_params(model_class),
    )

    if verbose:
        print(_format_detector_info(info))

    return info


def list_detectors(
    domain: str | None = None, capability: str | None = None
) -> list[str]:
    """
    List registered algorithms matching the given filters.

    This is a zero-import registry query, e.g.:

        omniad.list_detectors(capability="incremental_learning")
        omniad.list_detectors(domain="text")

    Parameters
    ----------
    domain : str | None, optional
        Restrict to algorithms in this domain.
    capability : str | None, optional
        Restrict to algorithms declaring this capability slug.

    Returns
    -------
    names : list[str]
    """
    names = filter_by_domain(domain) if domain is not None else list(_REGISTRY.keys())
    if capability is None:
        return names

    result = []
    for name in names:
        try:
            cls = _resolve_adapter_class(name)
        except ImportError:
            continue
        if capability in cls.get_capabilities():
            result.append(name)
    return result
