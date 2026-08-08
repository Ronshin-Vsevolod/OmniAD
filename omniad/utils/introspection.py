"""
Shared introspection helpers.

Used both by the .pyi stub generator (scripts/generate_stubs.py) and
by omniad.describe() — a single source of truth for "what parameters
does this algorithm accept", so IDE hints and console discoverability
never drift apart.
"""

from __future__ import annotations

import inspect
from typing import Any

from omniad.core.exceptions import ConfigError


def extract_init_params(cls: type) -> dict[str, Any]:
    """
    Extract public __init__ parameters and their defaults.

    Parameters
    ----------
    cls : type
        Adapter class to inspect.

    Returns
    -------
    params : dict[str, Any]
        Mapping of parameter name to its default, or "required" if
        no default. `self` and `**kwargs` are excluded.
    """
    # cls is a bare `type`, so accessing __init__ is technically unsound
    # for custom metaclasses; safe here since we only introspect our
    # own adapter classes.
    sig = inspect.signature(cls.__init__)  # type: ignore[misc]
    return {
        name: (
            param.default
            if param.default is not inspect.Parameter.empty
            else "required"
        )
        for name, param in sig.parameters.items()
        if name != "self" and param.kind != inspect.Parameter.VAR_KEYWORD
    }


def resolve_default_inner_detector(cls: type) -> str:
    """
    Read the default value of a composition adapter's `detector=`
    constructor parameter (single source of truth — avoids hardcoding
    the default name a second time next to the signature).

    Parameters
    ----------
    cls : type
        Adapter class exposing a string-defaulted `detector` parameter.

    Returns
    -------
    name : str
        Registry name of the default inner detector.

    Raises
    ------
    ConfigError
        If `detector` has no string default (adapter misconfigured).
    """
    default = extract_init_params(cls).get("detector")
    if not isinstance(default, str):
        raise ConfigError(
            f"{cls.__name__} must declare a string default for `detector=` "
            f"to support class-level capability introspection."
        )
    return default


def resolve_delegated_capabilities(detector_name: str) -> set[str]:
    """
    Capabilities of a registered OmniAD detector, restricted to those
    marked delegatable in CAPABILITY_REGISTRY.

    Parameters
    ----------
    detector_name : str
        Name registered in omniad.registry.

    Returns
    -------
    capabilities : set[str]
    """
    from omniad import _resolve_adapter_class
    from omniad.core.mixins import CAPABILITY_REGISTRY

    delegatable = {info.slug for info in CAPABILITY_REGISTRY if info.delegatable}
    inner_caps = _resolve_adapter_class(detector_name).get_capabilities()
    return delegatable & inner_caps


def resolve_delegated_capabilities_dict(detector_name: str) -> dict[str, bool]:
    """Boolean-lookup version of `resolve_delegated_capabilities`."""
    from omniad.core.mixins import CAPABILITY_REGISTRY

    active = resolve_delegated_capabilities(detector_name)
    return {info.slug: info.slug in active for info in CAPABILITY_REGISTRY}
