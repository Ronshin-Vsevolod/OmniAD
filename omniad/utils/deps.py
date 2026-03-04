"""
Lazy dependency checking utilities.
"""
import importlib.util
from typing import Union


def is_available(package_name: str) -> bool:
    """Check if a package is importable without actually importing it."""
    spec = importlib.util.find_spec(package_name)
    return spec is not None


def check_dependency(
    group: Union[list[str], str, None],
    algo_name: str,
    checks: dict[str, str],
) -> None:
    """
    Verify that the required optional dependency is installed.

    Raises
    ------
    ImportError
        With a user-friendly install instruction if package is missing.
    """
    if group is None:
        return

    groups = [group] if isinstance(group, str) else group

    for g in groups:
        package = checks.get(g)
        if package and not is_available(package):
            raise ImportError(
                f"Algorithm '{algo_name}' requires the '{g}' extras.\n\n"
                f"  pip install omniad[{g}]\n\n"
                f"Missing package: {package}"
            )
