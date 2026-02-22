"""
Lazy dependency checking utilities.
"""
import importlib.util
from typing import Optional


def is_available(package_name: str) -> bool:
    """Check if a package is importable without actually importing it."""
    spec = importlib.util.find_spec(package_name)
    return spec is not None


def check_dependency(
    group: Optional[str],
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

    package = checks.get(group)

    if package is None:
        return

    if not is_available(package):
        raise ImportError(
            f"Algorithm '{algo_name}' requires the '{group}' extras.\n\n"
            f"  pip install omniad[{group}]\n\n"
            f"Missing package: {package}"
        )
