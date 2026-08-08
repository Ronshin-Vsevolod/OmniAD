import pytest

from omniad.utils.deps import check_dependency, is_available


def test_is_available_true_for_installed_package() -> None:
    assert is_available("numpy") is True


def test_is_available_false_for_nonexistent_package() -> None:
    assert is_available("this_package_does_not_exist_12345") is False


def test_check_dependency_none_group_is_noop() -> None:
    check_dependency(None, "SomeAlgo", {})


def test_check_dependency_raises_with_install_hint() -> None:
    checks = {"deep": "this_package_does_not_exist_12345"}
    with pytest.raises(ImportError, match=r"pip install omniad\[deep\]"):
        check_dependency("deep", "SomeAlgo", checks)


def test_check_dependency_accepts_list_of_groups() -> None:
    checks = {"deep": "numpy", "text": "this_package_does_not_exist_12345"}
    with pytest.raises(ImportError):
        check_dependency(["deep", "text"], "SomeAlgo", checks)


def test_every_requires_group_is_registered_in_dependency_checks() -> None:
    """
    Regression guard for the 'streaming' incident: a `requires` group
    absent from _DEPENDENCY_CHECKS makes check_dependency() silently
    no-op instead of raising.
    """
    from omniad.registry import _DEPENDENCY_CHECKS, _REGISTRY

    referenced = {g for e in _REGISTRY.values() for g in (e.get("requires") or [])}
    assert referenced <= set(_DEPENDENCY_CHECKS)
