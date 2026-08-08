import pytest

from omniad import get_detector
from omniad.core.exceptions import ConfigError


def test_unknown_algorithm_raises_config_error() -> None:
    with pytest.raises(ConfigError):
        get_detector("DefinitelyNotRegistered")


def test_preset_merges_with_explicit_kwargs(fake_registry_entry: str) -> None:
    model = get_detector(fake_registry_entry, preset="fast", contamination=0.9)
    assert model.contamination == 0.9  # explicit kwarg wins over preset
    assert model.n_estimators == 999  # value comes from the fixture's own preset


def test_unknown_preset_raises_config_error(fake_registry_entry: str) -> None:
    with pytest.raises(ConfigError):
        get_detector(fake_registry_entry, preset="not_a_real_preset")


def test_explicit_kwargs_without_preset_keep_other_defaults(
    fake_registry_entry: str
) -> None:
    model = get_detector(fake_registry_entry, n_estimators=42)
    assert model.n_estimators == 42
    assert model.contamination == 0.1  # DummyDetector's own default, untouched
