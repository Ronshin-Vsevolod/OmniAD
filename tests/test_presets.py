import pytest

from omniad import get_detector
from omniad.core.exceptions import ConfigError


def test_preset_applies_params() -> None:
    """Verify that preset parameters are applied to the model."""
    model = get_detector("IsolationForest", preset="fast")
    assert model.n_estimators == 50  # type: ignore[attr-defined]
    assert model.n_jobs == -1  # type: ignore[attr-defined]


def test_preset_overridden_by_explicit_kwargs() -> None:
    """Verify that explicit kwargs override preset values."""
    model = get_detector("IsolationForest", preset="fast", n_estimators=10)
    assert model.n_estimators == 10  # type: ignore[attr-defined]
    assert model.n_jobs == -1  # type: ignore[attr-defined]


def test_unknown_preset_raises() -> None:
    """Verify that unknown preset name raises ConfigError."""
    with pytest.raises(ConfigError, match="Unknown preset"):
        get_detector("IsolationForest", preset="nonexistent")


def test_preset_for_unknown_algo_raises() -> None:
    """Verify that preset for algo with no presets raises ConfigError."""
    with pytest.raises(ConfigError, match="Unknown preset"):
        get_detector("TfidfDetector", preset="fast")
