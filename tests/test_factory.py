import pytest

from omniad import get_detector
from omniad.core.exceptions import ConfigError


def test_factory_raises_unknown_algo() -> None:
    with pytest.raises(ConfigError, match="Unknown algorithm"):
        get_detector("SuperDuperAlgo")
