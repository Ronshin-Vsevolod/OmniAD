import numpy as np
import pandas as pd
import pytest

from omniad.utils.validation import validate_input


def test_validation_accepts_pandas() -> None:
    """Verify that the library processes DataFrame and Series."""
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})

    # 1. DataFrame -> Numpy
    X_out = validate_input(df)
    assert isinstance(X_out, np.ndarray)
    assert X_out.shape == (3, 2)

    # 2. Series -> Numpy (col vector)
    series = pd.Series([1, 2, 3])
    X_out_s = validate_input(series)
    assert isinstance(X_out_s, np.ndarray)
    assert X_out_s.shape == (3, 1)  # Must do reshape(-1, 1)


def test_validation_rejects_bad_pandas() -> None:
    """Verify that NaN values are captured in pandas."""
    df = pd.DataFrame({"a": [1, np.nan, 3]})

    from omniad.core.exceptions import DataFormatError

    with pytest.raises(DataFormatError):
        validate_input(df)
