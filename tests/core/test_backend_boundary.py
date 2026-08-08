import numpy as np
import pytest

from omniad.core.exceptions import BackendError, DataFormatError
from tests.doubles import DummyDetector

X = np.random.default_rng(0).normal(size=(20, 3))


def test_fit_error_translation() -> None:
    with pytest.raises(BackendError, match="fit"):
        DummyDetector(fail_on="fit").fit(X)


def test_predict_score_error_translation() -> None:
    model = DummyDetector().fit(X)
    model.fail_on = "predict"
    with pytest.raises(BackendError, match="predict_score"):
        model.predict_score(X)


def test_save_error_translation(tmp_path) -> None:
    model = DummyDetector().fit(X)
    model.fail_on = "save"
    with pytest.raises(BackendError, match="save"):
        model.save(str(tmp_path / "model"))


def test_load_error_translation(tmp_path, monkeypatch) -> None:
    model = DummyDetector().fit(X)
    model.save(str(tmp_path / "model"))

    broken = DummyDetector()
    monkeypatch.setattr(
        broken,
        "_load_backend",
        lambda path: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    with pytest.raises(BackendError, match="load"):
        broken.load(str(tmp_path / "model"))


def test_original_exception_preserved_as_cause() -> None:
    with pytest.raises(BackendError) as excinfo:
        DummyDetector(fail_on="fit").fit(X)
    assert isinstance(excinfo.value.__cause__, RuntimeError)


def test_domain_errors_pass_through_untouched() -> None:
    class RaisesDomainError(DummyDetector):
        def _fit_backend(self, X, y=None) -> None:
            raise DataFormatError("already a domain error")

    with pytest.raises(DataFormatError):
        RaisesDomainError().fit(X)
