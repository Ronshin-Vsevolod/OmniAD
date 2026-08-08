from omniad.core.mixins import CAPABILITY_REGISTRY
from tests.doubles import DummyDetector, DummyStreamingDetector


def test_plain_detector_has_no_extra_capabilities() -> None:
    assert DummyDetector.get_capabilities() == set()


def test_streaming_detector_declares_incremental_learning() -> None:
    assert DummyStreamingDetector.get_capabilities() == {"incremental_learning"}


def test_capabilities_property_matches_get_capabilities() -> None:
    model = DummyStreamingDetector()
    active = {slug for slug, ok in model.capabilities.items() if ok}
    assert active == type(model).get_capabilities()


def test_repr_contains_class_name_and_status() -> None:
    text = repr(DummyDetector())
    assert "DummyDetector" in text and "unfitted" in text


def test_str_lists_capabilities_or_says_none() -> None:
    assert "no extra capabilities" in str(DummyDetector())

    info = next(i for i in CAPABILITY_REGISTRY if i.slug == "incremental_learning")
    streaming_repr = str(DummyStreamingDetector())
    assert info.label in streaming_repr
    assert info.usage in streaming_repr


def test_repr_html_does_not_crash() -> None:
    assert "DummyDetector" in DummyDetector()._repr_html_()
