from omniad import describe, list_detectors
from omniad.registry import _REGISTRY


def test_list_detectors_returns_all_by_default() -> None:
    assert set(list_detectors()) == set(_REGISTRY.keys())


def test_list_detectors_filters_by_known_stable_domain_membership() -> None:
    """
    IsolationForest/LSTM domain membership is a structural fact, not a
    tunable one — it changing would be an intentional registry edit,
    not ML-quality tuning, so pinning it here is appropriate.
    """
    tabular = list_detectors(domain="tabular")
    assert "IsolationForest" in tabular
    assert "LSTM" not in tabular


def test_list_detectors_filters_by_known_stable_capability() -> None:
    assert "IsolationForest" in list_detectors(capability="feature_importance")


def test_describe_does_not_instantiate_or_fit(fake_registry_entry: str) -> None:
    info = describe(fake_registry_entry, verbose=False)
    assert info.class_name == "_TestOnlyDummyAdapter"
    assert info.domain == "tabular"
    assert info.params["n_estimators"] == 7


def test_describe_verbose_prints_summary(fake_registry_entry: str, capsys) -> None:
    describe(fake_registry_entry, verbose=True)
    assert fake_registry_entry in capsys.readouterr().out


def test_list_detectors_capability_filter_skips_unimportable_algorithms(
    broken_registry_entry: str,
) -> None:
    """
    list_detectors(capability=...) must not let one unimportable
    algorithm (e.g. missing extras) crash the whole query — it should
    be silently skipped, same as get_detector() would raise ImportError
    for that name specifically, not for everyone else's query.
    """
    result = list_detectors(capability="feature_importance")
    assert "IsolationForest" in result
    assert broken_registry_entry not in result
