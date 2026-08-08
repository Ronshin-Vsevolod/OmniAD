"""
The one genuinely cross-domain test in the suite: capability slugs
mean the same thing regardless of domain, so a single generic check
applies to every registered algorithm.
"""
from __future__ import annotations

import pytest

from omniad import get_detector, list_detectors
from omniad.core.mixins import CAPABILITY_REGISTRY
from tests.contracts.capability_checks import CAPABILITY_CHECKS
from tests.support import AllDomainDatasets, fast_kwargs

ALL_ALGOS = list_detectors()


def test_every_capability_has_a_contract_check() -> None:
    registered = {info.slug for info in CAPABILITY_REGISTRY}
    missing = registered - set(CAPABILITY_CHECKS)
    assert not missing, f"No contract check for capabilities: {missing}"


@pytest.mark.parametrize("algo_name", ALL_ALGOS)
def test_declared_capabilities_are_truthful(
    algo_name: str, all_domain_datasets: AllDomainDatasets
) -> None:
    X_train, X_test = all_domain_datasets.for_algo(algo_name)

    model = get_detector(algo_name, **fast_kwargs(algo_name))
    model.fit(X_train)

    active = [slug for slug, ok in model.capabilities.items() if ok]
    active.sort(key=lambda s: s == "incremental_learning")  # mutating check last

    for slug in active:
        CAPABILITY_CHECKS[slug](model, X_test)
