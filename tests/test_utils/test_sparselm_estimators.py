"""Test sparselm estimator untilities."""
import pytest

from CEAuto.utils.sparselm_estimators import (unsupported_estimator_names,
                                              is_subclass,
                                              estimator_factory,
                                              prepare_estimator,
                                              all_estimator_names,
                                              supported_estimator_names)


@pytest.mark.parametrize("name", all_estimator_names)
def test_estimator_factory(name):
    # Groups are required, and hierarchy might be as well.
    is_l0 = is_subclass(name, "MIQP_L0")
    # Only groups are required.
    is_group = is_subclass(name, "GroupLasso")
    # sparse_bound would also be needed.
    is_subset = is_subclass(name, "BestSubsetSelection")
    kwargs = {}
    if is_l0 or is_group:
        kwargs["groups"] = [[i] for i in range(10)]
    if is_subset:
        kwargs["sparse_bound"] = 5
    assert "OverlapGroupLasso" in all_estimator_names
    assert "OverlapGroupLasso" in unsupported_estimator_names
    if name in unsupported_estimator_names:
        with pytest.raises(ValueError):
            _ = estimator_factory(name, **kwargs)
    else:
        estimator = estimator_factory(name, **kwargs)
        assert estimator.__class__.__name__ == name


def test_bad_estimator_name():
    with pytest.raises(ValueError):
        _ = estimator_factory("what-ever")
    with pytest.raises(ValueError):
        _ = estimator_factory("overlap-group-lasso")


# TODO: finished and pass these tests.
@pytest.mark.parametrize("name", supported_estimator_names)
def test_prepare_estimator(subspace, name):
    return


@pytest.mark.parametrize("name", supported_estimator_names)
def test_prepare_estimator_sin(subspace_sin, name):
    return
