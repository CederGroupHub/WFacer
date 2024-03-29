"""Test sparselm estimator utilities."""
from itertools import chain

import numpy as np
import numpy.testing as npt
import pytest
from sparselm.model import OrdinaryLeastSquares
from sparselm.stepwise import StepwiseEstimator

from WFacer.utils.sparselm_estimators import (
    all_estimator_names,
    estimator_factory,
    is_subclass,
    prepare_estimator,
    supported_estimator_names,
    unsupported_estimator_names,
)


@pytest.mark.parametrize("name", all_estimator_names)
def test_estimator_factory(name):
    # Groups are required, and hierarchy might be as well.
    is_l0 = is_subclass(name, "MIQPl0")
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
    with pytest.raises(ValueError):
        _ = estimator_factory("stepwise-estimator")


@pytest.mark.parametrize("name", supported_estimator_names)
def test_prepare_estimator(subspace, name):
    if is_subclass(name, "MIQPl0") or is_subclass(name, "GroupLasso"):
        # eta not scanned.
        param_grid = {"alpha": [1e-10, 1e-5, 1e-2]}
        estimator = prepare_estimator(
            subspace,
            name,
            "grid-search",
            param_grid=param_grid,
            center_point_external=True,
        )
        num_point_terms = len(subspace.function_inds_by_size[1])
        total_terms = len(subspace.external_terms) + subspace.num_corr_functions
        # Centered by points and ewald; each func is a group itself
        # Groups should always be a 1d array.
        assert isinstance(estimator, StepwiseEstimator)
        full_scope = sorted(set(chain(*estimator.estimator_feature_indices)))
        npt.assert_array_equal(full_scope, np.arange(total_terms, dtype=int))
        assert len(estimator.steps) == 2
        assert tuple(name for name, _ in estimator.steps) == ("center", "main")
        assert [sub.__class__.__name__ for _, sub in estimator.steps] == [
            "GridSearchCV",
            "GridSearchCV",
        ]
        assert estimator.steps[0][1].estimator.__class__.__name__ == "Lasso"
        assert estimator.steps[1][1].estimator.__class__.__name__ == name
        groups = estimator.steps[1][1].estimator.groups
        assert len(np.array(groups).shape) == 1
        assert groups[0] == 0
        assert len(groups) == (subspace.num_corr_functions - num_point_terms - 1)
        npt.assert_array_equal(np.arange(len(groups)), groups)
        if is_subclass(name, "MIQPl0"):
            hierarchy = estimator.steps[1][1].estimator.hierarchy
            assert np.all(np.array(list(chain(*hierarchy))) >= 0)
            function_inds = list(
                range(1 + num_point_terms, subspace.num_corr_functions)
            )
            # print("Num funcs:", subspace.num_corr_functions)
            # print("Func inds:", function_inds)
            # print("func_hierarchy:", subspace.function_hierarchy())
            hierarchy_reconstruct = [
                [function_inds[i] for i in sub_funcs] for sub_funcs in hierarchy
            ]
            hierarchy_standard = [
                [fid for fid in sub if fid >= 1 + num_point_terms]
                for sub in subspace.function_hierarchy()[num_point_terms + 1 :]
            ]
            # print("passed in hierarchy:", estimator.hierarchy)
            # print("reconstructed:", hierarchy_reconstruct)
            # assert False
            assert hierarchy_reconstruct == hierarchy_standard
        assert not estimator.steps[0][1].estimator.fit_intercept
        assert not estimator.steps[1][1].estimator.fit_intercept

        # Not centered.
        estimator = prepare_estimator(
            subspace,
            name,
            "grid-search",
            param_grid=param_grid,
            center_point_external=False,
        )
        assert not isinstance(estimator, StepwiseEstimator)
        assert estimator.__class__.__name__ == "GridSearchCV"
        assert len(np.array(estimator.estimator.groups).shape) == 1
        assert estimator.estimator.groups[0] == 0
        assert len(estimator.estimator.groups) == total_terms
        npt.assert_array_equal(
            np.arange(len(estimator.estimator.groups)), estimator.estimator.groups
        )
        if is_subclass(name, "MIQPl0"):
            assert estimator.estimator.hierarchy == subspace.function_hierarchy()
        assert not estimator.estimator.fit_intercept

    # OLS should never be stepwise.
    if name == "OrdinaryLeastSquares":
        estimator = prepare_estimator(
            subspace, name, "what-ever", None, center_point_external=False
        )
        assert isinstance(estimator, OrdinaryLeastSquares)


@pytest.mark.parametrize("name", supported_estimator_names)
def test_prepare_estimator_sin(subspace_sin, name):
    if is_subclass(name, "MIQPl0") or is_subclass(name, "GroupLasso"):
        param_grid = {"alpha": [1e-10, 1e-5, 1e-2]}
        estimator = prepare_estimator(
            subspace_sin, name, "grid-search", param_grid, center_point_external=True
        )
        num_point_orbs = len(subspace_sin.orbits_by_size[1])
        num_point_funcs = len(subspace_sin.function_inds_by_size[1])
        total_terms = len(subspace_sin.external_terms) + subspace_sin.num_corr_functions
        # Centered by points and ewald; each func is a group itself
        # Groups should always be a 1d array.
        assert isinstance(estimator, StepwiseEstimator)
        full_scope = sorted(set(chain(*estimator.estimator_feature_indices)))
        npt.assert_array_equal(full_scope, np.arange(total_terms, dtype=int))
        assert len(estimator.steps) == 2
        assert tuple(name for name, _ in estimator.steps) == ("center", "main")
        assert [sub.__class__.__name__ for _, sub in estimator.steps] == [
            "GridSearchCV",
            "GridSearchCV",
        ]
        assert estimator.steps[0][1].estimator.__class__.__name__ == "Lasso"
        assert estimator.steps[1][1].estimator.__class__.__name__ == name
        groups = estimator.steps[1][1].estimator.groups
        assert len(np.array(groups).shape) == 1
        assert groups[0] == 0
        assert len(groups) == (subspace_sin.num_corr_functions - num_point_funcs - 1)
        # groups always starts with 0.
        npt.assert_array_equal(
            (
                subspace_sin.function_orbit_ids[1 + num_point_funcs :]
                - num_point_orbs
                - 1
            ),
            groups,
        )
        if is_subclass(name, "MIQPl0"):
            hierarchy = estimator.steps[1][1].estimator.hierarchy
            assert np.all(np.array(list(chain(*hierarchy))) >= 0)
            orbit_inds = list(range(1 + num_point_orbs, subspace_sin.num_orbits))
            # print("Num orbits:", subspace_sin.num_orbits)
            # print("len orbits:", len(subspace_sin.orbits))
            # print("Orbit inds:", orbit_inds)
            # print("orbit_hierarchy:", subspace_sin.orbit_hierarchy())
            hierarchy_reconstruct = [
                [orbit_inds[i] for i in sub_orbs] for sub_orbs in hierarchy
            ]
            hierarchy_standard = [
                [oid for oid in sub if oid >= 1 + num_point_orbs]
                for sub in subspace_sin.orbit_hierarchy()[num_point_orbs + 1 :]
            ]
            # print("passed in hierarchy:", estimator.hierarchy)
            # print("reconstructed:", hierarchy_reconstruct)
            # assert False
            assert hierarchy_reconstruct == hierarchy_standard
        assert not estimator.steps[0][1].estimator.fit_intercept
        assert not estimator.steps[1][1].estimator.fit_intercept

        # Not centered.
        estimator = prepare_estimator(
            subspace_sin, name, "grid-search", param_grid, center_point_external=False
        )
        assert not isinstance(estimator, StepwiseEstimator)
        assert estimator.__class__.__name__ == "GridSearchCV"
        assert len(np.array(estimator.estimator.groups).shape) == 1
        assert estimator.estimator.groups[0] == 0
        assert len(estimator.estimator.groups) == total_terms
        npt.assert_array_equal(
            np.append(
                subspace_sin.function_orbit_ids,
                np.arange(len(subspace_sin.external_terms)) + subspace_sin.num_orbits,
            ),
            estimator.estimator.groups,
        )
        if is_subclass(name, "MIQPl0"):
            assert estimator.estimator.hierarchy == subspace_sin.orbit_hierarchy()
        assert not estimator.estimator.fit_intercept

    # OLS should never be stepwise.
    if name == "OrdinaryLeastSquares":
        estimator = prepare_estimator(
            subspace_sin, name, "what-ever", None, center_point_external=True
        )
        assert isinstance(estimator, OrdinaryLeastSquares)
