"""Test sparselm estimator untilities."""
from itertools import chain

import pytest
import numpy as np
import numpy.testing as npt

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


@pytest.mark.parametrize("name", supported_estimator_names)
def test_prepare_estimator(subspace, name):
    if is_subclass(name, "MIQP_L0") or is_subclass(name, "GroupLasso"):
        estimator = prepare_estimator(subspace, name)
        num_point_terms = len(subspace.function_inds_by_size[1])
        total_terms = len(subspace.external_terms) + subspace.num_corr_functions
        # Centered by points and ewald; each func is a group itself
        # Groups should always be a 1d array.
        assert len(np.array(estimator.groups).shape) == 1
        assert estimator.groups[0] == 0
        assert len(estimator.groups) == (subspace.num_corr_functions
                                         - num_point_terms - 1)
        npt.assert_array_equal(np.arange(len(estimator.groups)),
                               estimator.groups)
        if is_subclass(name, "MIQP_L0"):
            assert np.all(np.array(list(chain(*estimator.hierarchy))) >= 0)
            function_inds = list(range(1 + num_point_terms,
                                       subspace.num_corr_functions))
            # print("Num funcs:", subspace.num_corr_functions)
            # print("Func inds:", function_inds)
            # print("func_hierarchy:", subspace.function_hierarchy())
            hierarchy_reconstruct = [[function_inds[i] for i in sub_funcs]
                                     for sub_funcs in estimator.hierarchy]
            hierarchy_standard = [[fid for fid in sub if fid >= 1 + num_point_terms]
                                  for sub in
                                  subspace.function_hierarchy()[num_point_terms + 1:]]
            # print("passed in hierarchy:", estimator.hierarchy)
            # print("reconstructed:", hierarchy_reconstruct)
            # assert False
            assert hierarchy_reconstruct == hierarchy_standard
        assert not estimator.fit_intercept

        # Not centered.
        estimator = prepare_estimator(subspace, name,
                                      center_point_external=False)
        assert len(np.array(estimator.groups).shape) == 1
        assert estimator.groups[0] == 0
        assert len(estimator.groups) == total_terms
        npt.assert_array_equal(np.arange(len(estimator.groups)),
                               estimator.groups)
        if is_subclass(name, "MIQP_L0"):
            assert estimator.hierarchy == subspace.function_hierarchy()
        assert not estimator.fit_intercept


@pytest.mark.parametrize("name", supported_estimator_names)
def test_prepare_estimator_sin(subspace_sin, name):
    if is_subclass(name, "MIQP_L0") or is_subclass(name, "GroupLasso"):
        estimator = prepare_estimator(subspace_sin, name)
        num_point_orbs = len(subspace_sin.orbits_by_size[1])
        num_point_funcs = len(subspace_sin.function_inds_by_size[1])
        total_terms = len(subspace_sin.external_terms) + subspace_sin.num_corr_functions
        # Centered by points and ewald; each func is a group itself
        # Groups should always be a 1d array.
        assert len(np.array(estimator.groups).shape) == 1
        assert estimator.groups[0] == 0
        assert len(estimator.groups) == (subspace_sin.num_corr_functions
                                         - num_point_funcs - 1)
        # groups always starts with 0.
        npt.assert_array_equal((subspace_sin.function_orbit_ids[1 + num_point_funcs:]
                                - num_point_orbs - 1),
                               estimator.groups)
        if is_subclass(name, "MIQP_L0"):
            assert np.all(np.array(list(chain(*estimator.hierarchy))) >= 0)
            orbit_inds = list(range(1 + num_point_orbs,
                                    subspace_sin.num_orbits))
            # print("Num orbits:", subspace_sin.num_orbits)
            # print("len orbits:", len(subspace_sin.orbits))
            # print("Orbit inds:", orbit_inds)
            # print("orbit_hierarchy:", subspace_sin.orbit_hierarchy())
            hierarchy_reconstruct = [[orbit_inds[i] for i in sub_orbs]
                                     for sub_orbs in estimator.hierarchy]
            hierarchy_standard = [[oid for oid in sub if oid >= 1 + num_point_orbs]
                                  for sub in
                                  subspace_sin.orbit_hierarchy()[num_point_orbs + 1:]]
            # print("passed in hierarchy:", estimator.hierarchy)
            # print("reconstructed:", hierarchy_reconstruct)
            # assert False
            assert hierarchy_reconstruct == hierarchy_standard
        assert not estimator.fit_intercept

        # Not centered.
        estimator = prepare_estimator(subspace_sin, name,
                                      center_point_external=False)
        assert len(np.array(estimator.groups).shape) == 1
        assert estimator.groups[0] == 0
        assert len(estimator.groups) == total_terms
        npt.assert_array_equal(np.append(subspace_sin.function_orbit_ids,
                                         np.arange(len(subspace_sin.external_terms))
                                         + subspace_sin.num_orbits),
                               estimator.groups)
        if is_subclass(name, "MIQP_L0"):
            assert estimator.hierarchy == subspace_sin.orbit_hierarchy()
        assert not estimator.fit_intercept
