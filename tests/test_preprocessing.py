"""Test preprocessing functions."""
import numpy as np
import numpy.testing as npt
import yaml
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core import Lattice
from smol.cofe.extern import EwaldTerm
from smol.cofe.space.domain import get_species

from WFacer.jobs import _preprocess_options
from WFacer.preprocessing import (
    construct_prim,
    get_cluster_subspace,
    get_initial_ce_coefficients,
    get_prim_specs,
    parse_comp_constraints,
    reduce_prim,
)


def test_reduce_prim(prim):
    sm = StructureMatcher()
    sc = prim.copy()
    sc.make_supercell([[-1, 1, 1], [1, -1, 1], [1, 1, -1]])

    assert sm.fit(reduce_prim(sc), prim)


def test_prim_specs():
    bits = [["Li+", "Mn2+", "Mn3+", "Vacancy"], ["O2-", "F-"]]
    bits = [[get_species(p) for p in sl_bits] for sl_bits in bits]
    sl_sites = [[0, 1], [2]]
    lat = Lattice.from_parameters(2, 2, 2, 60, 60, 60)
    frac_coords = [[0.25, 0.25, 0.25], [0.75, 0.75, 0.75], [0, 0, 0]]
    prim = construct_prim(bits, sl_sites, lat, frac_coords)
    specs = get_prim_specs(prim)

    assert bits == specs["bits"]
    assert sl_sites == specs["sublattice_sites"]
    assert specs["charge_decorated"]
    assert np.isclose(specs["nn_distance"], np.sqrt(3) / np.sqrt(2))


def test_cluster_subspace(prim):
    specs = get_prim_specs(prim)
    space = get_cluster_subspace(prim, specs["charge_decorated"], specs["nn_distance"])
    assert space.basis_type == "indicator"  # This is default.
    if specs["charge_decorated"]:
        assert len(space.external_terms) == 1
        assert isinstance(space.external_terms[0], EwaldTerm)
    else:
        assert len(space.external_terms) == 0

    d_nn = specs["nn_distance"]
    cutoffs = {1: np.inf, 2: 3.5 * d_nn, 3: 2 * d_nn, 4: 2 * d_nn}
    filtered_orbits = [
        orbit
        for orbit in space.orbits
        if (
            1 <= len(orbit.base_cluster.sites) <= 4
            and orbit.base_cluster.diameter <= cutoffs[len(orbit.base_cluster.sites)]
        )
    ]

    assert filtered_orbits == space.orbits


def test_parse_comp_constraints():
    bits = [["Li+", "Mn2+", "Mn3+", "Vacancy"], ["O2-", "F-"]]
    bits = [[get_species(b) for b in sl_bits] for sl_bits in bits]
    sl_sizes = [2, 1]
    species_constraints = {"Li+": (0.1, 0.2), "Mn2+": (0.3, 0.5), "O2-": 0.8}
    geq_constraints = [({"Li+": 2, "Mn3+": 1}, 1)]
    d = {
        "species_concentration_constraints": species_constraints,
        "geq_constraints": geq_constraints,
        "leq_constraints": [],
        "eq_constraints": [],
    }
    eqs, leqs, geqs = parse_comp_constraints(d, bits, sl_sizes)
    leqs = [np.append(left, right) for left, right in leqs]
    geqs = [np.append(left, right) for left, right in geqs]
    leqs_standard = [
        [1, 0, 0, 0, 0, 0, 0.4],
        [0, 1, 0, 0, 0, 0, 1.0],
        [0, 0, 0, 0, 1, 0, 0.8],
    ]
    geqs_standard = [
        [2, 0, 1, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0.2],
        [0, 1, 0, 0, 0, 0, 0.6],
        [0, 0, 0, 0, 1, 0, 0],
    ]
    assert len(eqs) == 0
    npt.assert_array_equal(leqs, leqs_standard)
    npt.assert_array_equal(geqs, geqs_standard)


def test_options():
    options = _preprocess_options({})
    # Update this yaml if new options have been implemented!
    with open("./data/default_options.yaml") as fin:
        default_options = yaml.load(fin, Loader=yaml.SafeLoader)
    assert set(options.keys()) == set(default_options.keys())
    for k in options.keys():
        if default_options[k] is not None:
            if isinstance(default_options[k], list):
                npt.assert_array_almost_equal(default_options[k], options[k])
            else:
                assert default_options[k] == options[k]


def test_initialize_coefs(subspace):
    coefs = get_initial_ce_coefficients(subspace)
    npt.assert_array_almost_equal(
        coefs[: subspace.num_corr_functions], np.zeros(subspace.num_corr_functions)
    )
    if len(subspace.external_terms) > 0:
        npt.assert_array_almost_equal(
            coefs[-len(subspace.external_terms) :],
            np.ones(len(subspace.external_terms)),
        )
