"""Test running jobs."""
from copy import deepcopy
from itertools import chain

import numpy as np
import numpy.testing as npt
import pytest
from atomate2.vasp.schemas.calculation import Calculation
from atomate2.vasp.schemas.task import TaskDocument
from jobflow import Flow, Job, Response
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core import Element, Structure
from pymatgen.entries.computed_entries import ComputedEntry
from smol.cofe.space.domain import Vacancy

from AceCE.jobs import (
    calculate_structures,
    enumerate_structures,
    fit_calculations,
    initialize_document,
    parse_calculations,
    update_document,
)
from AceCE.preprocessing import get_prim_specs

from .utils import execute_job_function


def get_element_structure(structure):
    # Re-create the structure without any decoration.
    all_coords = []
    all_species = []
    for site in structure:
        if not isinstance(site.specie, Vacancy):
            all_coords.append(site.frac_coords)
            all_species.append(site.specie.symbol)

    s = Structure(structure.lattice, all_species, all_coords)
    return s


def gen_fake_taskdoc(structure, energy):
    """Generate a fake TaskDocument from structure.

    This function only fakes entry and structure! All other fields
    in TaskDocument will be empty.
    Args:
        structure (Structure):
            A fake structure.
        energy (float):
            A fake energy to assign to the structure.
    Returns:
        TaskDocument.
    """
    s = get_element_structure(structure)

    entry = ComputedEntry(s.composition, energy, data={"some_test": 100})
    # Need to insert a successful calculation in calcs_reversed as well.
    fake_calc = Calculation(has_vasp_completed="successful")

    return TaskDocument(structure=s, entry=entry, calcs_reversed=[fake_calc])


# Add more if new tests are required.
fix_charge_settings = {"Li": 1, "Ca": 1, "Br": -1}


@pytest.fixture
def initial_document(prim):
    specs = get_prim_specs(prim)
    options = {
        "cutoffs": {2: 7, 3: 4},
        "objective_num_sites": 64,
        "comp_enumeration_step": 16,
        "num_structs_per_iter_init": 50,
        "num_structs_per_iter_add": 30,
        "other_properties": ["some_test"],
        "estimator_type": "lasso",
        "optimizer_type": "grid-search",
        "param_grid": {"alpha": 2 ** np.linspace(-25, 3, 15)},
    }
    if specs["charge_decorated"]:
        options["decorator_types"] = ["fixed-charge"]
        elements = [el.symbol for el in prim.composition.element_composition.keys()]
        options["decorator_kwargs"] = [
            {"labels": {el: fix_charge_settings[el] for el in elements}}
        ]

    init_job = initialize_document(prim, options=options)
    return execute_job_function(init_job)


@pytest.fixture
def enum_output(initial_document):
    enum_job = enumerate_structures(initial_document)
    return execute_job_function(enum_job)


@pytest.fixture
def coefs_truth(initial_document):
    space = initial_document.cluster_subspace
    coefs = np.random.random(space.num_corr_functions + len(space.external_terms))
    coefs[0] = 1
    return coefs


@pytest.fixture
def calc_output(coefs_truth, enum_output):
    # Fake TaskDocuments for testing.
    num_structures = len(enum_output["new_features"])
    fake_energies = (
        np.dot(enum_output["new_features"], coefs_truth)
        + np.random.normal(size=(num_structures,)) * 0.001
    )
    taskdocs = [
        gen_fake_taskdoc(s, e)
        for s, e in zip(enum_output["new_structures"], fake_energies)
    ]
    return taskdocs


@pytest.fixture
def parse_output(calc_output, enum_output, initial_document):
    parse_job = parse_calculations(calc_output, enum_output, initial_document)
    return execute_job_function(parse_job)


@pytest.fixture
def fit_output(parse_output, initial_document):
    fit_job = fit_calculations(parse_output, initial_document)
    return execute_job_function(fit_job)


def test_initial_document(initial_document):
    assert initial_document.project_name == "ace-work"
    assert initial_document.ce_options["cutoffs"] == {2: 7, 3: 4}

    option_cutoffs = initial_document.ce_options["cutoffs"]
    generated_cutoffs = initial_document.cluster_subspace.cutoffs
    for k in generated_cutoffs:
        assert k in option_cutoffs
        assert option_cutoffs[k] >= generated_cutoffs[k]

    # Assert all aliases have been removed.
    sc_matrices = initial_document.supercell_matrices
    cluster_subspace = initial_document.cluster_subspace
    alias = []
    for m in sc_matrices:
        alias_m = cluster_subspace.get_aliased_orbits(m)
        alias_m = {
            sorted(sub_orbit)[0]: set(sorted(sub_orbit)[1:]) for sub_orbit in alias_m
        }
        alias.append(alias_m)
    to_remove = deepcopy(alias[0])
    for alias_m in alias[1:]:
        for key in to_remove:
            if key in alias_m:
                to_remove[key] = to_remove[key].intersection(alias_m[key])
    to_remove = sorted(list(set(chain(*to_remove.values()))))

    assert len(to_remove) == 0


def test_enumerate_structures(initial_document, enum_output):
    cluster_subspace = initial_document.cluster_subspace
    for s, m, f in zip(
        enum_output["new_structures"],
        enum_output["new_sc_matrices"],
        enum_output["new_features"],
    ):
        f0 = cluster_subspace.corr_from_structure(s, scmatrix=m)
        npt.assert_array_almost_equal(f, f0)
    # Enumerated structures printed out and found to be reasonable.
    # from monty.serialization import dumpfn
    #
    # structures = enum_output["new_structures"]
    # name = "".join(
    #     [el.symbol for el in structures[0].composition.element_composition.keys()]
    # )
    # dumpfn(structures, f"./structures_{name}.json")


def test_calculate_structures(initial_document, enum_output):
    calc_job = calculate_structures(enum_output, initial_document)
    # Will not perform the actual calculation here, only check flow structure.
    response = execute_job_function(calc_job)

    assert isinstance(response, Response)
    assert isinstance(response.replace, Flow)

    assert initial_document.project_name == "ace-work"
    flow = response.replace
    assert isinstance(flow.output, list)
    assert flow.name == "ace-work_iter_0_calculations"
    assert len(flow.output) == len(enum_output["new_structures"])
    # Each sub-flow has 3 jobs. (Relax, tight (default) and static.)
    assert len(flow.jobs) == len(enum_output["new_structures"])
    assert isinstance(flow.jobs[0], Flow)
    assert flow.jobs[0].name == "ace-work_iter_0_enum_0"
    job = flow.jobs[0].jobs[0]
    assert isinstance(job, Job)
    assert len(flow.jobs[0].jobs) == 3
    assert job.name == "ace-work_iter_0_enum_0_relax"


def test_parse_calculations(enum_output, parse_output):
    n_enum = len(enum_output["new_structures"])
    assert 1 <= n_enum <= 50
    # Sometimes can't get 50 structures for LiCaBr. This should be fine because
    # LiCaBr requires low ewald energy, that greatly restricts its sample space.

    # Assert all structures can be correctly mapped and not duplicated, because
    # they are all the enumerated result of the first iteration, and are not
    # supposed to duplicate.
    assert parse_output["wrangler"].num_structures == n_enum

    # Assert all structures are properly decorated.
    sm = StructureMatcher()
    specs = get_prim_specs(parse_output["wrangler"].cluster_subspace.structure)
    for ent, und in zip(
        parse_output["wrangler"].entries, parse_output["undecorated_entries"]
    ):
        assert sm.fit(get_element_structure(ent.structure), und.structure)
        assert ent.structure.charge == 0
        if specs["charge_decorated"]:
            carry_charge = [
                (
                    not isinstance(site.specie, (Element, Vacancy))
                    or site.species.oxi_state != 0
                )
                for site in ent.structure
            ]
            assert np.any(carry_charge)
    # Assert other properties are correctly parsed.
    assert len(parse_output["undecorated_entries"]) == n_enum
    assert len(parse_output["computed_properties"]) == n_enum
    assert parse_output["computed_properties"][0]["some_test"] == 100


def test_fit_calculations(coefs_truth, parse_output, fit_output):
    # Check type of the fit: lasso has only 1 parameter.
    assert len(fit_output["params"]) == 1

    # The quality of the fit should not be too bad.
    data_wrangler = parse_output["wrangler"]
    e_predict = np.dot(data_wrangler.feature_matrix, fit_output["coefs"])
    e = data_wrangler.get_property_vector("energy")
    r2 = 1 - np.sum((e_predict - e) ** 2) / (np.var(e) * len(e))
    assert r2 >= 0.8


def test_update_document(enum_output, parse_output, fit_output, initial_document):
    update_job = update_document(
        enum_output, parse_output, fit_output, initial_document
    )
    new_document = execute_job_function(update_job)
    # print("prim:", initial_document.cluster_subspace.structure)
    # print("Num enumeration:", len(enum_output["new_structures"]))
    # print("Num parsed structures:", parse_output["wrangler"].num_structures)
    # print("parsed structures:", parse_output["wrangler"].structures)
    # print("fit output:", fit_output)

    assert new_document.last_iter_id == 0
    n_structures = len(enum_output["new_structures"])
    assert n_structures <= 50  # Sometimes can't get 50 structures for LiCaBr.
    assert new_document.data_wrangler.num_structures == n_structures
    assert len(new_document.undecorated_entries) == n_structures
    assert len(new_document.enumerated_structures) == n_structures
    assert len(new_document.coefs_history) == 1
    npt.assert_array_almost_equal(new_document.coefs_history[-1], fit_output["coefs"])
