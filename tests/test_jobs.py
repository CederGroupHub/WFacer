"""Test running jobs."""
from copy import deepcopy
from itertools import chain

import numpy as np
import numpy.testing as npt
import pytest
from emmet.core.tasks import TaskDoc  # atomate2 >= 0.0.11.
from emmet.core.vasp.calculation import Calculation  # atomate2 >= 0.0.11.
from jobflow import Flow, Job, Maker, OnMissing, Response
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core import Element, Lattice, Structure
from pymatgen.entries.computed_entries import ComputedEntry
from smol.cofe.space.domain import Vacancy

from WFacer.jobs import (
    _get_iter_id_from_enum_id,
    _get_structure_job_maker,
    calculate_structures_job,
    enumerate_structures,
    fit_calculations,
    initialize_document,
    parse_calculations,
    update_document,
)
from WFacer.preprocessing import get_prim_specs

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
    """Generate a fake TaskDoc from structure.

    This function only fakes entry and structure! All other fields
    in TaskDoc will be empty.
    Args:
        structure (Structure):
            A fake structure.
        energy (float):
            A fake energy to assign to the structure.
    Returns:
        TaskDoc.
    """
    s = get_element_structure(structure)

    entry = ComputedEntry(s.composition, energy, data={"some_test": 100})
    # Need to insert a successful calculation in calcs_reversed as well.
    fake_calc = Calculation(has_vasp_completed="successful")

    return TaskDoc(structure=s, entry=entry, calcs_reversed=[fake_calc])


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

    return initialize_document(prim, options=options)


@pytest.fixture
def enum_output(initial_document):
    return enumerate_structures(initial_document)


@pytest.fixture
def coefs_truth(initial_document):
    space = initial_document.cluster_subspace
    coefs = np.random.random(space.num_corr_functions + len(space.external_terms))
    coefs[0] = 1
    return coefs


@pytest.fixture
def calc_output(coefs_truth, enum_output):
    # Fake TaskDocs for testing.
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
    return parse_calculations(calc_output, enum_output, initial_document)


@pytest.fixture
def fit_output(parse_output, initial_document):
    return fit_calculations(parse_output, initial_document)


def test_initial_document(initial_document):
    assert initial_document.project_name == "ace-work"
    assert initial_document.ce_options["cutoffs"] == {2: 7, 3: 4}
    assert initial_document.enumerated_features is None
    assert initial_document.last_iter_id == -1

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


def test_structure_single_job():
    # CP2K is only supported after atomate2 >= 0.0.11.
    valid_makers = [
        "atomate2.vasp.jobs.core:relax-maker",
        "atomate2.vasp.jobs.core:static-maker",
        "atomate2.vasp.jobs.core:TightRelaxMaker",
        # "atomate2.cp2k.jobs.core:relax-maker",
        # "atomate2.cp2k.jobs.core:static-maker",
    ]
    # These should warn and return None.
    wrong_makers = [
        "atomate2.whatever.jobs.core:relax-maker",
        "atomate2.vasp.jobs:relax-maker",
        "atomate2.cp2k.jobs.core:tight-relax-maker",
        "atomate2.forcefields.jobs:CHGNet-tight-relax-maker",
    ]
    # These should throw NonImplementedError.
    unsupported_makers = [
        "atomate2.amset.jobs:amset-maker",
        "atomate2.lobster.jobs:lobster-maker",
    ]

    # TODO: test these after the next atomate2 release.
    # force_makers = [
    # "atomate2.forcefields.jobs:CHGNetRelaxMaker",
    # "atomate2.forcefields.jobs:CHGNetStaticMaker",
    # "atomate2.forcefields.jobs:M3GNetRelaxMaker",
    # "atomate2.forcefields.jobs:M3GNetStaticMaker",
    # ]

    for maker_name in valid_makers:
        maker = _get_structure_job_maker(maker_name)
        assert isinstance(maker, Maker)
        assert maker.stop_children_kwargs == {"handle_unsuccessful": "error"}
        assert maker.input_set_generator is not None

    for maker_name in unsupported_makers:
        with pytest.raises(NotImplementedError):
            _ = _get_structure_job_maker(maker_name)

    # None is returned.
    for maker_name in wrong_makers:
        assert _get_structure_job_maker(maker_name) is None

    # Test a specific case of input set generator.
    maker = _get_structure_job_maker(
        "atomate2.vasp.jobs.core:relax-maker",
        generator_kwargs={
            "user_incar_settings": {"ENCUT": 1000},
        },
    )
    s = Structure(Lattice.cubic(3.0), ["Co2+", "O2-"], [[0, 0, 0], [0.5, 0.5, 0.5]])
    incar = maker.input_set_generator.get_input_set(s, potcar_spec=True).incar
    assert incar["ENCUT"] == 1000


def test_calculate_structures(initial_document, enum_output):
    calc_job = calculate_structures_job(enum_output, initial_document)
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
    assert isinstance(flow.jobs[0].jobs[0], Job)
    for jobs in flow.jobs:
        assert len(jobs.jobs) == 3
        # Does not allow failure before relaxation.
        assert jobs.jobs[0].config.on_missing_references == OnMissing.ERROR
        # Allows failure for relaxation.
        for job in jobs.jobs[1:]:
            assert job.config.on_missing_references == OnMissing.NONE
    job = flow.jobs[0].jobs[0]
    assert job.name == "ace-work_iter_0_enum_0_relax"
    job = flow.jobs[0].jobs[-1]
    assert job.name == "ace-work_iter_0_enum_0_static"


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
    all_iter_ids = []
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
        # Iteration indices are correctly parsed!
        assert ent.data["properties"]["spec"]["iter_id"] == _get_iter_id_from_enum_id(
            ent.data["properties"]["spec"]["enum_id"], 50, 30
        )
        all_iter_ids.append(ent.data["properties"]["spec"]["iter_id"])
    assert sorted(set(all_iter_ids)) == list(range(max(all_iter_ids) + 1))
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
    new_document = update_document(
        enum_output, parse_output, fit_output, initial_document
    )
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
