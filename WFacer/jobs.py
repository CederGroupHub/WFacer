"""Unitary jobs used by Maker."""
import logging
from copy import deepcopy
from warnings import warn

import numpy as np
from atomate2.vasp.jobs.core import RelaxMaker, StaticMaker, TightRelaxMaker
from atomate2.vasp.schemas.calculation import Status
from atomate2.vasp.sets.core import (
    RelaxSetGenerator,
    StaticSetGenerator,
    TightRelaxSetGenerator,
)
from jobflow import Flow, OnMissing, Response, job
from pymatgen.analysis.elasticity.strain import Deformation
from smol.cofe import ClusterExpansion
from smol.moca import CompositionSpace

from .enumeration import (
    enumerate_compositions_as_counts,
    enumerate_matrices,
    generate_training_structures,
    truncate_cluster_subspace,
)
from .fit import fit_ecis_from_wrangler
from .preprocessing import (
    get_cluster_subspace,
    get_initial_ce_coefficients,
    get_prim_specs,
    parse_comp_constraints,
    process_calculation_options,
    process_composition_options,
    process_convergence_options,
    process_decorator_options,
    process_fit_options,
    process_structure_options,
    process_subspace_options,
    process_supercell_options,
    reduce_prim,
)
from .schema import CeOutputsDocument
from .specie_decorators import PmgGuessChargeDecorator, decorator_factory
from .utils.task_document import get_entry_from_taskdoc
from .wrangling import CeDataWrangler

log = logging.getLogger(__name__)


def _preprocess_options(options):
    sc_options = process_supercell_options(options)
    comp_options = process_composition_options(options)
    struct_options = process_structure_options(options)
    calc_options = process_calculation_options(options)
    decor_options = process_decorator_options(options)
    space_options = process_subspace_options(options)
    fit_options = process_fit_options(options)
    conv_options = process_convergence_options(options)

    options = dict()
    options.update(sc_options)
    options.update(comp_options)
    options.update(struct_options)
    options.update(calc_options)
    options.update(decor_options)
    options.update(space_options)
    options.update(fit_options)
    options.update(conv_options)
    return options


def _enumerate_structures(
    subspace,
    coefs,
    iter_id,
    sc_matrices,
    compositions,
    enumerated_structures,
    enumerated_features,
    options,
):
    """Enumerate structures in an interaction."""
    ce = ClusterExpansion(subspace, coefs)
    keep_ground_states = options["keep_ground_states"]
    num_structs_init = options["num_structs_per_iter_init"]
    num_structs_add = options["num_structs_per_iter_add"]
    mc_generator_kwargs = options["sample_generator_kwargs"]
    init_method = options["init_method"]
    add_method = options["add_method"]
    n_parallel = options["n_parallel"]
    duplicacy_criteria = options["duplicacy_criteria"]
    if iter_id == 0:
        method = init_method
        n_select = num_structs_init
    else:
        method = add_method
        n_select = num_structs_add
    new_structures, new_sc_matrices, new_features = generate_training_structures(
        ce,
        sc_matrices,
        compositions,
        enumerated_structures,
        enumerated_features,
        keep_ground_states,
        n_select,
        mc_generator_kwargs,
        n_parallel,
        duplicacy_criteria=duplicacy_criteria,
        method=method,
    )
    return new_structures, new_sc_matrices, new_features


def _get_vasp_makers(options):
    """Get required vasp makers."""
    relax_gen_kwargs = options["relax_generator_kwargs"]
    relax_generator = RelaxSetGenerator(**relax_gen_kwargs)
    relax_maker_kwargs = options["relax_maker_kwargs"]
    # Force throwing out an error instead of defusing children, as parsing and
    # fitting jobs are children of structure jobs and will be defused
    # as well if not taken care of!

    # Error handling will be taken care of by lost run detection and
    # fixing functionalities in WFacer.fireworks_patches.
    relax_maker_kwargs["stop_children_kwargs"] = {"handle_unsuccessful": "error"}
    relax_maker = RelaxMaker(input_set_generator=relax_generator, **relax_maker_kwargs)
    static_gen_kwargs = options["static_generator_kwargs"]
    static_generator = StaticSetGenerator(**static_gen_kwargs)
    static_maker_kwargs = options["static_maker_kwargs"]
    static_maker_kwargs["stop_children_kwargs"] = {"handle_unsuccessful": "error"}
    static_maker = StaticMaker(
        input_set_generator=static_generator, **static_maker_kwargs
    )
    tight_gen_kwargs = options["tight_generator_kwargs"]
    tight_generator = TightRelaxSetGenerator(**tight_gen_kwargs)
    tight_maker_kwargs = options["tight_maker_kwargs"]
    tight_maker_kwargs["stop_children_kwargs"] = {"handle_unsuccessful": "error"}
    tight_maker = TightRelaxMaker(
        input_set_generator=tight_generator, **tight_maker_kwargs
    )
    return relax_maker, static_maker, tight_maker


def _check_flow_convergence(taskdoc):
    """Check vasp convergence for a single structure."""
    try:
        status = taskdoc.calcs_reversed[0].has_vasp_completed
        if status == Status.FAILED:
            return False
    except AttributeError:
        return False
    return True


def _get_decorators(options, is_charge_decorated):
    """Get required decorators."""
    decorators = [
        decorator_factory(dt, **kw)
        for dt, kw in zip(options["decorator_types"], options["decorator_kwargs"])
    ]
    decorated_properties = [d.decorated_prop_name for d in decorators]
    if len(decorated_properties) != len(set(decorated_properties)):
        raise ValueError(
            "Can not use multiple decorators for decorating" " the same property!!"
        )
    if is_charge_decorated and len(decorators) == 0:
        warn(
            "Cluster expansion is charge decorated, but"
            " no charge decoration method is specified."
            " Use default PmgGuessCharge at your risk!"
        )
        decorators.append(PmgGuessChargeDecorator())
    return decorators


def _filter_out_failed_entries(entries, entry_ids):
    """Mark failed entries as none and return successful indices."""
    new_entry_ids = [eid for ent, eid in zip(entries, entry_ids) if ent is not None]
    new_entries = [ent for ent in entries if ent is not None]
    return new_entries, new_entry_ids


def _get_iter_id_from_enum_id(enum_id, num_structs_init, num_structs_add):
    """Calculate in which iteration this structure was enumerated.

    Iteration index starts from 0.
    """
    if enum_id < num_structs_init:
        return 0
    else:
        return (enum_id - num_structs_init) // num_structs_add + 1


def enumerate_structures(last_ce_document):
    """Enumerate new structures for DFT computation.

    Args:
        last_ce_document(CeOutputsDocument):
            The last cluster expansion outputs document.
    Returns:
        dict:
           Newly enumerated structures, super-cell matrices
           and feature vectors.

    """
    # Current iteration index. If starting from scratch,
    # last_iter_id will be -1.
    iter_id = last_ce_document.last_iter_id + 1
    # Must be pre-processed options.
    options = last_ce_document.ce_options

    enumerated_structures = deepcopy(last_ce_document.enumerated_structures)
    enumerated_features = deepcopy(last_ce_document.enumerated_features)

    subspace = last_ce_document.cluster_subspace
    sc_matrices = last_ce_document.supercell_matrices
    compositions = last_ce_document.compositions

    if sc_matrices is None or len(sc_matrices) == 0:
        raise ValueError(
            f"No super-cell matrix available!"
            f" Loaded document:\n {last_ce_document.dict()}"
        )
    if compositions is None or len(compositions) == 0:
        raise ValueError(
            f"No composition available!"
            f" Loaded document:\n {last_ce_document.dict()}"
        )

    # Historical coefs.
    coefs_history = deepcopy(last_ce_document.coefs_history)
    coefs_history = coefs_history or []
    if len(coefs_history) == 0:
        coefs = get_initial_ce_coefficients(subspace)
    else:
        coefs = np.array(coefs_history[-1])

    # Enumerate structures.
    log.info("Enumerating new structures.")
    new_structures, new_sc_matrices, new_features = _enumerate_structures(
        subspace,
        coefs,
        iter_id,
        sc_matrices,
        compositions,
        enumerated_structures,
        enumerated_features,
        options,
    )

    return {
        "new_structures": new_structures,
        "new_sc_matrices": new_sc_matrices,
        "new_features": new_features,
    }


# Separate job definition from function to enable easier testing and
# custom flow writing.
enumerate_structures_job = job(enumerate_structures)


def get_structure_calculation_flows(enum_output, last_ce_document):
    """Get workflows for newly enumerated structures.

    Args:
        enum_output(dict):
            Output by enumeration job.
        last_ce_document(CeOutputsDocument):
            The last cluster expansion outputs document.
    Returns:
        list[Flow], list[OutputReference]:
            Flows for each structure and their output references pointing
            at the final TaskDocument.
    """
    project_name = last_ce_document.project_name
    iter_id = last_ce_document.last_iter_id + 1
    if last_ce_document.enumerated_structures is None:
        struct_id = 0
    else:
        struct_id = len(last_ce_document.enumerated_structures)
    options = last_ce_document.ce_options
    relax_maker, static_maker, tight_maker = _get_vasp_makers(options)

    log.info("Performing ab-initio calculations.")
    new_structures = enum_output["new_structures"]
    flows = []
    for i, structure in enumerate(new_structures):
        fid = i + struct_id
        log.info(f"Calculating enumerated structure id: {fid}.")
        flow_name = project_name + f"_iter_{iter_id}_enum_{fid}"
        relax_maker.name = flow_name + "_relax"
        tight_maker.name = flow_name + "_tight"
        static_maker.name = flow_name + "_static"
        deformation = Deformation(options["apply_strain"])
        def_structure = deformation.apply_to_structure(structure)

        jobs = list()
        jobs.append(relax_maker.make(def_structure))
        if options["add_tight_relax"]:
            tight_job = tight_maker.make(
                jobs[-1].output.structure, prev_vasp_dir=jobs[-1].output.dir_name
            )
            # Allow failure in relaxation.
            tight_job.config.on_missing_references = OnMissing.NONE
            jobs.append(tight_job)
        static_job = static_maker.make(
            jobs[-1].output.structure, prev_vasp_dir=jobs[-1].output.dir_name
        )
        # Allow failure in relaxation.
        static_job.config.on_missing_references = OnMissing.NONE
        jobs.append(static_job)
        flows.append(Flow(jobs, output=jobs[-1].output, name=flow_name))
    outputs = [flow.output for flow in flows]

    return flows, outputs


@job
def calculate_structures_job(enum_output, last_ce_document):
    """Calculate newly enumerated structures.

    Note: it will replace itself with workflows to run for
    each structure.
    Args:
        enum_output(dict):
            Output by enumeration job.
        last_ce_document(CeOutputsDocument):
            The last cluster expansion outputs document.
    Returns:
        list[TaskDocument]:
            Results of VASP calculations as TaskDocument.
    """
    project_name = last_ce_document.project_name
    iter_id = last_ce_document.last_iter_id + 1

    flows, outputs = get_structure_calculation_flows(enum_output, last_ce_document)

    calc_flow = Flow(
        flows, output=outputs, name=project_name + f"_iter_{iter_id}" + "_calculations"
    )
    return Response(replace=calc_flow)


def parse_calculations(taskdocs, enum_output, last_ce_document):
    """Parse finished calculations into CeDataWrangler.

    Gives CeDataEntry with full decoration. Each computed structure
    will be re-decorated and re-inserted every iteration.
    Args:
        taskdocs(list[TaskDocument]):
            Task documents generated by vasp computations of
            added structures.
        enum_output(dict):
            Output by enumeration job.
        last_ce_document(CeOutputsDocument):
            The last cluster expansion outputs document.
    Returns:
        dict
            Updated wrangler, all entries before decoration,
            and all computed properties.
    """
    options = last_ce_document.ce_options
    prim_specs = last_ce_document.prim_specs

    undecorated_entries = deepcopy(last_ce_document.undecorated_entries)
    computed_properties = deepcopy(last_ce_document.computed_properties)
    enumerated_matrices = deepcopy(last_ce_document.enumerated_matrices)
    undecorated_entries = undecorated_entries or []
    computed_properties = computed_properties or []
    enumerated_matrices = enumerated_matrices or []
    log.info("Loading computations.")

    n_enum = len(enum_output["new_structures"])
    if len(taskdocs) != n_enum:
        raise ValueError(
            f"Number of TaskDocuments: {len(taskdocs)}"
            f" does not match with number of newly enumerated"
            f" structures: {n_enum}"
        )

    for doc in taskdocs:
        # Handles fizzled structures that will miss reference.
        if doc is None or doc.structure is None:
            undecorated_entries.append(None)
            computed_properties.append(None)
            continue

        flow_converged = _check_flow_convergence(doc)
        if flow_converged:
            undecorated_entry, properties = get_entry_from_taskdoc(
                doc, options["other_properties"], options["decorator_types"]
            )
            undecorated_entries.append(undecorated_entry)
            computed_properties.append(properties)
        else:
            # Calculations failed.
            undecorated_entries.append(None)
            computed_properties.append(None)

    decorators = _get_decorators(options, prim_specs["charge_decorated"])
    successful_entries = deepcopy(undecorated_entries)
    successful_entry_ids = list(range(len(undecorated_entries)))
    successful_entries, successful_entry_ids = _filter_out_failed_entries(
        successful_entries, successful_entry_ids
    )
    n_calc_finished = len(successful_entries)
    log.info(
        f"{n_calc_finished}/{len(undecorated_entries)}"
        f" structures successfully calculated."
    )

    log.info("Performing site decorations.")
    for dec, kw in zip(decorators, options["decorator_train_kwargs"]):
        dec.train(successful_entries, **kw)
        # Failed entries will be returned as None, and get filtered out.
        successful_entries = dec.decorate(successful_entries)
        successful_entries, successful_entry_ids = _filter_out_failed_entries(
            successful_entries, successful_entry_ids
        )

    successful_properties = [
        p for i, p in enumerate(computed_properties) if i in successful_entry_ids
    ]
    sc_matrices = enumerated_matrices + enum_output["new_sc_matrices"]
    successful_scmatrices = [
        m for i, m in enumerate(sc_matrices) if i in successful_entry_ids
    ]
    log.info(
        f"{len(successful_entries)}/{n_calc_finished}"
        f" structures successfully decorated."
    )

    # Wrangler must be cleared and reloaded each time
    # because decorator parameters can change.
    log.info("Loading data to wrangler.")
    wrangler = CeDataWrangler(last_ce_document.data_wrangler.cluster_subspace)
    for eid, prop, entry, mat in zip(
        successful_entry_ids,
        successful_properties,
        successful_entries,
        successful_scmatrices,
    ):
        # Save iteration index and the structure's index in
        # all enumerated structures.
        iter_id = _get_iter_id_from_enum_id(
            eid,
            options["num_structs_per_iter_init"],
            options["num_structs_per_iter_add"],
        )
        prop["spec"] = {"iter_id": iter_id, "enum_id": eid}
        wrangler.add_entry(entry, properties=prop, supercell_matrix=mat)
    log.info(
        f"{wrangler.num_structures}/{len(successful_entries)}"
        f" structures successfully mapped."
    )

    return {
        "wrangler": wrangler,
        "undecorated_entries": undecorated_entries,
        "computed_properties": computed_properties,
    }


parse_calculations_job = job(parse_calculations)


def fit_calculations(parse_output, last_ce_document):
    """Fit a new set of coefficients.

    Args:
        parse_output(dict):
            Output by parse job.
        last_ce_document(CeOutputsDocument):
            The last cluster expansion outputs document.
    Returns:
        dict:
           Dictionary containing fitted CE information.
    """
    options = last_ce_document.ce_options
    _, coefs, cv, cv_std, rmse, params = fit_ecis_from_wrangler(
        parse_output["wrangler"],
        options["estimator_type"],
        options["optimizer_type"],
        options["param_grid"],
        options["use_hierarchy"],
        options["center_point_external"],
        options["filter_unique_correlations"],
        estimator_kwargs=options["estimator_kwargs"],
        optimizer_kwargs=options["optimizer_kwargs"],
        **options["fit_kwargs"],
    )
    return {"coefs": coefs, "cv": cv, "cv_std": cv_std, "rmse": rmse, "params": params}


fit_calculations_job = job(fit_calculations)


def update_document(enum_output, parse_output, fit_output, last_ce_document):
    """Update the document to current iteration.

    Args:
        enum_output(dict):
            Output by enumeration job.
        fit_output(dict):
            Output by fit job.
        parse_output(dict):
            Output by parse job.
        last_ce_document(CeOutputsDocument):
            The last cluster expansion outputs document.
    Returns:
        CeOutputDocument:
            The updated document.
    """
    ce_document = deepcopy(last_ce_document)
    ce_document.data_wrangler = deepcopy(parse_output["wrangler"])
    ce_document.undecorated_entries = parse_output["undecorated_entries"]
    ce_document.computed_properties = parse_output["computed_properties"]
    if ce_document.coefs_history is None:
        ce_document.coefs_history = []
    ce_document.coefs_history.append(fit_output["coefs"])
    if ce_document.cv_history is None:
        ce_document.cv_history = []
    ce_document.cv_history.append(fit_output["cv"])
    if ce_document.cv_std_history is None:
        ce_document.cv_std_history = []
    ce_document.cv_std_history.append(fit_output["cv_std"])
    if ce_document.rmse_history is None:
        ce_document.rmse_history = []
    ce_document.rmse_history.append(fit_output["rmse"])
    if ce_document.params_history is None:
        ce_document.params_history = []
    ce_document.params_history.append(fit_output["params"])
    if ce_document.enumerated_structures is None:
        ce_document.enumerated_structures = []
    ce_document.enumerated_structures.extend(enum_output["new_structures"])
    if ce_document.enumerated_matrices is None:
        ce_document.enumerated_matrices = []
    ce_document.enumerated_matrices.extend(enum_output["new_sc_matrices"])
    # enumerated_features requires a list.
    if ce_document.enumerated_features is None:
        ce_document.enumerated_features = []
    ce_document.enumerated_features.extend(enum_output["new_features"])
    return ce_document


update_document_job = job(update_document)


def initialize_document(prim, project_name="ace-work", options=None):
    """Initialize an empty cluster expansion document.

    In this job, a cluster subspace will be created, super-cells
    and compositions will also be enumerated.
    Args:
        prim(structure):
            A primitive cell structure (no need to be reduced) with
            partial occupancy on some sub-lattice. This defines the
            lattice model of your cluster expansion.
        project_name(str): optional
            Name of the cluster expansion project. Since the underscore
            will be used to separate fields of job names, it should not
            appear in the project name!
        options(dict): optional
            A dictionary including all options to set up the automatic
            workflow.
            For available options, see docs in preprocessing.py.
    """
    # Pre-process options.
    options = options or {}
    log.info("Pre-processing primitive cell and workflow options.")
    options = _preprocess_options(options)

    # Reduce prim and get necessary specs.
    prim = reduce_prim(prim, **options["spacegroup_kwargs"])
    prim_specs = get_prim_specs(prim)
    bits = prim_specs["bits"]
    sublattice_sites = prim_specs["sublattice_sites"]
    sublattice_sizes = [len(sites) for sites in sublattice_sites]
    charge_decorated = prim_specs["charge_decorated"]
    nn_distance = prim_specs["nn_distance"]
    eq_constraints, leq_constraints, geq_constraints = parse_comp_constraints(
        options, bits, sublattice_sizes
    )

    # Get the cluster subspace. Other external terms than ewald not supported yet.
    # Cutoffs keys must be integers while pyyaml may load them as strings.
    cutoffs = {int(k): float(v) for k, v in options["cutoffs"].items()}
    subspace = get_cluster_subspace(
        prim,
        charge_decorated,
        nn_distance=nn_distance,
        cutoffs=cutoffs,
        use_ewald=options["use_ewald"],
        ewald_kwargs=options["ewald_kwargs"],
        **options["from_cutoffs_kwargs"],
    )

    # Enumerate supercell matrices, and remove aliased orbits from subspace.
    log.info("Enumerating super-cell matrices.")
    objective_sc_size = options["objective_num_sites"] // len(prim)
    sc_matrices = options["sc_matrices"] or enumerate_matrices(
        objective_sc_size,
        subspace,
        options["supercell_from_conventional"],
        options["max_sc_condition_number"],
        options["min_sc_angle"],
        **options["spacegroup_kwargs"],
    )

    # Not necessarily the same as the objective size.
    sc_size = subspace.num_prims_from_matrix(sc_matrices[0])
    # Supercells must be the same size
    if not np.allclose(
        [subspace.num_prims_from_matrix(m) for m in sc_matrices], sc_size
    ):
        raise ValueError(
            f"Provided super-cell matrices {sc_matrices} does"
            f" not have the same size! This is not allowed!"
        )
    subspace = truncate_cluster_subspace(subspace, sc_matrices)

    # Enumerate compositions as "counts" format in smol.moca.CompositionSpace.
    log.info("Enumerating valid compositions.")
    # Mute additional constraints if not needed.
    if len(eq_constraints) == 0:
        eq_constraints = None
    if len(leq_constraints) == 0:
        leq_constraints = None
    if len(geq_constraints) == 0:
        geq_constraints = None
    comp_space = CompositionSpace(
        bits,
        sublattice_sizes,
        charge_balanced=True,
        other_constraints=eq_constraints,
        geq_constraints=geq_constraints,
        leq_constraints=leq_constraints,
        optimize_basis=False,
        table_ergodic=False,
    )  # Not doing table flips.
    compositions = np.array(options["compositions"]).astype(
        int
    ) or enumerate_compositions_as_counts(
        sc_size,
        comp_space=comp_space,
        bits=bits,
        sublattice_sizes=sublattice_sizes,
        comp_enumeration_step=options["comp_enumeration_step"],
    )

    # Set up the initial document.
    init_ce_document = CeOutputsDocument(
        project_name=project_name,
        cluster_subspace=subspace,
        prim_specs=prim_specs,
        data_wrangler=CeDataWrangler(subspace),
        ce_options=options,
        supercell_matrices=sc_matrices,
        compositions=compositions,
    )

    return init_ce_document


initialize_document_job = job(initialize_document)
