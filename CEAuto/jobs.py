"""Unitary jobs used by Maker."""
from copy import deepcopy
import numpy as np
from jobflow import Response, Flow, job
from itertools import product
import logging

from pymatgen.analysis.elasticity.strain import Deformation

from smol.cofe import ClusterExpansion
from smol.moca import CompositionSpace

from atomate2.vasp.jobs.core import (RelaxMaker,
                                     StaticMaker,
                                     TightRelaxMaker)
from atomate2.vasp.sets.core import (RelaxSetGenerator,
                                     StaticSetGenerator,
                                     TightRelaxSetGenerator)
from atomate2.vasp.schemas.calculation import Status

from .schema import CeOutputsDocument
from .wrangling import CeDataWrangler
from .preprocessing import (reduce_prim,
                            get_prim_specs,
                            parse_comp_constraints,
                            get_cluster_subspace,
                            get_initial_ce_coefficients,
                            process_supercell_options,
                            process_composition_options,
                            process_structure_options,
                            process_calculation_options,
                            process_decorator_options,
                            process_subspace_options,
                            process_fit_options,
                            process_convergence_options)
from .enumeration import (enumerate_matrices,
                          enumerate_compositions_as_counts,
                          truncate_cluster_subspace,
                          generate_initial_training_structures,
                          generate_additive_training_structures)
from .fit import fit_ecis_from_wrangler
from .utils.task_document import get_entry_from_taskdoc
from .specie_decorators import decorator_factory, PmgGuessChargeDecorator


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


def _enumerate_structures(subspace, coefs, iter_id,
                          sc_matrices, compositions,
                          enumerated_structures,
                          enumerated_features,
                          options):
    """Enumerate structures in an interation."""
    ce = ClusterExpansion(subspace, coefs)
    keep_ground_states = options["keep_ground_states"]
    num_structs_init = options["num_structs_per_iter_init"]
    num_structs_add = options["num_structs_per_iter_add"]
    mc_generator_kwargs = options["sample_generator_kwargs"]
    init_method = options["init_method"]
    add_method = options["add_method"]
    if iter_id == 0:
        new_structures, new_sc_matrices, new_features = \
            generate_initial_training_structures(ce,
                                                 product(sc_matrices,
                                                         compositions),
                                                 keep_ground_states,
                                                 num_structs_init,
                                                 mc_generator_kwargs,
                                                 method=init_method
                                                 )
    else:
        new_structures, new_sc_matrices, new_features = \
            generate_additive_training_structures(ce,
                                                  product(sc_matrices,
                                                          compositions),
                                                  enumerated_structures,
                                                  enumerated_features,
                                                  keep_ground_states,
                                                  num_structs_add,
                                                  mc_generator_kwargs,
                                                  method=add_method
                                                  )
    return new_structures, new_sc_matrices, new_features


def _get_vasp_makers(options):
    """Get required vasp makers."""
    relax_gen_kwargs = options["relax_generator_kwargs"]
    relax_generator = RelaxSetGenerator(**relax_gen_kwargs)
    relax_maker_kwargs = options["relax_maker_kwargs"]
    relax_maker = RelaxMaker(input_set_generator=relax_generator,
                             **relax_maker_kwargs)
    static_gen_kwargs = options["static_generator_kwargs"]
    static_generator = StaticSetGenerator(**static_gen_kwargs)
    static_maker_kwargs = options["static_maker_kwargs"]
    static_maker = StaticMaker(input_set_generator=static_generator,
                               **static_maker_kwargs)
    tight_gen_kwargs = options["tight_generator_kwargs"]
    tight_generator = TightRelaxSetGenerator(**tight_gen_kwargs)
    tight_maker_kwargs = options["tight_maker_kwargs"]
    tight_maker = TightRelaxMaker(input_set_generator=tight_generator,
                                  **tight_maker_kwargs)
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
    decorators = [decorator_factory(dt, **kw) for dt, kw
                  in zip(options["decorator_types"],
                         options["decorator_kwargs"])]
    decorated_properties = [d.decorated_prop_name for d in
                            decorators]
    if len(decorated_properties) != len(set(decorated_properties)):
        raise ValueError("Can not use multiple decorators for decorating"
                         " the same property!!")
    if is_charge_decorated and len(decorators) == 0:
        logging.warning("Cluster expansion is charge decorated, but"
                        " no charge decoration method is specified."
                        " Use default PmgGuessCharge at your risk!")
        decorators.append(PmgGuessChargeDecorator())
    return decorators


def _filter_out_failed_entries(entries, entry_ids):
    """Mark failed entries as none and return successful indices."""
    entry_ids = [eid for eid, ent in zip(entries, entry_ids)
                 if ent is not None]
    entries = [ent for ent in entries
               if ent is not None]
    return entries, entry_ids


@job
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
    iter_id = last_ce_document.last_iter_id + 1
    # Must be pre-processed options.
    options = last_ce_document.ce_options

    enumerated_structures = deepcopy(last_ce_document.enumerated_structures)
    enumerated_matrices = deepcopy(last_ce_document.enumerated_matrices)
    enumerated_features = deepcopy(last_ce_document.enumerated_features)

    subspace = last_ce_document.cluster_subspace
    sc_matrices = last_ce_document.supercell_matrices
    compositions = last_ce_document.compositions

    # Historical coefs.
    coefs_history = deepcopy(last_ce_document.coefs_history)
    if len(coefs_history) == 0:
        coefs = get_initial_ce_coefficients(subspace)
    else:
        coefs = np.array(coefs_history[-1])

    # Enumerate structures.
    logging.info("Enumerating new structures.")
    new_structures, new_sc_matrices, new_features = \
        _enumerate_structures(subspace, coefs, iter_id,
                              sc_matrices, compositions,
                              enumerated_structures,
                              enumerated_features,
                              options)

    return \
        {"new_structures": new_structures,
         "new_sc_matrices": new_sc_matrices,
         "new_features": new_features}


@job
def calculate_structures(enum_output, last_ce_document):
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
            Results of VASP calculations.
    """
    project_name = last_ce_document.project_name
    iter_id = last_ce_document.last_iter_id + 1
    struct_id = len(last_ce_document.enumerated_structures)
    options = last_ce_document.ce_options
    relax_maker, static_maker, tight_maker \
        = _get_vasp_makers(options)

    logging.info("Performing ab-initio calculations.")
    new_structures = enum_output["new_structures"]
    flows = []
    for i, structure in enumerate(new_structures):
        fid = i + struct_id
        logging.info(f"Calculating enumerated structure id: {fid}.")
        flow_name = project_name + f"_enum_{fid}"
        relax_maker.name = flow_name + "_relax"
        tight_maker.name = flow_name + "_tight"
        static_maker.name = flow_name + "_static"
        deformation = Deformation(options["apply_strain"])
        def_structure = deformation.apply_to_structure(structure)

        jobs = list()
        jobs.append(relax_maker.make(def_structure))
        if options["add_tight_relax"]:
            jobs.append(tight_maker.make(jobs[-1].output.structure,
                                         prev_vasp_dir=
                                         jobs[-1].output.dir_name))
        jobs.append(static_maker.make(jobs[-1].output.structure,
                                      prev_vasp_dir=
                                      jobs[-1].output.dir_name))
        flows.append(Flow(jobs, output=jobs[-1].output, name=flow_name))
    outputs = [flow.output for flow in flows]

    calc_flow = Flow(flows, outputs, 
                     name=project_name +
                     f"_iter_{iter_id}" +
                     "_calculation")
    return Response(replace=calc_flow)


@job
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
    iter_id = last_ce_document.last_iter_id + 1
    options = last_ce_document.ce_options
    prim_specs = last_ce_document.prim_specs

    undecorated_entries = deepcopy(last_ce_document.undecorated_entries)
    computed_properties = deepcopy(last_ce_document.computed_properties)
    logging.info("Loading computations.")

    for doc in taskdocs:
        flow_converged = _check_flow_convergence(doc)
        if flow_converged:
            undecorated_entry, properties \
                = get_entry_from_taskdoc(doc,
                                         options["other_properties"],
                                         options["decorator_types"])
            undecorated_entries.append(undecorated_entry)
            computed_properties.append(properties)
        else:
            # Calculations failed.
            undecorated_entries.append(None)
            computed_properties.append(None)

    decorators = _get_decorators(options,
                                 prim_specs["charge_decorated"])
    successful_entries = deepcopy(undecorated_entries)
    successful_entry_ids = list(range(len(undecorated_entries)))
    successful_entries, successful_entry_ids \
        = _filter_out_failed_entries(successful_entries,
                                     successful_entry_ids)
    n_calc_finished = len(successful_entries)
    logging.info(f"{n_calc_finished}/{len(undecorated_entries)}"
                 f" structures successfully calculated.")

    logging.info("Performing site decorations.")
    for dec, kw in zip(decorators, options["decorator_train_kwargs"]):
        dec.train(successful_entries, **kw)
        # Failed entries will be returned as None, and get filtered out.
        successful_entries = dec.decorate(successful_entries)
        successful_entries, successful_entry_ids \
            = _filter_out_failed_entries(successful_entries,
                                         successful_entry_ids)

    successful_properties = [p for i, p
                             in enumerate(computed_properties)
                             if i in successful_entry_ids]
    sc_matrices = (deepcopy(last_ce_document.enumerated_matrices)
                   + enum_output["new_sc_matrices"])
    successful_scmatrices = [m for i, m
                             in enumerate(sc_matrices)
                             if i in successful_entry_ids]
    logging.info(f"{len(successful_entries)}/{n_calc_finished}"
                 f" structures successfully decorated.")

    # Wrangler must be cleared and reloaded each time
    # because decorator parameters can change.
    logging.info("Loading data to wrangler.")
    wrangler = CeDataWrangler(last_ce_document.data_wrangler
                              .cluster_subspace)
    for eid, prop, entry, mat in zip(successful_entry_ids,
                                     successful_properties,
                                     successful_entries,
                                     successful_scmatrices):
        # Save iteration index and enumerated structure index.
        prop["spec"] = {"iter_id": iter_id, "enum_id": eid}
        wrangler.add_entry(entry,
                           properties=prop,
                           supercell_matrix=mat)
    logging.info(f"{wrangler.num_structures}/{len(successful_entries)}"
                 f" structures successfully mapped.")

    return \
        {"wrangler": wrangler,
         "undecorated_entries": undecorated_entries,
         "computed_properties": computed_properties
         }


@job
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
    coefs, cv, cv_std, params \
        = fit_ecis_from_wrangler(parse_output["wrangler"],
                                 options["estimator_type"],
                                 options["optimizer_type"],
                                 options["param_grid"],
                                 options["use_hierarchy"],
                                 estimator_kwargs=
                                 options["estimator_kwargs"],
                                 optimizer_kwargs=
                                 options["optimizer_kwargs"],
                                 **options["fit_kwargs"])
    return \
        {"coefs": coefs,
         "cv": cv,
         "cv_std": cv_std,
         "params": params
         }


@job
def update_document(enum_output,
                    parse_output,
                    fit_output,
                    last_ce_document):
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
    ce_document.data_wrangler = parse_output["wrangler"]
    ce_document.undecorated_entries \
        = parse_output["undecorated_entries"]
    ce_document.computed_properties \
        = parse_output["computed_properties"]
    ce_document.coefs_history.append(fit_output["coefs"])
    ce_document.cv_history.append(fit_output["cv"])
    ce_document.cv_std_history.append(fit_output["cv_std"])
    ce_document.params_history.append(fit_output["params"])
    ce_document.enumerated_structures \
        .extend(enum_output["new_structures"])
    ce_document.enumerated_matrices \
        .extend(enum_output["new_sc_matrices"])
    ce_document.enumerated_features \
        = np.append(ce_document.enumerated_features,
                    enum_output["new_features"])
    return ce_document


@job
def initialize_document(prim,
                        project_name="ceauto_work",
                        options=None):
    """Initialize an empty cluster expansion document.

    In this job, a cluster subspace will be created, super-cells
    and compositions will also be enumerated.
    Args:
        prim(structure):
            A primitive cell structure (no need to be reduced) with
            partial occupancy on some sub-lattice. This defines the
            lattice model of your cluster expansion.
        project_name(str):
            Name of the cluster expansion project.
        options(dict):
            A dictionary including all options to set up the automatic
            workflow.
            For available options, see docs in preprocessing.py.
    """
    # Pre-process options.
    options = options or {}
    logging.info("Pre-processing primitive cell and workflow options.")
    options = _preprocess_options(options)

    # Reduce prim and get necessary specs.
    prim = reduce_prim(prim, **options["spacegroup_kwargs"])
    prim_specs = get_prim_specs(prim)
    bits = prim_specs["bits"]
    sublattice_sites = prim_specs["sublattice_sites"]
    sublattice_sizes = [len(sites) for sites in sublattice_sites]
    charge_decorated = prim_specs["charge_decorated"]
    nn_distance = prim_specs["nn_distance"]
    eq_constraints, leq_constraints, geq_constraints \
        = parse_comp_constraints(options, bits, sublattice_sizes)

    # Get the cluster subspace. Other external terms than ewald not supported yet.
    subspace = get_cluster_subspace(prim, charge_decorated,
                                    nn_distance=nn_distance,
                                    cutoffs=options["cutoffs"],
                                    use_ewald=options["use_ewald"],
                                    ewald_kwargs=options["ewald_kwargs"],
                                    **options["from_cutoffs_kwargs"]
                                    )

    # Enumerate supercell matrices, and remove aliased orbits from subspace.
    logging.info("Enumerating super-cell matrices.")
    sc_matrices = (options["sc_matrices"] or
                   enumerate_matrices(options["objective_sc_size"],
                                      subspace,
                                      options["supercell_from_conventional"],
                                      options["spacegroup_kwargs"],
                                      options["max_sc_condition_number"],
                                      options["min_sc_angle"]
                                      )
                   )
    # Not necessarily the same as the objective size.
    sc_size = subspace.num_prims_from_matrix(sc_matrices[0])
    subspace = truncate_cluster_subspace(subspace, sc_matrices)

    # Enumerate compositions as "counts" format in smol.moca.CompositionSpace.
    logging.info("Enumerating valid compositions.")
    comp_space = CompositionSpace(bits, sublattice_sizes,
                                  charge_balanced=True,
                                  other_constraints=eq_constraints,
                                  geq_constraints=geq_constraints,
                                  leq_constraints=leq_constraints,
                                  optimize_basis=False,
                                  table_ergodic=False
                                  )  # Not doing table flips.
    compositions = (np.array(options["compositions"]).astype(int) or
                    enumerate_compositions_as_counts(sc_size,
                                                     comp_space=comp_space,
                                                     bits=bits,
                                                     sublattice_sizes=
                                                     sublattice_sizes,
                                                     comp_enumeration_step=
                                                     options["comp_enumeration_step"])
                    )

    # Set up the initial document.
    init_ce_document = CeOutputsDocument(project_name=project_name,
                                         cluster_subspace=subspace,
                                         prim_specs=prim_specs,
                                         ce_options=options,
                                         supercell_matrices=sc_matrices,
                                         compositions=compositions)

    return init_ce_document

