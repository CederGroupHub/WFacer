"""Automatic jobflow maker."""
from copy import deepcopy
import numpy as np
from jobflow import Maker, Response, Flow
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
                          enumerate_counts,
                          truncate_cluster_subspace,
                          generate_initial_training_structures,
                          generate_additive_training_structures)
from .wrangling import CeDataWrangler
from .fit import fit_ecis_from_wrangler
from .convergence import ce_converged
from .utils.task_document import get_entry_from_taskdoc
from .specie_decorators import decorator_factory, PmgGuessChargeDecorator


class CeAutoMaker(Maker):
    """The cluster expansion automatic maker.

    Attributes:
        name(str):
            Name of the cluster expansion workflow.
        options(dict):
            A dictionary including all options to set up the automatic
            workflow.
            For available options, see docs in preprocessing.py.
    """
    name: str = "ceauto_work"
    options: dict = {}

    def _preprocess_options(self):
        sc_options = process_supercell_options(self.options)
        comp_options = process_composition_options(self.options)
        struct_options = process_structure_options(self.options)
        calc_options = process_calculation_options(self.options)
        decor_options = process_decorator_options(self.options)
        space_options = process_subspace_options(self.options)
        fit_options = process_fit_options(self.options)
        conv_options = process_convergence_options(self.options)

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

    @staticmethod
    def _enumerate_structures(subspace, coefs, iter_id,
                              sc_matrices, compositions,
                              enumerated_structures,
                              enumerated_features,
                              options):
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

    @staticmethod
    def _get_vasp_makers(options):
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

    @staticmethod
    def _check_flow_convergence(jobs):
        for j in jobs:
            try:
                status = j.output.calcs_reversed[0].has_vasp_completed
                if status == Status.FAILED:
                    return False
            except AttributeError:
                return False
        return True

    @staticmethod
    def _get_decorators(options, is_charge_decorated):
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

    @staticmethod
    def _filter_out_failed_entries(entries, entry_ids):
        entry_ids = [eid for eid, ent in zip(entries, entry_ids)
                     if ent is not None]
        entries = [ent for ent in entries
                   if ent is not None]
        return entries, entry_ids

    def make(self, prim):
        """Make the workflow.

        Args:
            prim(Structure):
                A primitive cell structure (no need to be reduced) with
                partial occupancy on some sub-lattice. This defines the
                lattice model of your cluster expansion.
        Returns:
            Response:
                The response of a recursive dynamic workflow.
        """
        # Pre-process options.
        logging.info("Pre-processing primitive cell and workflow options.")
        options = self._preprocess_options()

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
        coefs = get_initial_ce_coefficients(subspace)

        # Enumerate compositions as "counts" format in smol.moca.CompositionSpace.
        logging.info("Enumerating valid compositions.")
        comp_space = CompositionSpace(bits, sublattice_sizes,
                                      charge_balanced=True,
                                      other_constraints=eq_constraints,
                                      geq_constraints=geq_constraints,
                                      leq_constraints=leq_constraints,
                                      optimize_basis=False,
                                      table_ergodic=False
                                      )
        compositions = (np.array(options["compositions"]).astype(int) or
                        enumerate_counts(sc_size,
                                         comp_space=comp_space,
                                         bits=bits,
                                         sublattice_sizes=sublattice_sizes,
                                         comp_enumeration_step=
                                         options["comp_enumeration_step"])
                        )

        iter_id = 0
        struct_id = 0
        # Length equals to number of enumerated structures.
        enumerated_structures = list()
        enumerated_matrices = list()
        enumerated_features = np.array([]).reshape(0, len(coefs))
        undecorated_entries = list()
        computed_properties = list()

        # Historical records.
        cv_history = list()
        cv_std_history = list()
        coefs_history = list()
        params_history = list()
        wrangler = CeDataWrangler(subspace)  # Empty wrangler.

        # TODO: Cut these to smaller functions.
        # TODO: Do not write a "while"-loop. Write iteration as a job instead,
        #  otherwise flow can't be dynamic.
        logging.info("Begin iteration.")
        while not ce_converged(coefs_history,
                               cv_history,
                               cv_std_history,
                               wrangler,
                               iter_id,
                               options):
            # Enumerate structures.
            new_structures, new_sc_matrices, new_features =\
                self._enumerate_structures(subspace, coefs, iter_id,
                                           sc_matrices, compositions,
                                           enumerated_structures,
                                           enumerated_features,
                                           options)
            enumerated_structures.extend(new_structures)
            enumerated_matrices.extend(new_sc_matrices)
            enumerated_features = np.append(enumerated_features, new_features)

            # Create calculations for all structures, and extract outputs.
            relax_maker, static_maker, tight_maker \
                = self._get_vasp_makers(options)
            for i, structure in enumerate(new_structures):
                fid = i + struct_id
                flow_name = self.name + f"_enum_{fid}"
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
                flow = Flow(jobs, jobs[-1].output, name=flow_name)

                flow_converged = self._check_flow_convergence(jobs)
                if flow_converged:
                    undecorated_entry, properties \
                        = get_entry_from_taskdoc(flow.output,
                                                 options["other_properties"],
                                                 options["decorator_types"])
                    undecorated_entries.append(undecorated_entry)
                    computed_properties.append(properties)
                else:
                    # Calculations failed.
                    undecorated_entries.append(None)
                    computed_properties.append(None)
            # No need to save taskdoc into maker document because they will
            # always be saved as job output.

            # Decoration (clear up and re-insert wrangler entries each time.)
            decorators = self._get_decorators(options,
                                              prim_specs["charge_decorated"])
            successful_entries = deepcopy(undecorated_entries)
            successful_entry_ids = list(range(len(undecorated_entries)))
            successful_entries, successful_entry_ids \
                = self._filter_out_failed_entries(successful_entries,
                                                  successful_entry_ids)
            n_calc_finished = len(successful_entries)
            logging.info(f"{n_calc_finished}/{len(undecorated_entries)}"
                         f" structures finished calculation.")
            for dec, kw in zip(decorators, options["decorator_train_kwargs"]):
                dec.train(successful_entries, **kw)
                # Failed entries will be returned as None, and get filtered out.
                successful_entries = dec.decorate(successful_entries)
                successful_entries, successful_entry_ids \
                    = self._filter_out_failed_entries(successful_entries,
                                                      successful_entry_ids)
            successful_properties = [p for i, p
                                     in enumerate(computed_properties)
                                     if i in successful_entry_ids]
            successful_scmatrices = [m for i, m
                                     in enumerate(enumerated_matrices)
                                     if i in successful_entry_ids]
            # Wrangler must be cleared and reloaded each time
            # because decorator can change.
            wrangler.remove_all_data()
            for eid, prop, entry, mat in zip(successful_entry_ids,
                                             successful_properties,
                                             successful_entries,
                                             successful_scmatrices):
                # Save iteration index and enumerated structure index.
                prop["spec"] = {"iter_id": iter_id, "enum_id": eid}
                wrangler.add_entry(entry,
                                   properties=prop,
                                   supercell_matrix=mat)

            # fit
            coefs, cv, cv_std, params\
                = fit_ecis_from_wrangler(wrangler,
                                         options["estimator_type"],
                                         options["optimizer_type"],
                                         options["param_grid"],
                                         options["use_hierarchy"],
                                         estimator_kwargs=
                                         options["estimator_kwargs"],
                                         optimizer_kwargs=
                                         options["optimizer_kwargs"],
                                         **options["fit_kwargs"])
            cv_history.append(cv)
            cv_std_history.append(cv_std)
            coefs_history.append(coefs)
            params_history.append(params)

            # Next iteration
            iter_id += 1
            struct_id += len(new_structures)
