"""Automatic jobflow maker."""
import numpy as np
from jobflow import Maker, Response
from itertools import product

from smol.cofe import ClusterExpansion
from smol.moca import CompositionSpace

from .preprocessing import (reduce_prim,
                            get_prim_specs,
                            parse_constraints,
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
from .convergence import ce_converged


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
        sc_options = process_supercell_options(self.options)
        comp_options = process_composition_options(self.options)
        struct_options = process_structure_options(self.options)
        calc_options = process_calculation_options(self.options)
        decor_options = process_decorator_options(self.options)
        space_options = process_subspace_options(self.options)
        fit_options = process_fit_options(self.options)
        conv_options = process_convergence_options(self.options)

        # Reduce prim and get necessary specs.
        prim = reduce_prim(prim, **sc_options["spacegroup_kwargs"])
        prim_specs = get_prim_specs(prim)
        bits = prim_specs["bits"]
        sublattice_sites = prim_specs["sublattice_sites"]
        sublattice_sizes = [len(sites) for sites in sublattice_sites]
        charge_decorated = prim_specs["charge_decorated"]
        nn_distance = prim_specs["nn_distance"]
        eq_constraints, leq_constraints, geq_constraints \
            = parse_constraints(comp_options, bits, sublattice_sizes)

        # Get the cluster subspace. Other external terms than ewald not supported yet.
        subspace = get_cluster_subspace(prim, charge_decorated,
                                        nn_distance=nn_distance,
                                        cutoffs=space_options["cutoffs"],
                                        use_ewald=space_options["use_ewald"],
                                        ewald_kwargs=space_options["ewald_kwargs"],
                                        **space_options["from_cutoffs_kwargs"]
                                        )

        # Enumerate supercell matrices, and remove aliased orbits from subspace.
        sc_matrices = (sc_options["sc_matrices"] or
                       enumerate_matrices(sc_options["objective_sc_size"],
                                          subspace,
                                          sc_options["supercell_from_conventional"],
                                          sc_options["spacegroup_kwargs"],
                                          sc_options["max_sc_condition_number"],
                                          sc_options["min_sc_angle"]
                                          )
                       )
        # Not necessarily the same as the objective size.
        sc_size = subspace.num_prims_from_matrix(sc_matrices[0])
        subspace = truncate_cluster_subspace(subspace, sc_matrices)
        coefs = get_initial_ce_coefficients(subspace)

        # Enumerate compositions as "counts" format in smol.moca.CompositionSpace.
        comp_space = CompositionSpace(bits, sublattice_sizes,
                                      charge_balanced=True,
                                      other_constraints=eq_constraints,
                                      geq_constraints=geq_constraints,
                                      leq_constraints=leq_constraints,
                                      optimize_basis=False,
                                      table_ergodic=False
                                      )
        compositions = (np.array(comp_options["compositions"]).astype(int) or
                        enumerate_counts(sc_size,
                                         comp_space=comp_space,
                                         bits=bits,
                                         sublattice_sizes=sublattice_sizes,
                                         comp_enumeration_step=
                                         comp_options["comp_enumeration_step"])
                        )

        iter_id = 0
        enumerated_structures = []
        enumerated_matrices = []
        enumerated_features = np.array([]).reshape(0, len(coefs))
        cv_history = []
        cv_std_history = []
        coefs_history = []  # A history record of fit coefficients.
        wrangler = CeDataWrangler(subspace)  # Empty wrangler.

        # TODO: define a function that is able to update charge assignment to all structures in wrangler.
        while not ce_converged(coefs_history,
                               cv_history,
                               cv_std_history,
                               wrangler,
                               iter_id,
                               conv_options):
            # Enumerate structures.
            coefs_history.append(coefs)
            ce = ClusterExpansion(subspace, coefs)
            keep_ground_states = struct_options["keep_ground_states"]
            num_structs_init = struct_options["num_structs_per_iter_init"]
            num_structs_add = struct_options["num_structs_per_iter_add"]
            mc_generator_kwargs = struct_options["sample_generator_kwargs"]
            init_method = struct_options["init_method"]
            add_method = struct_options["add_method"]
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
            enumerated_structures.extend(new_structures)
            enumerated_structures.extend(new_sc_matrices)
            enumerated_features = np.append(enumerated_features, new_features)

            # Create calculations for all structures.

            # Extract outputs

            # Decoration (clear up and re-insert wrangler entries each time.)

            # fit

            # Next iteration
            iter_id += 1


