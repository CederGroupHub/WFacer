"""This module implements a StructureEnumerator class for CE sampling.

Algorithm based on:

Ground state structures will also be added to the structure pool, but
they are not added here. They will be added in the convergence checker
module.
"""

__author__ = "Fengyu Xie"

from copy import deepcopy
from itertools import chain
from warnings import warn

import numpy as np
from joblib import Parallel, cpu_count, delayed
from pymatgen.core import Lattice
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from scipy.special import gammaln
from smol.moca import CompositionSpace

from .sample_generators import CanonicalSampleGenerator
from .utils.duplicacy import is_corr_duplicate, is_duplicate
from .utils.selection import select_added_rows, select_initial_rows
from .utils.supercells import get_three_factors, is_duplicate_sc


# TODO: in the future, may employ mcsqs type algos.
def enumerate_matrices(
    objective_sc_size,
    cluster_subspace,
    supercell_from_conventional=True,
    max_sc_cond=8,
    min_sc_angle=30,
    **kwargs,
):
    """Enumerate proper matrices with det size.

    Will give 1 unskewed matrix and 1 skewed matrix.
    Skewed matrix usually helps to avoid aliasing of clusters.

    Args:
        objective_sc_size(int):
            Objective supercell size in the number of primitive cells.
            Better be a multiple of det(conv_mat).
        cluster_subspace(smol.ClusterSubspace):
            The cluster subspace. cluster_subspace.structure must
            be pre-processed such that it is the true primitive cell
            in under its space group symmetry.
            Note: The cluster_subspace.structure must be reduced to a
            primitive cell!
        supercell_from_conventional(bool): optional
            Whether to enumerate supercell matrices in the form M@T, where
            M is an integer matrix, T is the primitive to conventional cell
            transformation matrix. Default to True.
        max_sc_cond(float):
            Maximum conditional number allowed of the skewed supercell
            matrices. By default set to 8, to prevent overstretching in one
            direction
        min_sc_angle(float):
            Minimum allowed angle of the supercell lattice. By default, set
            to 30, to prevent over-skewing.
        kwargs:
            keyword arguments to pass into SpaceGroupAnalyzer.
    Returns:
        List of 2D lists.
    """
    if not supercell_from_conventional:
        conv_mat = np.eye(3, dtype=int)
    else:
        prim = cluster_subspace.structure
        sa = SpacegroupAnalyzer(prim, **kwargs)
        t_inv = sa.get_conventional_to_primitive_transformation_matrix()
        conv_mat = np.round(np.linalg.inv(t_inv)).astype(int)

    conv_size = cluster_subspace.num_prims_from_matrix(conv_mat)
    if objective_sc_size % conv_size != 0:
        sc_size = objective_sc_size // conv_size * conv_size
        warn(
            f"Supercell size: {objective_sc_size} to enumerate "
            "is not divisible by primitive to conventional matrix"
            f" size {conv_size}."
            f" Will be rounded to {sc_size}!"
        )
    else:
        sc_size = objective_sc_size

    scs_diagonal = [
        np.diag(sorted(m, reverse=True))
        for m in get_three_factors(sc_size // conv_size)
    ]

    def get_skews(m, conv, space):
        # Get skews of a matrix. only upper-triangular used.
        skews = []
        ij_range = sorted({0, 1, m[0, 0] // 2, m[0, 0]})
        k_range = sorted({0, 1, m[1, 1] // 2, m[1, 1]})
        for i in ij_range:
            for j in ij_range:
                for k in k_range:
                    if i == 0 and j == 0 and k == 0:
                        continue
                    skewed = m.copy()
                    skewed[0, 1] = i
                    skewed[0, 2] = k
                    skewed[1, 2] = j
                    dupe = False
                    for m_old in skews:
                        if is_duplicate_sc(
                            m_old @ conv,
                            skewed @ conv,
                            space.structure,
                        ):
                            dupe = True
                            break
                    if not dupe:
                        skews.append(skewed)
        return skews

    scs_skew = list(
        chain(*[get_skews(sc, conv_mat, cluster_subspace) for sc in scs_diagonal])
    )

    # filter out bad matrices.
    lat = cluster_subspace.structure.lattice

    def cond_and_angle(sc):
        new_mat = np.dot(sc, lat.matrix)
        new_lat = Lattice(new_mat)
        return (
            np.linalg.cond(sc),
            min(
                [
                    new_lat.alpha,
                    new_lat.beta,
                    new_lat.gamma,
                    180 - new_lat.alpha,
                    180 - new_lat.beta,
                    180 - new_lat.gamma,
                ]
            ),
        )

    def filt_func_(sc):
        cond, angle = cond_and_angle(sc @ conv_mat)
        return cond <= max_sc_cond and angle >= min_sc_angle

    scs_diagonal = list(filter(filt_func_, scs_diagonal))
    scs_skew = list(filter(filt_func_, scs_skew))

    def alias_level(sc):
        return len(list(chain(*cluster_subspace.get_aliased_orbits(sc))))

    # Sort diagonal by low stretch, then low alias level.
    def diagonal_sort_key(sc):
        cond, angle = cond_and_angle(sc @ conv_mat)
        return cond, -angle, alias_level(sc @ conv_mat)

    # Sort diagonal by low stretch, then low alias level.
    def skew_sort_key(sc):
        cond, angle = cond_and_angle(sc @ conv_mat)
        return alias_level(sc @ conv_mat), -angle, cond

    scs_diagonal = sorted(scs_diagonal, key=diagonal_sort_key)
    scs_skew = sorted(scs_skew, key=skew_sort_key)

    # Select 1 diagonal, 1 off diagonal.
    # Must return lists for pydantic validation.
    return [
        np.round(scs_diagonal[0] @ conv_mat).astype(int).tolist(),
        np.round(scs_skew[0] @ conv_mat).astype(int).tolist(),
    ]


def truncate_cluster_subspace(cluster_subspace, sc_matrices):
    """Given supercell matrices, remove aliased orbits.

    Args:
        cluster_subspace(ClusterSubspace):
            Cluster subspace with aliased orbits.
        sc_matrices(3*3 ArrayLike):
            Enumerated super-cell matrices.

    Returns:
        ClusterSubspace: truncated subspace without aliased orbits.
    """
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
    if len(to_remove) > 0:
        warn(
            f"Orbit aliasing could not be avoided "
            f"with given supercells: {sc_matrices}!\n"
            f"Removed orbits with indices: {to_remove}"
        )
    cluster_subspace_new = cluster_subspace.copy()
    # Cannot call remove_orbit with an empty list.
    if len(to_remove) > 0:
        cluster_subspace_new.remove_orbits(to_remove)
    return cluster_subspace_new


def enumerate_compositions_as_counts(
    sc_size,
    comp_space=None,
    bits=None,
    sublattice_sizes=None,
    comp_enumeration_step=1,
    **kwargs,
):
    """Enumerate compositions in a given supercell size.

    Results will be returned in "counts" format
    (see smol.moca.CompositionSpace).
    Args:
        sc_size(int):
            The super-cell size in the number of prim cells.
        comp_space(CompositionSpace): optional
            Composition space in a primitive cell. If not given,
            arguments "bits" and "sublattice_sizes" must be given.
        bits(List[List[Species|DummySpecies|Element|Vacancy]]):
            Allowed species on each sub-lattice.
        sublattice_sizes(List[int]):
            Number of sites in each sub-lattice in a prim cell.
        comp_enumeration_step(int):
            Step in returning the enumerated compositions.
            If step = N > 1, on each dimension of the composition space,
            we will only yield one composition every N compositions.
            Default to 1.
        kwargs:
            Other keyword arguments to initialize CompositionSpace.
    Returns:
        Enumerated possible compositions in "counts" format, not normalized:
            2D np.ndarray[int]
    """
    if comp_space is None:
        if bits is None or sublattice_sizes is None:
            raise ValueError(
                "Must provide either comp_space or" " bits and sublattice_sizes!"
            )
        comp_space = CompositionSpace(bits, sublattice_sizes, **kwargs)
        # This object can be saved in process to avoid additional
        # enumeration cost.
    xs = comp_space.get_composition_grid(
        supercell_size=sc_size, step=comp_enumeration_step
    )
    ns = [
        comp_space.translate_format(
            x, sc_size, from_format="coordinates", to_format="counts", rounding=True
        )
        for x in xs
    ]
    return (
        np.array(ns).astype(int).tolist()
    )  # Must return a list to correctly validate.


def get_num_structs_to_sample(
    all_counts, num_structs_select, scale=3, min_num_per_composition=2
):
    """Get number of structures to sample in each McSampleGenerator.

    Args:
        all_counts(ArrayLike):
            All enumerated compositions in "counts" format.
        num_structs_select(int):
            Number of structures to eventually select.
        scale(int): optional
            How many structures to sample in total, divided by
            num_structs_select. Default is 3.
        min_num_per_composition(int): optional
            Minimum number of structures to sample per composition.
            Default to 2.
    """

    def get_ln_weight(counts):
        # Get number of configurations with the composition.
        return gammaln(np.sum(counts) + 1) - np.sum([gammaln(n + 1) for n in counts])

    # In total sample 3 * structures than finally to be selected.
    num_structs_total = num_structs_select * scale
    min_n = min_num_per_composition

    ln_weights = [get_ln_weight(counts) for counts in all_counts]
    weights = np.exp(ln_weights)
    num_structs = weights / np.sum(weights) * num_structs_total
    deficit = (num_structs < min_n).sum() * min_n - num_structs[
        num_structs < min_n
    ].sum()
    overflow = num_structs[num_structs > min_n] - min_n
    deltas = deficit * overflow / overflow.sum()
    num_structs[num_structs < min_n] = min_n
    num_structs[num_structs > min_n] -= deltas
    if np.any(num_structs < min_n):
        warn(
            "Too many compositions enumerated compared to "
            "the number of structures to enumerate. "
            "You may increase comp_enumeration_step, "
            "or increase num_structs_init. Force set "
            f"all supercell and compositions to generate {min_n} "
            "sample structures."
        )
        num_structs[:] = min_n
    else:
        num_structs = np.round(num_structs).astype(int)

    return num_structs


def _sample_single_generator(
    ce,
    all_previous_structs,
    all_previous_features,
    mc_generator_args,
    sc_matrix,
    counts,
    num_sample,
    duplicacy_criteria="correlations",
):
    """Create and sample a single generator.

    Used for parallel sampling multiple compositions.
    """
    generator = CanonicalSampleGenerator(
        ce,
        sc_matrix,
        counts,
        duplicacy_criteria=duplicacy_criteria,
        **mc_generator_args,
    )

    gs_occu = generator.get_ground_state_occupancy()
    gs_feat = generator.get_ground_state_features()
    gs_struct = generator.get_ground_state_structure()
    samples, samples_occu, samples_feat = generator.get_unfrozen_sample(
        previous_sampled_structures=[gs_struct] + all_previous_structs,
        previous_sampled_features=[gs_feat] + all_previous_features,
        num_samples=num_sample,
    )

    gs_dupe = False
    for old_struct, old_feat in zip(all_previous_structs, all_previous_features):
        if duplicacy_criteria == "correlations":
            gs_dupe = is_corr_duplicate(
                gs_struct, generator.processor, features2=old_feat
            )
        elif duplicacy_criteria == "structure":
            gs_dupe = is_duplicate(
                gs_struct, old_struct, remove_decorations=generator.remove_decorations
            )
        else:
            raise ValueError(f"{duplicacy_criteria} comparison not" f" supported!")
        # Must remove all decorations to avoid adding in exactly the same input.
        if gs_dupe:
            break

    return gs_struct, gs_occu, gs_feat, samples, samples_occu, samples_feat, gs_dupe


# Currently, only supporting canonical sample generator.
def generate_training_structures(
    ce,
    enumerated_matrices,
    enumerated_counts,
    previous_sampled_structures=None,
    previous_feature_matrix=None,
    keep_ground_states=True,
    num_structs=60,
    mc_generator_kwargs=None,
    n_parallel=None,
    duplicacy_criteria="correlations",
    **kwargs,
):
    """Generate training structures at the first iteration.

    Args:
        ce(ClusterExpansion):
            ClusterExpansion object initialized as null. If charge decorated,
            will contain an ewald contribution at 100%
        enumerated_matrices(list[3*3 ArrayLike[int]]):
            Previously enumerated supercell matrices. Must be the same super-cell
            size.
        enumerated_counts(list[1D ArrayLike]):
            Previously enumerated compositions in "counts" format. Must fit in
            the super-cell size.
            Note: Different super-cell sizes not supported!
        previous_sampled_structures(list[Structure]): optional
            Sample structures already calculated in past iterations.
            If given, that means you will add structures to an existing
            training set.
        previous_feature_matrix(list[list[[float]]): optional
            Correlation vectors of structures already calculated in past iterations.
        keep_ground_states(bool): optional
            Whether always to include the electrostatic ground states.
            Default to True.
        num_structs(int): optional
            Number of training structures to add at the iteration.
            At least 2~3 structures should be enumerated for each composition.
            And it is recommended that num_structs_init * 10 > 2 *
            len(supercell_and_counts).
            Default is 60.
        mc_generator_kwargs(dict): optional
            Keyword arguments for McSampleGenerator, except num_samples.
            Note: currently only support Canonical.
        n_parallel(int): optional
            Number of generators to run in parallel. Default is to use
            a quarter of cpu count.
        duplicacy_criteria(str):
            The criteria when to consider two structures as the same and
            old to add one of them into the candidate training set.
            Default is "correlations", which means to assert duplication
            if two structures have the same correlation vectors. While
            "structure" means two structures must be symmetrically equivalent
            after being reduced. No other option is allowed.
            Note that option "structure" might be significantly slower since
            it has to attempt reducing every structure to its primitive cell
            before matching. It should be used with caution.
        kwargs:
            Keyword arguments for utils.selection.select_initial_rows.
    Returns:
        list[Structure], list[3*3 list[list[int]]], list[list[float]]:
            Initial training structures, super-cell matrices,
            and normalized correlation vectors.
    """
    mc_generator_args = mc_generator_kwargs or {}
    n_parallel = n_parallel or min(cpu_count() // 4, len(enumerated_counts))
    if n_parallel == 0:
        if cpu_count() // 4 == 0:
            warn(
                f"Number of CPUs found on the executing environment: {cpu_count()} might"
                f" not be enough for parallelization! Setting parallel processes to 1."
            )
            n_parallel = 1

    previous_sampled_structures = previous_sampled_structures or []
    previous_feature_matrix = np.array(previous_feature_matrix).tolist() or []
    if len(previous_feature_matrix) != len(previous_sampled_structures):
        raise ValueError(
            "Must provide a feature vector for each" " structure passed in!"
        )

    # Scale the number of structures to select for each comp.
    num_samples = get_num_structs_to_sample(
        [counts for _ in enumerated_matrices for counts in enumerated_counts],
        num_structs,
    )

    with Parallel(n_jobs=n_parallel) as par:
        gs_id = 0
        keeps = []
        structures = []
        femat = []
        sc_matrices = []
        sc_matrix_indices = []

        for mid, sc_matrix in enumerate(enumerated_matrices):
            # This should work on pytest.
            results = par(
                delayed(_sample_single_generator)(
                    ce,
                    previous_sampled_structures + structures,
                    previous_feature_matrix + femat,
                    mc_generator_args,
                    sc_matrix,
                    counts,
                    num_sample,
                    duplicacy_criteria=duplicacy_criteria,
                )
                for counts, num_sample in zip(
                    enumerated_counts,
                    num_samples[
                        mid
                        * len(enumerated_counts) : (mid + 1)
                        * len(enumerated_counts)
                    ],
                )
            )

            for (
                gs_struct,
                gs_occu,
                gs_feat,
                samples,
                samples_occu,
                samples_feat,
                gs_dupe,
            ) in results:
                if gs_dupe:
                    structures.extend(samples)
                    femat.extend(samples_feat)
                    sc_matrices.extend([sc_matrix for _ in samples])
                    sc_matrix_indices.extend([mid for _ in samples])
                    gs_id += len(samples)
                else:
                    structures.extend([gs_struct] + samples)
                    femat.extend([gs_feat] + samples_feat)
                    sc_matrices.extend([sc_matrix for _ in range(len(samples) + 1)])
                    sc_matrix_indices.extend([mid for _ in range(len(samples) + 1)])
                    if keep_ground_states:
                        keeps.append(gs_id)
                    gs_id += len(samples) + 1

    femat = np.array(femat)

    # External terms such as the ewald term should not be taken into comparison,
    # when selecting structures
    num_external_terms = len(ce.cluster_subspace.external_terms)

    if len(previous_sampled_structures) == 0:
        # Start from scratch.
        selected_row_ids = select_initial_rows(
            femat,
            n_select=num_structs,
            keep_indices=keeps,
            num_external_terms=num_external_terms,
            **kwargs,
        )
    else:
        # Add to existing:
        selected_row_ids = select_added_rows(
            femat,
            np.array(previous_feature_matrix),
            n_select=num_structs,
            keep_indices=keeps,
            num_external_terms=num_external_terms,
            **kwargs,
        )

    # Must sort to ensure the same ordering between feature rows and structures.
    selected_row_ids = sorted(selected_row_ids)
    selected_structures = [s for i, s in enumerate(structures) if i in selected_row_ids]
    selected_matrices = [m for i, m in enumerate(sc_matrices) if i in selected_row_ids]
    selected_femat = femat[selected_row_ids, :].tolist()
    if len(selected_row_ids) < num_structs:
        warn(
            f"Expected to add {num_structs} new structures,"
            f" but only {len(selected_row_ids)}"
            f" non duplicate structures could be added."
        )
    return selected_structures, selected_matrices, selected_femat
