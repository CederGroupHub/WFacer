"""Test enumerating functions."""
from itertools import chain, product
from collections import defaultdict
import pytest
import numpy as np
import numpy.testing as npt
import logging

from pymatgen.core import Lattice, Element
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.analysis.structure_matcher import StructureMatcher

from smol.cofe import ClusterSubspace
from smol.moca import CompositionSpace, Ensemble
from smol.cofe.space.domain import Vacancy
from smol.moca.utils.occu import (get_dim_ids_by_sublattice,
                                  get_dim_ids_table,
                                  occu_to_counts)

from CEAuto.preprocessing import get_prim_specs
from CEAuto.enumeration import (enumerate_matrices,
                                truncate_cluster_subspace,
                                enumerate_compositions_as_counts,
                                get_num_structs_to_sample,
                                generate_initial_training_structures,
                                generate_additive_training_structures)


def test_enumerate_matrices(subspace):
    # 33 Will be rounded down to a multiple of conv_mat, which should be 32.
    mat_diagonal, mat_skew = enumerate_matrices(33, subspace)
    sa = SpacegroupAnalyzer(subspace.structure)
    t_inv = sa.get_conventional_to_primitive_transformation_matrix()
    conv_mat = np.round(np.linalg.inv(t_inv)).astype(int)
    npt.assert_array_almost_equal(2 * np.eye(3) @ conv_mat,
                                  mat_diagonal)

    def alias_level(sc):
        return len(list(chain(*subspace.get_aliasd_orbits(sc))))

    lat = subspace.structure.lattice

    def cond_and_angle(sc):
        new_mat = np.dot(sc, lat.matrix)
        new_lat = Lattice(new_mat)
        return (np.linalg.cond(sc),
                min([new_lat.alpha, new_lat.beta, new_lat.gamma,
                     180 - new_lat.alpha, 180 - new_lat.beta,
                     180 - new_lat.gamma]))

    assert alias_level(mat_diagonal) >= alias_level(mat_skew)

    cond, angle = cond_and_angle(mat_skew)
    assert not np.isclose(angle, 90)
    assert cond <= 8
    assert subspace.num_prims_from_matrix(mat_skew) == 32


def test_truncate_cluster_space(prim):
    a = prim.lattice.a
    cutoffs = {2: a * 3, 3: a * 2.5}

    sc = np.eye(3) * 2
    bad_subspace = ClusterSubspace.from_cutoffs(prim, cutoffs)

    assert len(bad_subspace.get_aliased_orbits(sc)) > 0

    good_subspace = truncate_cluster_subspace(bad_subspace,
                                              [sc])
    # No alias should remain, and some clusters must be kept.
    assert len(good_subspace.get_aliased_orbits(sc)) == 0
    assert len(good_subspace.orbits_by_size[2]) > 0
    assert len(good_subspace.orbits_by_size[3]) > 0


def test_enumerate_compositions(specs):
    bits = specs["bits"]
    bit_charges = [(0
                    if isinstance(sp, (Element, Vacancy))
                    else sp.oxi_state)
                   for sp in chain(*bits)]
    sl_sites = specs["sublattice_sites"]
    sl_sizes = [len(sites) for sites in sl_sites]
    comp_space = CompositionSpace(bits, sl_sites)

    with pytest.raises(ValueError):
        _ = enumerate_compositions_as_counts(32)

    counts = enumerate_compositions_as_counts(32,
                                              bits=bits,
                                              sublattice_sizes=sl_sizes,
                                              comp_enumeration_step=4)

    xs = [comp_space.translate_format(n, 32,
                                      from_format="counts",
                                      to_format="coordinates",
                                      rounding=True)
          for n in counts]

    xs_std = comp_space.get_composition_grid(supercell_size=32,
                                             step=4)
    npt.assert_array_almost_equal(xs, xs_std)

    bit_inds = get_dim_ids_by_sublattice(bits)
    for n in counts:
        assert np.isclose(np.dot(n, bit_charges), 0)
        for sl_ind, sl_bit_inds in enumerate(bit_inds):
            assert np.isclose(n[sl_bit_inds].sum(),
                              sl_sites[sl_ind] * 32)

    num_structs = get_num_structs_to_sample(counts, 1000)
    assert np.all(num_structs >= 2)
    assert np.abs(num_structs.sum() - 1000) / 1000 < 0.2


def test_enumerate_structures(cluster_expansion):
    sm = StructureMatcher()
    subspace = cluster_expansion.cluster_subspace
    supercells = enumerate_matrices(32, subspace)
    ensembles = [Ensemble.from_cluster_expansion(cluster_expansion,
                                                 m) for m in supercells]
    specs = get_prim_specs(subspace.structure)
    sl_sizes = [len(sl) for sl in specs["sublattice_sites"]]
    n_dims = sum(len(sl) for sl in specs["bits"])
    counts = enumerate_compositions_as_counts(32,
                                              bits=specs["bits"],
                                              sublattice_sizes=sl_sizes,
                                              comp_enumeration_step=4)
    sc_counts = list(product(supercells, counts))
    structures, matrices, feature_matrix \
        = generate_initial_training_structures(cluster_expansion,
                                               sc_counts,
                                               num_structs_init=300)
    assert len(structures) >= len(sc_counts)
    occus = [cluster_expansion.cluster_subspace.occupancy_from_structure(s, m)
             for s, m in zip(structures, matrices)]
    mids = [0 if np.allclose(m, supercells[0]) else 1 for m in matrices]
    ns = [tuple(occu_to_counts(occu, n_dims,
                               get_dim_ids_table(ensembles[mid]
                                                 .sublattices))
                .tolist()
                )
          for occu, mid in zip(occus, mids)]
    # When keeping the ground states, each composition must have at least 1 state.
    count_occurences = defaultdict(0)
    for n in ns:
        count_occurences[n] += 1

    for count in counts:
        assert count_occurences[tuple(count.tolist())] >= 1

    structures_add, matrices_add, feature_matrix_add \
        = generate_additive_training_structures(cluster_expansion,
                                                sc_counts,
                                                structures,
                                                feature_matrix,
                                                num_structs_add=200)
    # No structure should duplicate, and all must be charge balanced.
    all_structures = structures + structures_add
    for i in range(len(all_structures)):
        dupe = False
        assert np.isclose(all_structures[i].charge, 0)
        for j in range(i, len(all_structures)):
            if sm.fit(all_structures[i], all_structures[j]):
                dupe = True
        assert not dupe

    if len(structures) < 300:
        logging.warning(f"Number of structures in the initial pool is"
                        f" {len(structures)}, fewer than 300."
                        f" Your test case might be bad.")
    if len(structures_add) < 200:
        logging.warning(f"Number of structures in the added pool is"
                        f" {len(structures_add)}, fewer than 200."
                        f" Your test case might be bad.")
