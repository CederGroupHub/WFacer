"""Test convergence functions."""
import numpy as np
import pytest

from pymatgen.entries.computed_entries import ComputedStructureEntry

from smol.cofe import ClusterExpansion
from smol.moca import Ensemble, CompositionSpace

from CEAuto.convergence import ce_converged
from CEAuto.wrangling import CeDataWrangler
from CEAuto.preprocessing import get_prim_specs

from .utils import gen_random_neutral_occupancy


def gen_random_occu_from_counts(ensemble, counts):
    n_species = 0
    occu = np.zeros(ensemble.num_sites, dtype=int) - 1
    for sublatt in ensemble.sublattices:
        n_sublatt = counts[n_species: n_species + len(sublatt.encoding)]
        occu_sublatt = [code for code, n in zip(sublatt.encoding, n_sublatt)
                        for _ in range(n)]
        np.random.shuffle(occu_sublatt)
        occu[sublatt.sites] = occu_sublatt
        n_species += len(sublatt.encoding)
    return occu


def get_one_wrangler(subspace, coefs, bad_wangler=False):
    structures = []
    specs = []
    energies = []

    ensemble = Ensemble \
        .from_cluster_expansion(ClusterExpansion(subspace, coefs),
                                np.eye(3) * 2)
    prim_specs = get_prim_specs(subspace.structure)
    sl_sizes = (len(sl) for sl in prim_specs["sublattice_sites"])

    comp_space = CompositionSpace(prim_specs["bits"],
                                  sl_sizes)
    xs = comp_space.get_composition_grid(8, step=2)
    counts = [comp_space.translate_format(x, 8,
                                          from_format="coordinates",
                                          to_format="counts")
              for x in xs]

    standard_energies = np.random.random(size=5) * 10 - 2
    n_enum = 0
    for iter_id in range(3):
        for count in counts:
            if not bad_wangler:
                rand_energies = (standard_energies +
                                 np.random.random(size=5) * 0.001)
            else:
                # Energy decrease greatly for each iteration.
                rand_energies = (standard_energies +
                                 np.random.random(size=5)
                                 - iter_id * 5)
            rand_occus = [gen_random_occu_from_counts(ensemble, count)
                          for _ in range(5)]
            rand_strs = [ensemble.processor.structure_from_occupancy(o)
                         for o in rand_occus]
            rand_specs = [{"iter_id": iter_id, "enum_id": n_enum + i}
                          for i in range(5)]
            structures.extend(rand_strs)
            specs.extend(rand_specs)
            energies.extend(rand_energies)

            n_enum += 5

    wrangler = CeDataWrangler(subspace)
    for s, e, spec in zip(structures, energies, specs):
        wrangler.add_entry(ComputedStructureEntry(s, e),
                           properties={"spec": spec},
                           supercell_matrix=np.eye(3) * 2,
                           check_struct_duplicacy=False
                           )  # Must supress dupe check here.

    return wrangler


def test_ce_converged(subspace):
    coefs1 = np.random.random(size=(subspace.num_corr_functions
                                    + len(subspace.external_terms))) - 10

    coefs2 = np.random.normal(size=(subspace.num_corr_functions
                                    + len)) + 10

    # wrangler including 3 iterations in total.
    converged_wrangler = get_one_wrangler(subspace, bad_wangler=False)
    bad_wrangler = get_one_wrangler(subspace, bad_wangler=True)

    # Not enough iterations
    assert not ce_converged([coefs1],
                            [1.0],
                            [0.1],
                            converged_wrangler,
                            {"cv_tol": 5,
                             "std_cv_rtol": None,
                             "delta_cv_rtol": 0.5,
                             "delta_eci_rtol": 0.1,
                             "delta_min_e_rtol": 2,
                             "continue_on_finding_new_gs": False,
                             "max_iter": 10
                             }
                            )
    # This should be good.
    assert ce_converged([coefs1, coefs1, coefs1],
                        [1.0, 1.0, 1.0],
                        [0.1, 0.1, 0.1],
                        converged_wrangler,
                        {"cv_tol": 5,
                         "std_cv_rtol": None,
                         "delta_cv_rtol": 0.5,
                         "delta_eci_rtol": 0.1,
                         "delta_min_e_rtol": 2,
                         "continue_on_finding_new_gs": False,
                         "max_iter": 10
                         }
                        )

    # Bad because ECIs are off.
    assert not ce_converged([coefs1, coefs1, coefs2],
                            [1.0, 1.0, 1.0],
                            [0.1, 0.1, 0.1],
                            converged_wrangler,
                            {"cv_tol": 5,
                             "std_cv_rtol": None,
                             "delta_cv_rtol": 0.5,
                             "delta_eci_rtol": 0.1,
                             "delta_min_e_rtol": 2,
                             "continue_on_finding_new_gs": False,
                             "max_iter": 10
                             }
                            )

    # Bad because wranglers minimum energis cannot match.
    assert not ce_converged([coefs1, coefs1, coefs2],
                            [1.0, 1.0, 1.0],
                            [0.1, 0.1, 0.1],
                            bad_wrangler,
                            {"cv_tol": 5,
                             "std_cv_rtol": None,
                             "delta_cv_rtol": 0.5,
                             "delta_eci_rtol": 0.1,
                             "delta_min_e_rtol": 2,
                             "continue_on_finding_new_gs": False,
                             "max_iter": 10
                             }
                            )

    # Bad because std_cv is too large.
    assert not ce_converged([coefs1, coefs1, coefs2],
                            [1.0, 1.0, 1.0],
                            [0.1, 0.1, 1.0],
                            converged_wrangler,
                            {"cv_tol": 5,
                             "std_cv_rtol": 0.1,
                             "delta_cv_rtol": 0.5,
                             "delta_eci_rtol": 0.1,
                             "delta_min_e_rtol": 2,
                             "continue_on_finding_new_gs": False,
                             "max_iter": 10
                             }
                            )

    # Good just because max number of iteration has been exceeded,
    # But you should be warned.
    assert not ce_converged([coefs1, coefs1, coefs2],
                            [1.0, 1.0, 1.0],
                            [0.1, 0.1, 1.0],
                            converged_wrangler,
                            {"cv_tol": 5,
                             "std_cv_rtol": 0.1,
                             "delta_cv_rtol": 0.5,
                             "delta_eci_rtol": 0.1,
                             "delta_min_e_rtol": 2,
                             "continue_on_finding_new_gs": False,
                             "max_iter": 2
                             }
                            )
