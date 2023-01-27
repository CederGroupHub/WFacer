import os
import pytest
import numpy as np
from monty.serialization import loadfn

from pymatgen.entries.computed_entries import ComputedStructureEntry

from smol.cofe import ClusterExpansion
from smol.moca import Ensemble

from CEAuto.preprocessing import (reduce_prim,
                                  get_prim_specs,
                                  get_cluster_subspace)
from CEAuto.wrangling import CeDataWrangler
from .utils import gen_random_neutral_occupancy

# load test data files and set them up as fixtures
DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')

# some test structures to use in tests
files = ["LiCaBr_prim.json", "CrFeW_prim.json"]

test_structures = [loadfn(os.path.join(DATA_DIR, file)) for file in files]


@pytest.fixture(params=test_structures, scope='package')
def prim(request):
    return request.param


@pytest.fixture(scope="package")
def subspace(prim):
    prim = reduce_prim(prim)
    specs = get_prim_specs(prim)
    space = get_cluster_subspace(prim,
                                 specs["charge_decorated"],
                                 specs["nn_distance"],
                                 cutoffs={2: 7, 3: 5, 4: 5},
                                 use_ewald=True)
    return space


@pytest.fixture(scope="package")
def specs(prim):
    return get_prim_specs(prim)


@pytest.fixture(scope="package")
def cluster_expansion(subspace):
    coefs_ = (np.random.
              random(subspace.num_corr_functions +
                     len(subspace.external_terms)))
    coefs_ = coefs_ - 0.5
    coefs_[0] = 1.0
    coefs_[-len(subspace.external_terms):] = 0.3
    return ClusterExpansion(subspace, coefs_)


@pytest.fixture(scope="package")
def ensemble(cluster_expansion):
    return Ensemble.from_cluster_expansion(cluster_expansion,
                                           [[3, 0, 0],
                                            [0, 3, 0],
                                            [0, 0, 3]])


@pytest.fixture(scope="package")
def data_wrangler(ensemble):
    """A fictitious data wrangler."""
    n_entries_per_iter = 100
    n_iters = 8
    n_enum = 0
    structures = []
    specs = []
    energies = []
    for iter_id in range(n_iters):
        for s_id in range(n_entries_per_iter):
            occu = gen_random_neutral_occupancy(sublattices=ensemble.sublattices)
            structures.append(ensemble.processor.structure_from_occupancy(occu))
            specs.append({"iter_id": iter_id, "enum_id": n_enum + s_id})
            energies.append(ensemble.natural_parameters @
                            ensemble.compute_feature_vector(occu))
        n_enum += n_entries_per_iter
    noise = np.random.normal(loc=0, scale=np.sqrt(np.var(energies) * 0.0001,
                             size=(len(energies),)))
    energies = np.array(energies) + noise
    entries = [ComputedStructureEntry(s, e)
               for s, e in zip(structures, energies)]
    wrangler = CeDataWrangler(ensemble.processor.cluster_subspace)
    for ent, spec in zip(entries, specs):
        wrangler.add_entry(ent,
                           properties={"spec": spec},
                           supercell_matrix
                           =ensemble.processor.supercell_matrix
                           )
    return wrangler
