import os
import pytest
import numpy as np
from monty.serialization import loadfn
from smol.cofe import ClusterSubspace, ClusterExpansion
from smol.cofe.extern import EwaldTerm

from CEAuto.preprocessing import (reduce_prim,
                                  get_prim_specs,
                                  get_cluster_subspace)

# load test data files and set them up as fixtures
DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')

# some test structures to use in tests
files = ["LiCaBr_prim.json", "CrFeW_prim.json"]

test_structures = [loadfn(os.path.join(DATA_DIR, file)) for file in files]


@pytest.fixture(params=test_structures, scope='module')
def prim(request):
    return request.param


@pytest.fixture(scope="module")
def subspace(prim):
    prim = reduce_prim(prim)
    specs = get_prim_specs(prim)
    space = get_cluster_subspace(prim,
                                 specs["charge_decorated"],
                                 specs["nn_distance"],
                                 cutoffs={2: 7, 3: 5, 4: 5},
                                 use_ewald=True)
    return space


@pytest.fixture(scope="module")
def cluster_expansion(subspace):
    coefs_ = (np.random.
              random(subspace.num_corr_functions +
                     len(subspace.external_terms)))
    coefs_ = coefs_ - 0.5
    coefs_[0] = 1.0
    coefs_[-len(subspace.external_terms):] = 0.3
    return ClusterExpansion(subspace, coefs_)


@pytest.fixture(scope="module")
def data_wrangler(subspace):
    """A fictitious data wrangler."""
    n_entries_per_iter = 100
    n_iters = 10
    for iter_id in range(n_iters):

