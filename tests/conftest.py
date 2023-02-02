import os
import pytest
import numpy as np
from monty.serialization import loadfn

from smol.cofe import ClusterExpansion
from smol.moca import Ensemble

from CEAuto.preprocessing import (reduce_prim,
                                  get_prim_specs,
                                  get_cluster_subspace)
from .utils import gen_random_wrangler

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
def subspace_sin(prim):
    prim = reduce_prim(prim)
    specs = get_prim_specs(prim)
    space = get_cluster_subspace(prim,
                                 specs["charge_decorated"],
                                 specs["nn_distance"],
                                 basis_type="sinusoid",
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
def cluster_expansion_sin(subspace_sin):
    coefs_ = (np.random.
              random(subspace_sin.num_corr_functions +
                     len(subspace_sin.external_terms)))
    coefs_ = coefs_ - 0.5
    coefs_[0] = 1.0
    coefs_[-len(subspace_sin.external_terms):] = 0.3
    return ClusterExpansion(subspace_sin, coefs_)


@pytest.fixture(scope="package")
def ensemble(cluster_expansion):
    return Ensemble.from_cluster_expansion(cluster_expansion,
                                           [[3, 0, 0],
                                            [0, 3, 0],
                                            [0, 0, 3]])


@pytest.fixture(scope="package")
def ensemble_sin(cluster_expansion_sin):
    return Ensemble.from_cluster_expansion(cluster_expansion_sin,
                                           [[3, 0, 0],
                                            [0, 3, 0],
                                            [0, 0, 3]])


@pytest.fixture(scope="package")
def data_wrangler(ensemble):
    """A fictitious data wrangler."""
    return gen_random_wrangler(ensemble)


@pytest.fixture(scope="package")
def data_wrangler_sin(ensemble_sin):
    """A fictitious data wrangler, with sinusoid basis."""
    return gen_random_wrangler(ensemble_sin)
