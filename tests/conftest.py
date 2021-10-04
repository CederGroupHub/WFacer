import os
import pytest
from monty.serialization import loadfn
from smol.cofe import ClusterSubspace
from smol.cofe.extern import EwaldTerm

# load test data files and set them up as fixtures
DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')

# some test structures to use in tests
files = ['LiCaBr_prim.json']

test_structures = [loadfn(os.path.join(DATA_DIR, file)) for file in files]

@pytest.fixture(params=test_structures, scope='module')
def structure(request):
    return request.param

@pytest.fixture(params=test_structures)
def subspace(request):
    space = ClusterSubspace.from_cutoffs(request.param,
                                         cutoffs={2: 4, 3: 3, 4: 3},
                                         supercell_size='volume')
    space.add_external_term(EwaldTerm())
    return space
