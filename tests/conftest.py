import os
import pytest
import numpy as np
from monty.serialization import loadfn
from smol.cofe import ClusterSubspace
from smol.cofe.extern import EwaldTerm

from CEAuto import *

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
                                         cutoffs={2: 7, 3: 5, 4: 5},
                                         supercell_size='volume')
    space.add_external_term(EwaldTerm())
    return space

@pytest.fixture
def inputs_wrapper(structure):
    prim = structure
    options = {'decorators_types':['Magcharge'],
               'decorators_args':[{'labels_table':
                                  {'Li':[1],
                                   'Ca':[1],
                                   'Br':[-1]
                                  }}],
               'radius':{2:7.0, 3:5.0, 4:5.0},
               'extern_types': ['EwaldTerm']}
    # options are to be passed as kwargs, not dict.
    return InputsWrapper(prim=prim, **options)

@pytest.fixture
def history_wrapper(subspace):
    return HistoryWrapper(subspace)

@pytest.fixture
def history_wrapper_loaded(subspace):
    coefs = np.random.random(subspace.num_corr_functions +
                             len(subspace.external_terms))
    coefs[0] = 1.0
    coefs = coefs.tolist()
    cv = 0.998
    rmse = 0.005
    history = [{'coefs':coefs, 'cv':cv, 'rmse':rmse}]

    return HistoryWrapper(subspace, history)
