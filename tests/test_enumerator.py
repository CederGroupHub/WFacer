from CEAuto import StructureEnumerator, DataManager
from CEAuto.time_keeper import TimeKeeper
from CEAuto.utils.frame_utils import load_dataframes
from CEAuto.utils.sc_utils import is_proper_sc

import pytest
import numpy as np

DATADIR = os.path.join(os.path.dirname(__file__),'data')

sc_file = os.path.join(DATADIR,'sc_df_test.csv')
comp_file = os.path.join(DATADIR,'comp_df_test.csv')
fact_file = os.path.join(DATADIR,'fact_df_test.csv')

def validate_data(data_manager, history_wrapper, iter_id=0, module='enum'):
    tk = TimeKeeper()
    tk.set_to_data_status(data_manager.sc_df,
                          data_manager.comp_df,
                          data_manager.fact_df,
                          history_wrapper.history)
    assert tk.iter_id == iter_id
    assert tk.next_module_todo == module

@pytest.fixture
def data_manager_empty(inputs_wrapper):

    return DataManager(inputs_wrapper)

@pytest.fixture
def enumerator_empty(data_manager_empty, history_wrapper):
    # Empty enumerator starts from iteration 0 enum, cursor=0.
    validate_data(data_manager_empty, history_wrapper, iter_id=0, module='enum')
    return StructureEnumerator(data_manager_empty, history_wrapper)

@pytest.fixture
def data_manager_loaded(inputs_wrapper):
    sc_df, comp_df, fact_df = load_dataframes(sc_file=sc_file,
                                              comp_file=comp_file,
                                              fact_file=fact_file)

    dm = DataManager(inputs_wrapper, sc_df, comp_df, fact_df)
    dm.remove_entree_by_iters_modules(iter_ids=[1], modules=['enum','gs'])
    assert len(dm.fact_df) == 4
    assert np.all(dm.fact_df.entry_id == 0)
    return dm

@pytest.fixture
def enumerator_loaded(data_manager_loaded, history_wrapper_loaded):
    # Start from iteration 1.
    validate_data(data_manager_loaded, history_wrapper_loaded, iter_id=1, module='enum')
    return StructureEnumerator(data_manager_loaded, history_wrapper_loaded)

def test_enumerate_scs(enumerator_empty):
    lat = enumerator_empty.prim.lattice
    # Test without transmat.
    mats = enumerator_empty.enumerate_sc_matrices()
    for m in mats:
        assert is_proper_sc(m, lat,
                            max_cond=enumerator_empty.max_sc_cond)
