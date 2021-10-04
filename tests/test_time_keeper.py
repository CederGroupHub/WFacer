from CEAuto.utils.frame_utils import load_dataframes, save_dataframes
from CEAuto.time_keeper import TimeKeeper
from CEAuto.wrappers import HistoryWrapper

from monty.serialization import loadfn
import pytest
import numpy as np
from pymatgen.core import Composition
import os

DATADIR = os.path.join(os.path.dirname(__file__), 'data')

@pytest.fixture
def history(subspace):
    coefs = np.random.random(subspace.num_corr_functions)
    coefs[0] = 1.0
    coefs = coefs.tolist()
    cv = 0.998
    rmse = 0.005
    return [{'coefs':coefs, 'cv':cv, 'rmse':rmse}]

def test_dataframes():
    sc_file = os.path.join(DATADIR,'sc_df_test.csv')
    comp_file = os.path.join(DATADIR,'comp_df_test.csv')
    fact_file = os.path.join(DATADIR,'fact_df_test.csv')
    sc_df, comp_df, fact_df = load_dataframes(sc_file=sc_file,
                                              comp_file=comp_file,
                                              fact_file=fact_file)
    assert isinstance(sc_df.matrix.iloc[0], list)
    assert isinstance(comp_df.ucoord.iloc[0],list)
    assert isinstance(comp_df.comp.iloc[0],list)
    assert isinstance(comp_df.comp.iloc[0][0],Composition)
    assert isinstance(fact_df.other_props.iloc[0], dict)
    assert isinstance(fact_df.ori_occu.iloc[0], list)
    assert isinstance(fact_df.ori_corr.iloc[0], list)

    save_dataframes(sc_df=sc_df, comp_df=comp_df, fact_df=fact_df)
    sc_df_rel, comp_df_rel, fact_df_rel = load_dataframes()
    
    assert sc_df_rel.equals(sc_df)
    assert comp_df_rel.equals(comp_df)
    assert fact_df_rel.equals(fact_df)

    os.remove('sc_mats.csv')
    os.remove('comps.csv')
    os.remove('data.csv')

@pytest.fixture
def timekeeper():
    return TimeKeeper(0)

def test_advance(timekeeper):
    for i in range(50):
        assert timekeeper._cursor == i
        assert timekeeper.iter_id == i // len(timekeeper.modules)
        assert timekeeper.next_module_todo == timekeeper.modules[i % len(timekeeper.modules)]
        for j in range(len(timekeeper.modules)):
            if j >= i % len(timekeeper.modules):
                assert timekeeper.todo(timekeeper.modules[j])
            else:
                assert timekeeper.done(timekeeper.modules[j])
        timekeeper.advance()

def test_setter(timekeeper):
    for i in range(100):
        timekeeper.cursor = i
        assert timekeeper.cursor == i

def test_set_from_data(timekeeper,history):
    sc_file = os.path.join(DATADIR,'sc_df_test.csv')
    comp_file = os.path.join(DATADIR,'comp_df_test.csv')
    fact_file = os.path.join(DATADIR,'fact_df_test.csv')
    sc_df, comp_df, fact_df = load_dataframes(sc_file=sc_file,
                                              comp_file=comp_file,
                                              fact_file=fact_file)
    timekeeper.set_to_data_status(sc_df, comp_df, fact_df, history)
    assert timekeeper.iter_id == 1
    assert timekeeper.next_module_todo == 'feat'
