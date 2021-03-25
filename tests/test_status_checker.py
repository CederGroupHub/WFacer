from CEAuto.utils.frame_utils import load_dataframes, save_dataframes
from CEAuto.status_checker import StatusChecker
from smol.cofe import ClusterSubspace

from monty.serialization import loadfn
import pytest
import numpy as np
from pymatgen import Composition
import os

DATADIR = os.path.join(os.path.dirname(__file__),'data')

@pytest.fixture
def structure():
    return loadfn(os.path.join(DATADIR,'LiCaBr_prim.json'))

@pytest.fixture
def subspace(structure):
    return ClusterSubspace.from_cutoffs(structure,
                                        cutoffs={2:4.0,
                                                 3:3.0,
                                                 4:3.0})

@pytest.fixture
def history(subspace):
    coefs = np.random.random(subspace.num_corr_functions)
    coefs[0] = 1.0
    coefs = coefs.tolist()
    cv = 0.998
    rmse = 0.005
    return [{'coefs':coefs, 'cv':cv, 'rmse':rmse}]

def test_load_dataframes():
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
    assert isinstance(fact_df.other_props.iloc[0],dict)

    save_dataframes(sc_df=sc_df, comp_df=comp_df, fact_df=fact_df)
    sc_df_rel, comp_df_rel, fact_df_rel = load_dataframes()
    
    assert sc_df_rel.equals(sc_df)
    assert comp_df_rel.equals(comp_df)
    assert fact_df_rel.equals(fact_df)

    os.remove('sc_mats.csv')
    os.remove('comps.csv')
    os.remove('data.csv')

@pytest.fixture
def schecker(history):
    sc_file = os.path.join(DATADIR,'sc_df_test.csv')
    comp_file = os.path.join(DATADIR,'comp_df_test.csv')
    fact_file = os.path.join(DATADIR,'fact_df_test.csv')
    sc_df, comp_df, fact_df = load_dataframes(sc_file=sc_file,
                                              comp_file=comp_file,
                                              fact_file=fact_file)
    return StatusChecker(sc_df, comp_df, fact_df, history=history)

def test_auto_load():
    assert isinstance(StatusChecker.auto_load(), StatusChecker)

def test_re_load():
    sc_file = os.path.join(DATADIR,'sc_df_test.csv')
    comp_file = os.path.join(DATADIR,'comp_df_test.csv')
    fact_file = os.path.join(DATADIR,'fact_df_test.csv')
    checker = StatusChecker.auto_load(sc_file=sc_file,
                                      comp_file=comp_file,
                                      fact_file=fact_file)

    old_sc_df = checker.sc_df.copy()
    old_comp_df = checker.comp_df.copy()
    old_fact_df = checker.fact_df.copy()
    
    checker.re_load()

    assert old_sc_df.equals(checker.sc_df)
    assert old_comp_df.equals(checker.comp_df)
    assert old_fact_df.equals(checker.fact_df)

def test_iter_id(schecker):
    assert schecker.cur_iter_id == 1

def test_completed_module(schecker):
    assert schecker.last_completed_module == 'calc'

def test_before_after(schecker):
    print(schecker.fact_df.loc[:,['entry_id','iter_id','module','calc_status']])
    assert schecker.before('feat')
    assert schecker.before('fit')
    assert schecker.after('calc')
    assert schecker.after('write')
    assert schecker.after('enum')
    assert schecker.after('gs')
