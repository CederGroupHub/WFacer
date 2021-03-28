from CEAuto import DataManager, InputsWrapper
from CEAuto.status_checker import StatusChecker
from CEAuto.utils.frame_utils import load_dataframes
from CEAuto.utils.occu_utils import structure_from_occu

import pytest
import numpy as np
import pandas as pd

import os
from copy import deepcopy
from pymatgen import Structure, Composition
from smol.cofe.space.domain import Vacancy

from monty.serialization import loadfn

DATADIR = os.path.join(os.path.dirname(__file__),'data')


@pytest.fixture
def structure():
    return loadfn(os.path.join(DATADIR,'LiCaBr_prim.json'))


@pytest.fixture
def inputs_wrapper(structure):
    lat_data = {'prim':structure}
    options = {'decorators_types':['MagChargeDecorator'],
               'decorators_args':[{'labels_table':
                                  {'Li':[1],
                                   'Ca':[1],
                                   'Br':[-1],
                                   'Cr':[0],
                                   'Fe':[0],
                                   'W':[0]}}],
               'radius':{2:4.0,3:3.0,4:3.0}}
    return InputsWrapper(lat_data, options=options)


@pytest.fixture
def subspace(inputs_wrapper):
    return inputs_wrapper.subspace


@pytest.fixture
def history(subspace):
    coefs = np.random.random(subspace.num_corr_functions +
                             len(subspace.external_terms))
    coefs[0] = 1.0
    coefs = coefs.tolist()
    cv = 0.998
    rmse = 0.005
    return [{'coefs':coefs, 'cv':cv, 'rmse':rmse}]


@pytest.fixture
def schecker(history):
    sc_file = os.path.join(DATADIR,'sc_df_test.csv')
    comp_file = os.path.join(DATADIR,'comp_df_test.csv')
    fact_file = os.path.join(DATADIR,'fact_df_test.csv')

    sc_df, comp_df, fact_df = load_dataframes(sc_file=sc_file,
                                              comp_file=comp_file,
                                              fact_file=fact_file)

    return StatusChecker(sc_df, comp_df, fact_df, history = history)


@pytest.fixture
def data_manager(inputs_wrapper, schecker):
    sock =  DataManager(inputs_wrapper.prim,
                        inputs_wrapper.bits,
                        inputs_wrapper.sublat_list,
                        inputs_wrapper.subspace,
                        schecker)
    sc_file = os.path.join(DATADIR,'sc_df_test.csv')
    comp_file = os.path.join(DATADIR,'comp_df_test.csv')
    fact_file = os.path.join(DATADIR,'fact_df_test.csv')

    sc_df, comp_df, fact_df = load_dataframes(sc_file=sc_file,
                                              comp_file=comp_file,
                                              fact_file=fact_file)

    sock._sc_df = sc_df
    sock._comp_df = comp_df
    sock._fact_df = fact_df

    return sock


def test_copy(data_manager):
    dm = data_manager.copy()
    dm.reset(flush_and_reload=True)
    assert dm.cur_iter_id == 0 
    assert len(dm.fact_df) == 0
    assert len(data_manager.fact_df) == 8

    CURDIR = os.getcwd()
    sc_file = os.path.join(CURDIR,'sc_mats.csv')
    comp_file = os.path.join(CURDIR,'comps.csv')
    data_file = os.path.join(CURDIR,'data.csv')

    os.remove(sc_file)
    os.remove(comp_file)
    os.remove(data_file)


def test_get_structures(data_manager):
    # print(data_manager.fact_df)
    df = data_manager.fact_df_with_structures
    for m,s,corr in zip(df.matrix,df.ori_str,df.ori_corr):
        assert np.allclose(data_manager._subspace.
                           corr_from_structure(s,scmatrix=m)[:-1], corr)

    for m,s,corr in zip(df.matrix,df.map_str,df.map_corr):
        assert ((pd.isna(s) and np.all(pd.isna(corr))) or
                (not pd.isna(s) and not np.all(pd.isna(corr))))


def test_find_sc_id(data_manager):
    assert data_manager.find_sc_id(data_manager.sc_df.matrix.iloc[0]) == 0

def test_insert_supercell(data_manager):
    dm = data_manager.copy()
    new_id = dm.insert_one_supercell([[-2,2,2],[1,-1,1],[1,1,-1]])
    assert new_id == 1
    assert len(dm.sc_df) == 2


def test_find_comp_id(data_manager):
    ucoord = [0.25, 0.75]
    comp = [Composition({'Li+':1/12, 'Ca+':1/4}), Composition({'Br-':1.0})]
    c_id = data_manager.find_comp_id(ucoord,sc_id=0)
    assert c_id == 0
    c_id = data_manager.find_comp_id(comp, sc_id=0, comp_format='composition')
    assert c_id == 0


def test_insert_comp(data_manager):
    dm = data_manager.copy()
    sc_id, new_id = dm.insert_one_comp([0.25, 0.75], sc_mat=[[-1,1,1],[1,-1,1],[1,1,-1]])
    assert (sc_id == 0) and (new_id == 0)
    sc_id, new_id = dm.insert_one_comp([0.125,0.875], sc_mat=[[-2,2,2],[1,-1,1],[1,1,-1]])
    assert (sc_id == 1) and (new_id == 2)
    assert len(dm.sc_df) == 2
    assert len(dm.comp_df) == 3


def test_find_entry_id(data_manager):
    occu_1 = [0, 2, 2, 1, 2, 2, 2, 2, 1, 1, 2, 2, 0, 0, 0, 0]
    occu_2 = [2, 0, 2, 2, 1, 2, 2, 2, 1, 1, 2, 2, 0, 0, 0, 0]
    eid = data_manager.find_entry_id_from_occu(occu_1,sc_id=0)
    assert eid == 4
    eid = data_manager.find_entry_id_from_occu(occu_2,sc_id=0)
    assert eid is None


def test_insert_occu(data_manager):
    occu_2 = [2, 0, 2, 2, 1, 2, 2, 2, 1, 1, 2, 2, 0, 0, 0, 0]
    dm = data_manager.copy()
    sc_id, comp_id, new_id = dm.insert_one_occu(occu_2, sc_id=0)
    assert comp_id == 0 and new_id == 8
    assert len(dm.fact_df) == 9

    occu_3 = [2, 1, 2, 2, 0, 2, 2, 2, 0, 0, 2, 2, 0, 0, 0, 0]
    sc_id, comp_id, new_id = dm.insert_one_occu(occu_3, sc_id=0)
    assert comp_id == 2 and new_id == 9
    assert len(dm.comp_df)==3 and len(dm.fact_df) == 10


def test_insert_structure(data_manager):
    occu_3 = [2, 1, 2, 2, 0, 2, 2, 2, 0, 0, 2, 2, 0, 0, 0, 0]   
    dm = data_manager.copy()
    sc_id, comp_id, new_id = dm.insert_one_occu(occu_3, sc_id=0)
    assert new_id == 8 and comp_id == 2


def test_remove_entree(data_manager):
    dm = data_manager.copy()
    dm.remove_entree_by_id(entry_ids = [0])
    assert np.allclose(dm.fact_df.entry_id, np.arange(7))
    assert np.allclose(dm.fact_df.comp_id, [0,1,1,0,0,1,1])


def test_remove_comps(data_manager):
    dm = data_manager.copy()
    dm.remove_comps_by_id(comp_ids = [0])
    assert np.allclose(dm.fact_df.entry_id, np.arange(4))
    assert np.allclose(dm.comp_df.comp_id, [0])
    assert np.allclose(dm.fact_df.comp_id, [0,0,0,0])


def test_remove_supercells(data_manager):
    dm = data_manager.copy()
    dm.remove_supercells_by_id(sc_ids= [0])
    assert len(dm.sc_df) == 0
    assert len(dm.comp_df) == 0
    assert len(dm.fact_df) == 0


def test_get_eid_status(data_manager):
    assert set(data_manager.get_eid_w_status('SC')) == {0,1,2,3}
    assert set(data_manager.get_eid_w_status('CL')) == {4,5,6,7}
