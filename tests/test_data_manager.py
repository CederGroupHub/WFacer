from CEAuto import DataManager, InputsWrapper
from CEAuto.utils.frame_utils import load_dataframes, save_dataframes
from CEAuto.utils.occu_utils import structure_from_occu, occu_from_structure
from CEAuto.config_paths import *
from CEAuto.utils.math_utils import get_diag_matrices

import pytest
import numpy as np
import pandas as pd
from pandas._testing import assert_frame_equal
import random

import os
from copy import deepcopy
from pymatgen.core import Structure, Composition
from pymatgen.analysis.structure_matcher import StructureMatcher
from smol.cofe.space.domain import Vacancy

from monty.serialization import loadfn

DATADIR = os.path.join(os.path.dirname(__file__),'data')

sc_file = os.path.join(DATADIR,'sc_df_test.csv')
comp_file = os.path.join(DATADIR,'comp_df_test.csv')
fact_file = os.path.join(DATADIR,'fact_df_test.csv')


def assert_df_formats(sc_df, comp_df, fact_df):
    # Assert the correct formats.
    def assert_2d(ll, shape=None, name=(float, int)):
        if shape is not None:
            assert len(ll) == shape[0]
        for l in ll:
            assert isinstance(l, list)
            if shape is not None:
                assert len(l) == shape[1]
            for i in l:
                assert isinstance(i, name)

    def assert_1d(l, name=(float, int)):
        for i in l:
            assert isinstance(i, name)

    def assert_str(l):
        for i in l:
            assert isinstance(i, str)

    def assert_type_series(series, name, more_assertion=None, allow_none=False):
        for item in series:
            if not np.any(pd.isna(item)):
                assert isinstance(item, name)
            else:
                assert allow_none

            if more_assertion is not None and not np.any(pd.isna(item)):
                more_assertion(item)

    assert_type_series(sc_df.matrix, list, lambda x: assert_2d(x, shape=(3, 3), name=int))
    assert_type_series(comp_df.ucoord, list, assert_1d)
    assert_type_series(comp_df.ccoord, list, assert_1d)
    assert_type_series(comp_df.comp, list, lambda x: assert_1d(x, name=Composition))
    assert_type_series(comp_df.cstat, list, assert_2d)
    assert_type_series(comp_df.nondisc, list, assert_1d)
    assert_type_series(fact_df.ori_occu, list, lambda x: assert_1d(x, name=int))
    assert_type_series(fact_df.map_occu, list, lambda x: assert_1d(x, name=int), allow_none=True)
    assert_type_series(fact_df.ori_corr, list, assert_1d)
    assert_type_series(fact_df.map_corr, list, assert_1d, allow_none=True)
    assert_type_series(fact_df.other_props, dict, assert_str, allow_none=True)


def test_load_save_dfs():

    sc_df, comp_df, fact_df = load_dataframes(sc_file=sc_file,
                                              comp_file=comp_file,
                                              fact_file=fact_file)
    assert_df_formats(sc_df, comp_df, fact_df)

    save_dataframes(sc_df, comp_df, fact_df,
                    sc_file=SC_FILE,
                    comp_file=COMP_FILE,
                    fact_file=FACT_FILE)

    assert os.path.isfile(SC_FILE)
    assert os.path.isfile(COMP_FILE)
    assert os.path.isfile(FACT_FILE)
    assert os.path.getsize(SC_FILE) > 0
    assert os.path.getsize(COMP_FILE) > 0
    assert os.path.getsize(FACT_FILE) > 0

    sc_df2, comp_df2, fact_df2 = load_dataframes(sc_file=SC_FILE,
                                                 comp_file=COMP_FILE,
                                                 fact_file=FACT_FILE)   

    assert_df_formats(sc_df2, comp_df2, fact_df2)
    assert_frame_equal(sc_df, sc_df2)
    assert_frame_equal(comp_df, comp_df2)
    assert_frame_equal(fact_df, fact_df2)

    os.remove(SC_FILE)
    os.remove(COMP_FILE)
    os.remove(FACT_FILE)


@pytest.fixture
def data_manager(inputs_wrapper):

    sc_df, comp_df, fact_df = load_dataframes(sc_file=sc_file,
                                              comp_file=comp_file,
                                              fact_file=fact_file)

    return DataManager(inputs_wrapper, sc_df, comp_df, fact_df)


def test_copy(data_manager):
    dm = data_manager.copy()
    assert data_manager._iw.as_dict() == dm._iw.as_dict()
    assert_frame_equal(data_manager.sc_df, dm.sc_df)
    assert_frame_equal(data_manager.comp_df, dm.comp_df)
    assert_frame_equal(data_manager.fact_df, dm.fact_df)

    dm.reset()
    assert len(dm.sc_df) == 0
    assert len(dm.comp_df) == 0
    assert len(dm.fact_df) == 0
    assert len(data_manager.fact_df) == 8


def test_empty_dm(inputs_wrapper):
    dm = DataManager(inputs_wrapper)
    assert len(dm.sc_df) == 0
    assert len(dm.comp_df) == 0
    assert len(dm.fact_df) == 0

    sckeys = ['sc_id', 'matrix']
    compkeys = ['sc_id', 'comp_id', 'ucoord', 'ccoord', 'comp','cstat','nondisc']
    factkeys = ['entry_id', 'sc_id', 
                'comp_id',
                'iter_id', 'module',
                'ori_occu', 'ori_corr',
                'calc_status',
                'map_occu', 'map_corr',
                'e_prim', 'other_props']

    for k in sckeys:
        assert k in dm.sc_df
    for k in compkeys:
        assert k in dm.comp_df
    for k in factkeys:
        assert k in dm.fact_df


def test_save_load(data_manager):
    data_manager.auto_save()

    assert os.path.isfile(WRAPPER_FILE)
    assert os.path.isfile(SC_FILE)
    assert os.path.isfile(COMP_FILE)
    assert os.path.isfile(FACT_FILE)
    assert os.path.getsize(WRAPPER_FILE) > 0
    assert os.path.getsize(SC_FILE) > 0
    assert os.path.getsize(COMP_FILE) > 0
    assert os.path.getsize(FACT_FILE) > 0

    dm = DataManager.auto_load()
    assert data_manager._iw.as_dict() == dm._iw.as_dict()
    assert_frame_equal(data_manager.sc_df, dm.sc_df)
    assert_frame_equal(data_manager.comp_df, dm.comp_df)
    assert_frame_equal(data_manager.fact_df, dm.fact_df)

    os.remove(WRAPPER_FILE)
    os.remove(SC_FILE)
    os.remove(COMP_FILE)
    os.remove(FACT_FILE)


def test_muitiple_rebuilds(inputs_wrapper):
    for _ in range(3):
        mat = random.choice(get_diag_matrices(4))
        mat[0][1] = random.choice(list(range(-10, 10)))
        mat[0][2] = random.choice(list(range(-10, 10)))
        mat[1][2] = random.choice(list(range(-10, 10)))
        for _ in range(5):
            occu = [random.choice([0, 1, 2]) for _ in range(12)] + [0 for _ in range(4)]
            s = structure_from_occu(occu, inputs_wrapper.prim, mat)
            corr = inputs_wrapper.subspace.corr_from_structure(s, scmatrix=mat)
            for _ in range(10):
                occu_remap = occu_from_structure(s, inputs_wrapper.prim, mat)
                s_remap = structure_from_occu(occu_remap, inputs_wrapper.prim, mat)
                corr_remap = inputs_wrapper.subspace.corr_from_structure(s_remap, scmatrix=mat)
                assert StructureMatcher().fit(s, s_remap)
                assert np.allclose(corr_remap, corr)
                occu = deepcopy(occu_remap)
                s = s_remap.copy()
                corr = deepcopy(corr_remap)


def test_get_structures(data_manager):
    # print(data_manager.fact_df)
    df = data_manager.fact_df_with_structures
    for m, s, occu, corr in zip(df.matrix, df.ori_str, df.ori_occu, df.ori_corr):
        print("Length of corr:", len(corr))
        print("Length of subspace:",data_manager.subspace.num_corr_functions)
        assert np.allclose(data_manager.subspace.
                           corr_from_structure(s,scmatrix=m), corr)

        s_build = structure_from_occu(occu, data_manager._iw.prim, m)
        cspace_occu = data_manager.subspace.occupancy_from_structure(s,scmatrix=m, encode=True)
        util_occu = occu_from_structure(s, data_manager._iw.prim, m)
        assert np.allclose(cspace_occu, util_occu)
        assert StructureMatcher().fit(s, s_build)

        # cspace_occu and ori_occu won't match exactly. This is an issue with
        # StructureMatcher.get_mapping. So please only test symmetry matching of
        # structures.
        s_rebuild = structure_from_occu(util_occu, data_manager._iw.prim, m)
        assert StructureMatcher().fit(s_build, s_rebuild)

    for m, s, corr in zip(df.matrix, df.map_str, df.map_corr):
        assert ((pd.isna(s) and np.all(pd.isna(corr))) or
                (not pd.isna(s) and not np.all(pd.isna(corr))))


def test_find_sc_id(data_manager):
    sc_df = data_manager.sc_df
    for i in sc_df.sc_id:
        mats = sc_df.matrix[sc_df.sc_id == i].reset_index(drop=True)
        assert len(mats) == 1 # No duplicacy of supercell matrix in dataset.

        mat = mats.iloc[0]
        print("mat:", mat)
        print("mat shape:", np.array(mat, dtype=int).shape)
        assert data_manager.find_sc_id(mat) == i
        perm1 = [[0, 1, 0], [1, 0, 0], [0, 0, -1]]
        perm2 = [[0, 0, 1], [0, -1, 0], [1, 0, 0]]
        perm3 = [[-1, 0, 0], [0, 0, 1], [0, 1, 0]]
        mat1 = np.array(np.dot(perm1, mat), dtype=int)
        mat2 = np.array(np.dot(perm2, mat), dtype=int)
        mat3 = np.array(np.dot(perm3, mat), dtype=int)
        assert data_manager.find_sc_id(mat1) == i  # Symmetry equivalence also not duplicating.
        assert data_manager.find_sc_id(mat2) == i
        assert data_manager.find_sc_id(mat3) == i


def test_insert_supercell(data_manager):
    dm = data_manager.copy()
    new_id = dm.insert_one_supercell([[-2,2,2],[1,-1,1],[1,1,-1]])
    assert new_id == 1  # Can find duplicacy.
    assert len(dm.sc_df) == 2
    new_id2 = dm.insert_one_supercell([[-1, 5, 9], [2, 3, 4], [4, 1, -2]])
    assert new_id2 == 2
    assert len(dm.sc_df) == 3
    assert len(data_manager.sc_df) == 1


def test_find_comp_id(data_manager):
    # sl_sizes = [1, 3]
    # bits = [[Br-], [Li+, Ca+, Vac]]
    ucoord = [0.25, 0.75]
    comp = [Composition({'Br-': 1.0}), Composition({'Li+': 1 / 12, 'Ca+': 1 / 4})]
    ccoord = [0.75]
    c_id = data_manager.find_comp_id(ucoord, sc_id=0)
    assert c_id == 0
    ucoord_trans = data_manager.compspace.translate_format(comp, from_format='composition')
    print("ucoord:", ucoord)
    print("ucoord_trans:", ucoord_trans)
    print("compspace bits:", data_manager.compspace.bits)
    print("compspace slsizes:", data_manager.compspace.sl_sizes)
    c_id = data_manager.find_comp_id(comp, sc_id=0, comp_format='composition')
    assert c_id == 0
    c_id = data_manager.find_comp_id(ccoord, sc_id=0, comp_format='constr')
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
    assert len(data_manager.fact_df) == 8

    # Can detect new composition.
    occu_3 = [2, 1, 2, 2, 0, 2, 2, 2, 0, 0, 2, 2, 0, 0, 0, 0]
    sc_id, comp_id, new_id = dm.insert_one_occu(occu_3, sc_id=0)
    assert comp_id == 2 and new_id == 9
    assert len(dm.comp_df)==3 and len(dm.fact_df) == 10
    assert len(data_manager.comp_df) == 2 and len(data_manager.fact_df) == 8


def test_insert_structure(data_manager):
    occu = [2, 1, 2, 2, 0, 2, 2, 2, 0, 0, 2, 2, 0, 0, 0, 0]
    mat = data_manager.sc_df.matrix.reset_index(drop=True).iloc[0]
    s = structure_from_occu(occu, data_manager._iw.prim, mat)
    dm = data_manager.copy()
    sc_id, comp_id, new_id = dm.insert_one_structure(s, sc_id=0, iter_id=1)
    assert new_id == 8 and comp_id == 2
    assert len(dm.comp_df) == 3 and len(dm.fact_df) == 9
    assert len(data_manager.comp_df) == 2 and len(data_manager.fact_df) == 8

    #Can detect new matrix.
    mat = np.array([[2, 0, 0], [0, 2, 0], [0, 0, 1]], dtype=int)
    s2 = structure_from_occu(occu, data_manager._iw.prim, mat)
    sc_id, comp_id, new_id = dm.insert_one_structure(s2, sc_mat=mat, iter_id=1)
    assert sc_id == 1 and comp_id == 3 and new_id == 9
    assert len(dm.sc_df) == 2 and len(dm.comp_df) == 4 and len(dm.fact_df) == 10
    assert len(data_manager.sc_df) == 1 and len(data_manager.comp_df) == 2 and len(data_manager.fact_df) == 8


def test_remove_entree(data_manager):
    dm = data_manager.copy()
    dm.remove_entree_by_id(entry_ids = [0])
    assert np.allclose(dm.fact_df.entry_id, np.arange(7))
    assert np.allclose(dm.fact_df.comp_id, [0,1,1,0,0,1,1])
    assert np.allclose(dm.fact_df.sc_id, [0,0,0,0,0,0,0])

    occu_rm = dm.fact_df.loc[dm.fact_df.entry_id == 6, 'ori_occu'].reset_index(drop=True).iloc[0]
    sc_id_rm = dm.fact_df.loc[dm.fact_df.entry_id == 6, 'sc_id'].reset_index(drop=True).iloc[0]
    dm.remove_entree_by_id(entry_ids = [6])
    assert np.allclose(dm.fact_df.entry_id, np.arange(6))
    assert np.allclose(dm.fact_df.comp_id, [0,1,1,0,0,1])
    assert np.allclose(dm.fact_df.sc_id, [0,0,0,0,0,0])
    # Removed occu should not exist.
    for i, o in zip(dm.fact_df.sc_id, dm.fact_df.ori_occu):
        assert not (i == sc_id_rm and np.allclose(occu_rm, o))

    assert len(data_manager.fact_df) == 8

def test_remove_comps(data_manager):
    dm = data_manager.copy()
    ucoord_rm = dm.comp_df.loc[dm.comp_df.comp_id == 0, 'ucoord'].reset_index(drop=True).iloc[0]
    sc_id_rm = dm.comp_df.loc[dm.comp_df.comp_id == 1, 'sc_id'].reset_index(drop=True).iloc[0]

    dm.remove_comps_by_id(comp_ids = [0])
    assert np.allclose(dm.fact_df.entry_id, np.arange(4))
    assert np.allclose(dm.comp_df.comp_id, [0])
    assert np.allclose(dm.fact_df.comp_id, [0,0,0,0])
    
    # Removed ucoords should not exist anymore.
    fact_merge = dm.fact_df.merge(dm.comp_df, on='comp_id', how='left')
    print("fact_merge:", fact_merge)
    for i, u in zip(fact_merge.sc_id_x, fact_merge.ucoord):
        assert not (i == sc_id_rm and np.allclose(ucoord_rm, u))

    assert len(data_manager.comp_df) == 2 and len(data_manager.fact_df) == 8


def test_remove_supercells(data_manager):
    dm = data_manager.copy()
    dm.remove_supercells_by_id(sc_ids= [0])
    assert len(dm.sc_df) == 0
    assert len(dm.comp_df) == 0
    assert len(dm.fact_df) == 0
    assert (len(data_manager.sc_df) == 1 and len(data_manager.comp_df) == 2
            and len(data_manager.fact_df) == 8)


def test_reset(data_manager):
    dm = data_manager.copy()
    dm.reset()
    assert len(dm.sc_df) == 0
    assert len(dm.comp_df) == 0
    assert len(dm.fact_df) == 0
    assert (len(data_manager.sc_df) == 1 and len(data_manager.comp_df) == 2
            and len(data_manager.fact_df) == 8)


def test_get_eid_status(data_manager):
    assert sorted(data_manager.get_eid_w_status('SC')) == [0,1,2,3]
    assert sorted(data_manager.get_eid_w_status('CL')) == [4,5,6,7]
