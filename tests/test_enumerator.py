from CEAuto import StructureEnumerator, DataManager
from CEAuto.time_keeper import TimeKeeper
from CEAuto.utils.frame_utils import load_dataframes
from CEAuto.utils.supercells import is_proper_sc
from CEAuto.utils.comp_utils import check_comp_restriction
from CEAuto.utils.occu_utils import structure_from_occu

import pytest
import numpy as np
import os
import itertools

from pymatgen.core import Species
from pymatgen.analysis.structure_matcher import StructureMatcher

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
    assert np.all(dm.fact_df.sc_id == 0) # test dataset only has 1 sc_id
    return dm


@pytest.fixture
def enumerator_loaded(data_manager_loaded, history_wrapper_loaded):
    # Start from iteration 1.
    validate_data(data_manager_loaded, history_wrapper_loaded, iter_id=0, module='check')
    return StructureEnumerator(data_manager_loaded, history_wrapper_loaded)


def test_enumerate_scs(enumerator_empty):
    lat = enumerator_empty.prim.lattice
    # Test without transmat.
    mats = enumerator_empty.enumerate_sc_matrices()

    for mat in mats:
        assert np.array(mat).shape == (3, 3)
        assert isinstance(mat, list) and isinstance(mat[0], list)

    assert is_proper_sc(mats[0], lat,
                        max_cond=enumerator_empty.max_sc_cond)
    for mat in mats[1:]:
        assert mat[0][0] == mats[0][0][0]
        assert mat[1][1] == mats[0][1][1]
        assert mat[2][2] == mats[0][2][2]
        assert mat[0][0] <= mat[1][1] and mat[1][1] <= mat[2][2]
        assert mat[0][1] <= mat[0][0]
        assert mat[0][2] <= mat[0][0]
        assert mat[1][2] <= mat[0][0]
        assert mat[1][0] == 0
        assert mat[2][0] == 0
        assert mat[2][1] == 0
        assert int(round(abs(np.linalg.det(mat)))) == enumerator_empty.sc_size
    
    assert len(enumerator_empty.sc_df) == len(mats)


def test_enumerate_comps(enumerator_empty):
    if Species.from_string("Li+") in itertools.chain(*enumerator_empty.bits):
        enumerator_empty.comp_restrictions = {'Li+':(0.1, 0.9)}

    all_comps = enumerator_empty.enumerate_comps()
    dm = enumerator_empty._dm
    for comp in all_comps:
        assert dm.find_comp_id(comp, comp_format='composition') is not None
        assert check_comp_restriction(comp,
                                      enumerator_empty.bits,
                                      enumerator_empty.sl_sizes,
                                      enumerator_empty.comp_restrictions)

# This covers ce_handler as well!
def test_generate_structures(enumerator_loaded):
    n_init = len(enumerator_loaded.fact_df)
    fact_gen = enumerator_loaded.generate_structures(iter_id=1)
    assert len(fact_gen) != 0
    assert len(enumerator_loaded.fact_df) - n_init == len(fact_gen)
    assert np.all(fact_gen.iter_id == 1)
    # Weak test on no duplicacy of occupancy arrays.
    prim = enumerator_loaded.prim
    occus = fact_gen.ori_occu
    scmats = fact_gen.merge(enumerator_loaded.sc_df,on='sc_id',how='left').matrix
    strs = []
    for oid, (occu, mat) in enumerate(zip(occus, scmats)):
        strs.append(structure_from_occu(occu, prim, mat))

    sm = StructureMatcher()
    undupe_ids = []
    dupe_pairs = []
    for sid, s in enumerate(strs):
        dupe = None
        for uid in undupe_ids:
            if sm.fit(strs[uid], strs[sid]):
                dupe = uid
        if dupe is not None:
            dupe_pairs.append((uid, sid))
        else:
            undupe_ids.append(sid)

    if len(dupe_pairs) !=0:
        print("Error: found duplicacies in added structures!")
        print("Fact table:", fact_gen)
        print("Duplicacies:", dupe_pairs)

    assert len(dupe_pairs) == 0
