"""Testing mocked calc manager. Calculations will be skipped."""

from CEAuto.calc_manager import *
from CEAuto.calc_writer import *
from CEAuto import DataManager, InputsWrapper
from CEAuto.utils.frame_utils import load_dataframes

import os
import pandas as pd
import numpy as np

from monty.serialization import loadfn

CWD = os.path.join(os.path.dirname(__file__))
DATADIR = os.path.join(CWD,'data')


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
def data_manager(inputs_wrapper, history):
    sock =  DataManager(inputs_wrapper.prim,
                        inputs_wrapper.bits,
                        inputs_wrapper.sublat_list,
                        inputs_wrapper.subspace,
                        history)
    sc_file = os.path.join(DATADIR,'sc_df_test.csv')
    comp_file = os.path.join(DATADIR,'comp_df_test.csv')
    fact_file = os.path.join(DATADIR,'fact_df_test.csv')

    sc_df, comp_df, fact_df = load_dataframes(sc_file=sc_file,
                                              comp_file=comp_file,
                                              fact_file=fact_file)

    sock._sc_df = sc_df
    sock._comp_df = comp_df
    sock._fact_df = fact_df

    #reset status to an initial state!
    eids = sock.get_eid_w_status('CL')
    
    sock.set_status(eids,'NC')

    return sock

@pytest.fixture
def sge_manager(data_manager):
#Set very small time quota for testing.
    return ArchSGEManager(data_manager, time_limit = 40,
                          check_interval = 10,
                          debug_mode=True)

def sge_write(sge_manager):
    sge_writer = ArchVaspWriter(sge_manager._dm,
                                path=sge_manager.path,
                                debug_mode=True)
    sge_writer.auto_write_entree()

def sge_clear(sge_manager):
    eids = sge_manager._dm.get_eid_w_status('CL')
    for eid in eids:
        epath = os.path.join(sge_manager.path,str(eid))
        os.remove(epath)
     os.remove(sge_manager.path)
    
    #reset status
    sge_manager._dm.set_status(eids,'NC')

def test_sge_run(sge_manager):
    sge_write(sge_manaer)
    remain_quota = sge_manager.auto_run()
    assert remain_quota < 39
    eids = sge_manager._dm.get_eid_w_status('CL')
    assert len(eids) > 0
    assert not(np.any(sge_manager.entree_in_queue(eids)))
    for eid in eids:
        epath = os.path.join(sge_manager.path,str(eid))
        assert os.path.isdir(epath)
    sge_clear(sge_manager)

@pytest.fixture
def mon_manager(data_manager):
    return MongoFWManager(data_manager, time_limit = 40,
                          check_interval = 10,
                          debug_mode=True)

def mon_write(mon_manager):
    mon_writer = MongoVaspWriter(mon_manager._dm, debug_mode=True)
    mon_writer.auto_write_entree()

def mon_clear(mon_manager):
    eids = mon_manager._dm.get_eid_w_status('CL')
    for eid in eids:
        ename = 'ce_{}_{}'.format(mon_manager.root_name,eid)
        wf = mon_manager._lpad.workflows.find_one({'name':entry_name})
        mon_manager._lpad.delete_wf(wf['nodes'][0])
    
    #reset status
    mon_manager._dm.set_status(eids,'NC')

def test_mon_run(mon_manager):
    mon_write(mon_manager)
    remain_quota = mon_manager.auto_run()
    assert remain_quota < 39
    eids = mon_manager._dm.get_eid_w_status('CL')
    assert len(eids) > 0
    assert not(np.any(mon_manager.entree_in_queue(eids)))
    for eid in eids:
        ename = 'ce_{}_{}'.format(mon_manager.root_name,eid)
        wf = list(mon_manager._lpad.find({'name':ename}))
        assert len(wf) > 0
    mon_clear(mon_manager)
    

