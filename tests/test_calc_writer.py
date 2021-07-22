from CEAuto.calc_writer import *

import pytest
import os
import pandas as pd
import numpy as np
import shutil

from pymatgen.core import Structure
from pymatgen.core.periodic_table import Specie
from pymatgen.io.vasp import *
from pymatgen.analysis.structure_matcher import StructureMatcher

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

    return sock

@pytest.fixture
def arch_vasp_writer(data_manager):
    return ArchVaspWriter(data_manager, path = 'vasp_run_write')

@pytest.fixture
def mongo_vasp_writer(data_manager):
    return MongoVaspWriter(data_manager)


def decorate(s, decors):
    species = []
    for site in s:
        sym = site.specie.symbol
        species.append(Specie(sym, decors[sym]))
    return Structure(s.lattice, species, s.frac_coords)

def test_arch_write(arch_vasp_writer):
    s = (arch_vasp_writer._dm.
         fact_with_structures.ori_str.iloc[4])
    eid = (arch_vasp_writer._dm.fact_df.
           entry_id.iloc[4])

    arch_vasp_writer._write_single(s, eid)

    assert os.path.isdir(arch_vasp_writer.path)
    epath = os.path.join(arch_vasp_writer.path,str(eid))
    
    incar_path = os.path.join(epath, 'INCAR')
    poscar_path = os.path.join(epath, 'POSCAR')
    potcar_path = os.path.join(epath, 'POTCAR')
    kpoints_path = os.path.join(epathm 'KPOINTS')

    assert isinstance(Incar.from_file(incar_file), Incar)
    assert isinstance(Poscar.from_file(poscar_file), Poscar)
    s_rel = Poscar.from_file(poscar_file).structure
    s_rel = decorate(s_rel, decors=
                     {'Li': 1, 'Ca': 1, 'Br': -1})
    assert sm.fit(s_rel, s)

    assert isinstance(Potcar.from_file(potcar_file), Potcar)
    assert isinstance(Kpoints.from_file(kpoints_file), Kpoints)
    
    shutil.rmtree(arch_vasp_writer.path)

def test_mongo_write(mongo_vasp_writer):
    s = (mongo_vasp_writer._dm.
         fact_with_structures.ori_str.iloc[4])
    eid = (mongo_vasp_writer._dm.fact_df.
           entry_id.iloc[4])

    mongo_vasp_writer._write_single(s, eid)
    root_name = os.path.split(os.get_cwd())[-1]

    entry_name = 'ce_{}_{}'.format(root_name, eid)

    wfs = list(mongo_vasp_writer._lpad.workflows.
               find({'name':entry_name}))

    assert len(wfs) == 1
    wf_dict = wfs[0]

    nodes = wf_dict['nodes']
    assert len(nodes) == 2

    wf = mongo_vasp_writer._lpad.get_wf_by_fw_id(nodes[0])
    fw_names = [fw.name for fw in wf.fws]

    assert (entry_name + '_optimization') in fw_names
    assert (entry_name + '_static') in fw_names

    mongo_vasp_writer._lpad.delete_wf(nodes[0])
 
    wfs = list(mongo_vasp_writer._lpad.workflows.
               find({'name':entry_name}))
    assert len(wfs) == 0
