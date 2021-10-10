from CEAuto import InputsWrapper, HistoryWrapper
from CEAuto.calc_writer.base import BaseWriter
from CEAuto.calc_manager.base import BaseManager
from CEAuto.calc_reader.base import BaseReader
from CEAuto.config_paths import *
from CEAuto.utils.format_utils import merge_dicts

import pytest
import numpy as np
import os
from shutil import copyfile

from smol.cofe.space.domain import get_allowed_species
from smol.cofe import ClusterSubspace
from smol.cofe.extern import EwaldTerm

from .utils import assert_msonable


def test_bits(inputs_wrapper):
    prim_bits = get_allowed_species(inputs_wrapper.prim)
    for sl_id, sl in enumerate(inputs_wrapper.sublat_list):
        sl_bits = prim_bits[sl[0]]
        for sid in sl:
            assert prim_bits[sid] == sl_bits
        sl_bits = inputs_wrapper.bits[sl_id]
        for sid in sl:
            assert prim_bits[sid] == sl_bits

    assert inputs_wrapper.bits == [prim_bits[sl[0]]
                                   for sl in inputs_wrapper.sublat_list]

    bits = [tuple(sl_bits) for sl_bits in inputs_wrapper.bits]
    assert len(set(bits)) == len(inputs_wrapper.bits)

    bits = [tuple(sl_bits) for sl_bits in prim_bits]
    assert len(set(bits)) == len(inputs_wrapper.bits)       

def test_prim(inputs_wrapper):
    bits = inputs_wrapper.bits
    lattice = inputs_wrapper.lattice
    frac_coords = inputs_wrapper.frac_coords
    sublat_list = inputs_wrapper.sublat_list

    prim_copy = InputsWrapper(bits=bits, 
                              lattice=lattice,
                              frac_coords=frac_coords,
                              sublat_list=sublat_list).prim
    prim = inputs_wrapper.prim

    print("prim_copy", prim_copy)
    print("prim", prim)

    assert prim.lattice == prim_copy.lattice
    assert np.allclose(prim.frac_coords,prim_copy.frac_coords)
    assert (get_allowed_species(prim) ==
            get_allowed_species(prim_copy))
    assert abs(prim.charge) < 1E-8
    assert abs(prim_copy.charge) < 1E-8

def test_subspace_iw(inputs_wrapper):
    cs = ClusterSubspace.from_cutoffs(inputs_wrapper.prim,
                                      cutoffs={2:7.0, 3:5.0, 4:5.0})
    if inputs_wrapper.is_charged_ce:
        cs.add_external_term(EwaldTerm())

    assert get_allowed_species(inputs_wrapper.prim) == get_allowed_species(cs.structure)
    print("iw space options:", inputs_wrapper.space_options)
    print("Subspace comparison:")
    print("iw expansion str:\n", inputs_wrapper.subspace.expansion_structure)
    print("cs expansion str:\n", cs.expansion_structure)
    print("iw cutoffs:", inputs_wrapper.subspace.cutoffs)
    print("cs cutoffs:", cs.cutoffs)
    print("iw num_corr_functions:", inputs_wrapper.subspace.num_corr_functions)
    print("cs num_corr_funtions:", cs.num_corr_functions)
    print("iw external terms:", inputs_wrapper.subspace.external_terms)
    print("cs external terms:", cs.external_terms)
    for i, (iw_o, cs_o) in enumerate(zip(inputs_wrapper.subspace.orbits, cs.orbits)):
        print("orbit id:", i)
        print("  iw orbit base:", iw_o.base_cluster)
        print("  cs orbit base:", cs_o.base_cluster)
        print("  iw orbit bit_combos:", iw_o.bit_combos)
        print("  cs orbit bit_combos:", cs_o.bit_combos)
        print("  iw orbit bases:", iw_o.bases_array)
        print("  iw orbit bases:", cs_o.bases_array)

    assert cs == inputs_wrapper.subspace

def test_last_ce(history_wrapper):
    ce = history_wrapper.last_ce
    assert np.all(ce.coefs[ : -1] == 0)
    assert np.all(ce.coefs[-1] == 1)

def test_calc_writer(inputs_wrapper):
    assert isinstance(inputs_wrapper.calc_writer,
                      BaseWriter)

def test_calc_manager(inputs_wrapper):
    assert isinstance(inputs_wrapper.calc_manager,
                      BaseManager)


def test_calc_reader(inputs_wrapper):
    assert isinstance(inputs_wrapper.calc_reader,
                      BaseReader)

def test_options(inputs_wrapper):
    options_d = [inputs_wrapper.enumerator_options,
                 inputs_wrapper.calc_writer_options,
                 inputs_wrapper.calc_reader_options,
                 inputs_wrapper.calc_manager_options,
                 inputs_wrapper.featurizer_options,
                 inputs_wrapper.fitter_options,
                 inputs_wrapper.gs_checker_options,
                 inputs_wrapper.space_options]
    merged_d = merge_dicts(options_d, keep_all=True)
    print("merged_d:\n", merged_d)
    for key, vals in merged_d.items():
        assert len(vals) >= 1
        if len(vals) > 1:
            assert all(v == vals[0] for v in vals)
            assert key in (list(inputs_wrapper.calc_writer_options.keys()) +
                           list(inputs_wrapper.calc_reader_options.keys()) +
                           list(inputs_wrapper.calc_manager_options.keys()))
        # No duplicacy of keys.

def test_copy_iw(inputs_wrapper):
    assert isinstance(inputs_wrapper.copy(), InputsWrapper)

def test_copy_hw(history_wrapper):
    assert isinstance(history_wrapper.copy(), HistoryWrapper)

def test_msonable_iw(inputs_wrapper):
    assert_msonable(inputs_wrapper)

def test_msonable_hw(history_wrapper):
    assert_msonable(history_wrapper)

def test_saveload_iw(inputs_wrapper):
    inputs_wrapper.auto_save()
    assert os.path.isfile(WRAPPER_FILE)
    iw2 = InputsWrapper.auto_load()

    iwd1 = inputs_wrapper.as_dict()
    iwd2 = iw2.as_dict()
    for key in iwd2:
        if key not in iwd1:
            print("{} not in new iwd.".format(key))
        if iwd1[key] != iwd2[key]:
            print("Differing key: ", key)
            print("iw1:", iwd1[key])
            print("iw2:", iwd2[key])

    assert abs(inputs_wrapper.prim.charge) < 1E-8
    assert abs(iw2.prim.charge) < 1E-8
    assert inputs_wrapper.as_dict() == iw2.as_dict()
    os.remove(WRAPPER_FILE)

def test_saveload_hw(history_wrapper):
    history_wrapper.auto_save()
    assert os.path.isfile(CE_HISTORY_FILE)
    hw2 = HistoryWrapper.auto_load()
    assert history_wrapper.as_dict() == hw2.as_dict()
    os.remove(CE_HISTORY_FILE)

def test_load_options_iw():
    path = os.path.dirname(os.path.abspath(__file__))
    options_file = os.path.join(path, 'data/options_test.yaml')
    wrapper_file = os.path.join(path, 'data/inputs_wrapper_test.json')
    prim_file = os.path.join(path, 'data/LiCaBr_prim.json')

    assert os.path.isfile(options_file)
    assert os.path.isfile(wrapper_file)
    assert os.path.isfile(prim_file)

    copyfile(prim_file, './LiCaBr_prim.json')

    iw1 = InputsWrapper.auto_load(options_file=options_file)
    iw2 = InputsWrapper.auto_load(wrapper_file=wrapper_file)
    d1 = iw1.as_dict()
    d2 = iw2.as_dict()
    d1.pop('prim')
    d2.pop('prim')
    d1['_subspace'].pop('structure')
    d2['_subspace'].pop('structure')
    d1['_subspace'].pop('expansion_structure')
    d2['_subspace'].pop('expansion_structure')
    # These structures are not comparable, because the inputswrapper
    # from a fresh prim file will reconstruct self.prim to be charge
    # neutral.

    assert iw1.subspace.num_corr_functions == iw2.subspace.num_corr_functions
    assert len(iw1.subspace.external_terms) == len(iw2.subspace.external_terms)
    assert get_allowed_species(iw1.prim) == get_allowed_species(iw2.prim)

    assert d1 == d2

    os.remove('./LiCaBr_prim.json')
