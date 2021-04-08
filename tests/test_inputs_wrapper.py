from CEAuto import *
from CEAuto.calc_writer.base import BaseWriter
from CEAuto.calc_manager.base import BaseManager
from CEAuto.calc_reader.base import BaseReader

import pytest
import numpy as np

from smol.cofe.space.domain import get_allowed_species
from smol.cofe import ClusterSubspace
from smol.cofe.extern import EwaldTerm

from pymatgen.analysis.structure_matcher import (StructureMatcher,
                                                 OrderDisorderElementComparator)

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


def test_bits(inputs_wrapper):
    prim_bits = get_allowed_species(inputs_wrapper.prim)
    for sl in inputs_wrapper.sublat_list:
        sl_bits = prim_bits[sl[0]]
        for sid in sl:
            assert prim_bits[sid] == sl_bits
    assert inputs_wrapper.bits == [prim_bits[sl[0]]
                                   for sl in inputs_wrapper.sublat_list]
    bits = [tuple(sl_bits) for sl_bits in inputs_wrapper.bits]

    assert len(set(bits)) == len(inputs_wrapper.bits)

def test_prim(inputs_wrapper):
    lat_data = {'bits':inputs_wrapper.bits,
                'lattice':inputs_wrapper.lattice,
                'frac_coords':inputs_wrapper.frac_coords,
                'sublat_list':inputs_wrapper.sublat_list}

    prim_copy = InputsWrapper(lat_data).prim
    prim = inputs_wrapper.prim

    assert (prim.lattice == prim_copy.lattice and
            np.allclose(prim.frac_coords,prim_copy.frac_coords) and
            get_allowed_species(prim) == get_allowed_species(prim_copy))

def test_subspace(inputs_wrapper):
    cs = ClusterSubspace.from_cutoffs(inputs_wrapper.prim,
                                      cutoffs={2:4.0,3:3.0,4:3.0})
    if inputs_wrapper.is_charged_ce:
        cs.add_external_term(EwaldTerm())

    assert cs == inputs_wrapper.subspace
    _ = inputs_wrapper.get_ce_n_iters_ago()

def test_calc_writer(inputs_wrapper):
    assert isinstance(inputs_wrapper.calc_writer,
                      BaseWriter)

def test_calc_manager(inputs_wrapper):
    assert isinstance(inputs_wrapper.calc_manager,
                      BaseManager)


def test_calc_reader(inputs_wrapper):
    assert isinstance(inputs_wrapper.calc_reader,
                      BaseReader)

def test_msonable(inputs_wrapper):
    iw_reload = InputsWrapper.from_dict(inputs_wrapper.as_dict())
    assert iw_reload.options == inputs_wrapper.options
