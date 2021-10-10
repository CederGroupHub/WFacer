from CEAuto.ce_handler import *
from CEAuto.utils.occu_utils import occu_to_species_stat

from smol.moca import CompSpace
from smol.cofe.space.domain import get_allowed_species

from monty.serialization import loadfn
import os
import pytest
import numpy as np
import random
from pymatgen.analysis.structure_matcher import StructureMatcher

DATADIR = os.path.join(os.path.dirname(__file__),'data')
handlers = [CanoncialMCHandler]

def get_bits_sublats(prim):
    prim_bits = get_allowed_species(prim)
    bits = []
    sl_list = []

    for s_id, s_bits in enumerate(prim_bits):
        dupe = False
        for sl_id, sl_bits in enumerate(bits):
            if sl_bits == s_bits:
                sl_list[sl_id].append(s_id)
                dupe = True
                break
        if not dupe:
            sl_list.append([s_id])
            bits.append(s_bits)

    return bits, sl_list
            

@pytest.fixture
def cluster_expansion(cluster_subspace):
    coefs_ = (np.random.
              random(cluster_subspace.num_corr_functions +
                     len(cluster_subspace.external_terms)))
    coefs_ = coefs_ - 0.5
    coefs_[0] = 1.0
    coefs_[-len(cluster_subspace.external_terms):] = 0.3
    return ClusterExpansion(cluster_subspace, coefs_)

def get_comp_space(cluster_expansion):
    prim = cluster_expansion.cluster_subspace.structure
    bits, sl_list = get_bits_sublats(prim)
    sl_sizes = [len(sl) for sl in sl_list]

    return CompSpace(bits, sl_sizes)

@pytest.fixture(params=handlers)
def handler(cluster_expansion, request):
    sc_mat = [[-1,1,1],[1,-1,1],[1,1,-1]]

    cspace = get_comp_space(cluster_expansion)

    kwargs = {}
    if request.param == CanonicalMCHandler:
        compstat = random.choice(cspace.int_grids(sc_size=4, form='compstat'))
        kwargs['compstat'] = compstat

    if request.param == SemigrandDiscMCHandler:
        mu = [0.3 + i*0.01 for i in range(cspace.dim)]
        kwargs['mu'] = mu
    if request.param == SemigrandMCHandler:
        chemical_potentials = {}
        for i, sp in enumerate(cspace.species):
            chemical_potentials[sp] = 0.3 + i*0.01
        kwargs['chemical_potentials'] = chemical_potentials

    return request.param(cluster_expansion, sc_mat, **kwargs)


def test_init_occu(handler):
    cspace = get_comp_space(handler.ce)

    int_comp = random.choice(cspace.int_grid(sc_size=4, form='compstat'))
    init_occu = handler._initialize_occu_from_int_comp(int_comp)

    int_comp_rel = occu_to_species_stat(int_occu, handler.bits,
                                        handler.sc_sublat_list)

    assert int_comp == int_comp_rel


def test_solve(handler):
    solutions = [handler.solve() for _ in range(5)]
    occus = [sol[0] for sol in solutions]
    es = [sol[1] for sol in solutions]
    
    sol_counts = []
    sm = StructureMatcher()
    for o in occus:
        dupe = False
        s1 = handler._processor.structure_from_occupancy(o)
        for kid, (k, v) in enumerate(sol_counts):
            s2 = handler._processor.structure_from_occupancy(k)
            if sm.fit(s1, s2):
                sol_counts[kid][1] += 1
                dupe = True
                break
        if not dupe:
            sol_counts.append((o,1))

    dedup_counts = [v for k,v in sol_counts]
    assert max(dedup_counts) >= 3
    assert 2*(max(es)-min(es))/(max(es)+min(es)) < 0.05


def test_get_sample(handler):
    samples = handler.get_unfreeze_sample()
    cspace = get_comp_space(handler.ce)

    for occu in samples:
        compstat = occu_to_species_stat(occu, handler.bits,
                                        handler.sc_sublat_list)
        ucoord = cspace.translate_format(compstat,
                                         from_format='compstat',
                                         to_format='unconstr')

        assert cspace._is_in_subspace(ucoord, sc_size=4)
        if 'Canonical' in handler.__class__.__name__:
            assert handler.compstat == compstat
