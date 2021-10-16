from CEAuto.ce_handler import *
from CEAuto.utils.comp_utils import normalize_compstat
from smol.moca.utils.occu_utils import occu_to_species_stat

from smol.moca.comp_space import CompSpace
from smol.cofe.space.domain import get_allowed_species
from smol.cofe import ClusterExpansion

from monty.serialization import loadfn
import os
import pytest
import numpy as np
import random
import itertools

from pymatgen.analysis.structure_matcher import StructureMatcher

DATADIR = os.path.join(os.path.dirname(__file__),'data')
handlers = [CanonicalmcHandler, SemigrandmcHandler]

def get_compspace(handler):
    return CompSpace(handler.bits, handler.sl_sizes)


def get_cluster_expansion(iw):
    coefs_ = (np.random.
              random(iw.subspace.num_corr_functions +
                     len(iw.subspace.external_terms)))
    coefs_ = coefs_ - 0.5
    coefs_[0] = 1.0
    coefs_[-len(iw.subspace.external_terms):] = 0.3
    return ClusterExpansion(iw.subspace, coefs_)


@pytest.fixture(params=handlers)
def handler(inputs_wrapper, request):
    sc_mat = [[-1,1,1],[1,-1,1],[1,1,-1]]

    cspace = CompSpace(inputs_wrapper.bits, inputs_wrapper.sl_sizes)
    cluster_expansion = get_cluster_expansion(inputs_wrapper)

    kwargs = {}
    if "Canonical" in request.param.__name__:
        compstat = random.choice(cspace.int_grids(sc_size=4, form='compstat'))
        compstat = normalize_compstat(compstat, sc_size=4)
        # Must initialize with normalized comp.
        kwargs['compstat'] = compstat
    elif "Semigrand" in request.param.__name__:
        # Inactive species should not be included in chemical potentials.
        bits = [sl_bits for sl_bits in inputs_wrapper.bits if len(sl_bits) > 1]
        kwargs['chemical_potentials'] = {k: kid * 0.01
                                         for kid, k in
                                         enumerate(itertools.chain(*bits))}
    # Add more as you implement.

    return request.param(cluster_expansion, sc_mat, **kwargs)


def test_init_occu(handler):
    cspace = get_compspace(handler)

    int_comp = random.choice(cspace.int_grids(sc_size=4, form='compstat'))
    init_occu = handler._initialize_occu_from_int_comp(int_comp)

    int_comp_rel = occu_to_species_stat(init_occu, handler.ensemble.all_sublattices)

    assert int_comp == int_comp_rel


def test_ground_state(handler):
    solutions = [handler.get_ground_state() for _ in range(5)]
    occus = [sol[0] for sol in solutions]
    es = [sol[1] for sol in solutions]

    sol_counts = []
    sm = StructureMatcher()
    for o in occus:
        dupe = False
        s1 = handler.processor.structure_from_occupancy(o)
        for kid, (k, v) in enumerate(sol_counts):
            s2 = handler.processor.structure_from_occupancy(k)
            if sm.fit(s1, s2):
                sol_counts[kid][1] += 1
                dupe = True
                break
        if not dupe:
            sol_counts.append([o,1])

    dedup_counts = [v for k,v in sol_counts]
    assert max(dedup_counts) >= 3
    assert 2*(max(es)-min(es))/(max(es)+min(es)) < 0.05


def test_get_sample(handler):
    samples = handler.get_unfreeze_sample()
    cspace = get_compspace(handler)

    # Can get deduplicated samples under requirements.
    dedup_structures = []
    for occu in samples:
        compstat = occu_to_species_stat(occu, handler.ensemble.all_sublattices)
        ucoord = cspace.translate_format(compstat,
                                         from_format='compstat',
                                         to_format='unconstr')
        s = handler.processor.structure_from_occupancy(occu)

        for s2 in dedup_structures:
            assert not StructureMatcher().fit(s, s2)

        dedup_structures.append(s)

        assert cspace._is_in_subspace(ucoord, sc_size=4)

        if 'Canonical' in handler.__class__.__name__:
            assert handler.compstat == normalize_compstat(compstat, sc_size=4)

    assert len(dedup_structures) > 0
