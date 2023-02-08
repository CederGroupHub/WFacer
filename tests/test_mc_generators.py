"""Test sample generators."""
import pytest
from itertools import chain
import random
import numpy as np
import numpy.testing as npt

from pymatgen.analysis.structure_matcher import StructureMatcher

from smol.moca import Ensemble
from smol.cofe.space.domain import get_allowed_species

from CEAuto.sample_generators import (CanonicalSampleGenerator,
                                      SemigrandSampleGenerator)

from .utils import (gen_random_neutral_counts,
                    gen_random_neutral_occupancy,
                    get_counts_from_occu)

all_generators = [CanonicalSampleGenerator, SemigrandSampleGenerator]


@pytest.fixture(params=all_generators)
def generator(cluster_expansion, request):
    sc_matrix = [[3, 0, 0], [0, 3, 0], [0, 0, 3]]
    ensemble = Ensemble.from_cluster_expansion(cluster_expansion,
                                               sc_matrix)
    if "Canonical" in request.param.__name__:
        counts = gen_random_neutral_counts(ensemble.sublattices)
        return request.param(cluster_expansion,
                             sc_matrix,
                             counts)
    elif "Semigrand" in request.param.__name__:
        species = set(chain(*get_allowed_species(cluster_expansion
                                                 .cluster_subspace
                                                 .structure)))
        chempots = {p: random.random() for p in species}
        return request.param(cluster_expansion,
                             sc_matrix,
                             chempots)


def test_ground_state(generator):
    sm = StructureMatcher()
    occu = generator.get_ground_state_occupancy()
    s = generator.get_ground_state_structure()
    assert sm.fit(generator.processor.structure_from_occupancy(occu),
                  s)
    assert np.isclose(s.charge, 0)

    if "Canonical" in generator.__class__.__name__:
        counts = get_counts_from_occu(occu, generator.sublattices)
        npt.assert_array_almost_equal(counts, generator.counts)


def test_unfreeze(generator):
    sm = StructureMatcher()
    prev_occus = [gen_random_neutral_occupancy(generator.sublattices)
                  for _ in range(20)]
    prev_structs = [generator.processor.structure_from_occupancy(o)
                    for o in prev_occus]
    sample = generator.get_unfrozen_sample(prev_structs, 50)
    # No duplication with old and among themselves.
    for sid, s in enumerate(sample):
        dupe = False
        for s_old in prev_structs + sample[sid + 1:]:
            if sm.fit(s_old, s):
                dupe = True
                break
        assert not dupe
        if "Canonical" in generator.__class__.__name__:
            occu = generator.processor.occupancy_from_structure(s)
            counts = get_counts_from_occu(occu, generator.sublattices)
            npt.assert_array_almost_equal(counts, generator.counts)

