"""Test sample generators."""
import pytest
from itertools import chain
import random
import numpy as np
import numpy.testing as npt

from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core import Element

from smol.moca import Ensemble
from smol.cofe.space.domain import get_allowed_species, Vacancy

from CEAuto.sample_generators import (CanonicalSampleGenerator,
                                      SemigrandSampleGenerator)

from .utils import (gen_random_neutral_counts,
                    gen_random_neutral_occupancy,
                    get_counts_from_occu)

all_generators = [CanonicalSampleGenerator, SemigrandSampleGenerator]


def get_oxi_state(s):
    if isinstance(s, (Element, Vacancy)):
        return 0
    else:
        return s.oxi_state


@pytest.fixture(params=all_generators)
def generator(cluster_expansion, request):
    sc_matrix = [[3, 0, 0], [0, 3, 0], [0, 0, 3]]
    ensemble = Ensemble.from_cluster_expansion(cluster_expansion,
                                               sc_matrix,
                                               processor_type="expansion")
    if "Canonical" in request.param.__name__:
        counts = gen_random_neutral_counts(ensemble.sublattices)
        return request.param(cluster_expansion,
                             sc_matrix,
                             counts,
                             duplicacy_criteria="structure")
    elif "Semigrand" in request.param.__name__:
        species = set(chain(*get_allowed_species(cluster_expansion
                                                 .cluster_subspace
                                                 .structure)))
        chempots = {p: random.random() for p in species}
        generator = request.param(cluster_expansion,
                                  sc_matrix,
                                  chempots,
                                  duplicacy_criteria="structure")

        assert len(generator.sampler.mckernels) == 1
        usher = generator.sampler.mckernels[0].mcusher.__class__.__name__
        charge_decorated = False
        for sp in species:
            if get_oxi_state(sp) != 0:
                charge_decorated = True
        if charge_decorated:
            assert usher == "TableFlip"
        else:
            assert usher == "Flip"

        return generator


def test_ground_state(generator):
    sm = StructureMatcher()
    occu = generator.get_ground_state_occupancy()
    s = generator.get_ground_state_structure()
    assert sm.fit(generator.processor.structure_from_occupancy(occu),
                  s)
    assert np.isclose(s.charge, 0)
    corr_std = generator.ce.cluster_subspace\
        .corr_from_structure(s, scmatrix=generator.processor.supercell_matrix)
    npt.assert_array_almost_equal(corr_std,
                                  generator.get_ground_state_features())

    if "Canonical" in generator.__class__.__name__:
        counts = get_counts_from_occu(occu, generator.sublattices)
        npt.assert_array_almost_equal(counts, generator.counts)


def test_unfreeze(generator):
    sm = StructureMatcher()
    prev_occus = [gen_random_neutral_occupancy(generator.sublattices)
                  for _ in range(20)]
    processor = generator.processor
    prev_structs = [processor.structure_from_occupancy(o)
                    for o in prev_occus]
    prev_feats = [(processor.compute_feature_vector(o) / processor.size)
                  .tolist()
                  for o in prev_occus]
    sample, sample_occus, sample_feats =\
        generator.get_unfrozen_sample(prev_structs, prev_feats, 50)
    gs = generator.get_ground_state_structure()
    assert len(sample) >= 20   # Number of samples should not be too few!
    # No duplication with old and among themselves.
    for sid, (s, occu) in enumerate(zip(sample, sample_occus)):
        dupe = False
        for s_old in prev_structs + [gs] + sample[sid + 1:]:
            if sm.fit(s_old, s):
                dupe = True
                break
        assert not dupe
        if "Canonical" in generator.__class__.__name__:
            counts = get_counts_from_occu(occu, generator.sublattices)
            npt.assert_array_almost_equal(counts, generator.counts)
