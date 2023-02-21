"""Test occupancy generation."""
import numpy as np
import numpy.testing as npt
from smol.moca.utils.occu import get_dim_ids_table, occu_to_counts

from WFacer.utils.occu import get_random_occupancy_from_counts

from ..utils import gen_random_neutral_counts


def test_random_occu_from_counts(ensemble):
    for _ in range(10):
        counts = gen_random_neutral_counts(ensemble.sublattices)
        for _ in range(10):
            occu = get_random_occupancy_from_counts(ensemble, counts)

            assert np.all(occu >= 0)

            table = get_dim_ids_table(ensemble.sublattices)
            npt.assert_array_equal(occu_to_counts(occu, len(counts), table), counts)
