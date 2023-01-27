import numpy as np

from pymatgen.core import Composition
from pymatgen.analysis.structure_matcher import StructureMatcher

from smol.cofe.wrangling.tools import _energies_above_hull

from CEAuto.utils.convex_hull \
    import (get_min_energy_structures_by_composition,
            get_hull)


def _comp_equals(c1, c2):
    return (c1.element_composition.fractional_composition.
            almost_equals(c2.element_composition.
                          fractional_composition))


def test_min_energy_structures(data_wrangler):
    min_e = \
        get_min_energy_structures_by_composition(data_wrangler)
    prim_size = len(data_wrangler.cluster_subspace.structure)

    # All compositions in wrangler must appear in min_e,
    # and all structures in min_e must match its composition.
    for comp, (e, s) in min_e.items():
        assert _comp_equals(comp, s.composition)
    for entry in data_wrangler:
        entry_comp = Composition({k: v / entry.data["size"] / prim_size
                                  for k, v
                                  in entry.structure.composition
                                 .element_composition.items()})
        assert entry_comp in min_e
        emin, smin = min_e[entry_comp]
        assert entry.energy >= emin
        assert _comp_equals(entry_comp, smin.composition)


def test_hull(data_wrangler):
    hull = get_hull(data_wrangler)
    sm = StructureMatcher()
    prim_size = len(data_wrangler.cluster_subspace.structure)
    # A hull must contain something, unless wrangler is empty.
    assert len(hull) > 0
    e_above_hull = _energies_above_hull(data_wrangler.structures,
                                        data_wrangler.energies,
                                        data_wrangler
                                        .cluster_subspace.structure)
    for i, entry in enumerate(data_wrangler.entries):
        if np.isclose(e_above_hull[i], 0):
            comp = Composition({k: v / entry.data["size"] / prim_size
                                for k, v
                                in entry.structure.composition
                               .element_composition.items()})
            assert np.isclose(hull[comp][0],
                              entry.energy / entry.data["size"] / prim_size)
            assert sm.fit(data_wrangler.structures[i],
                          hull[comp][1])


