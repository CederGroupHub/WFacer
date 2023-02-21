import numpy.testing as npt
from pymatgen.core import DummySpecies, Element, Lattice, Structure

from AceCE.utils.duplicacy import clean_up_decoration, is_duplicate


def test_remove_decorations(data_wrangler):
    for s in data_wrangler.structures:
        s_clean = clean_up_decoration(s)
        assert s_clean.charge == 0
        npt.assert_array_almost_equal(s.lattice.matrix, s_clean.lattice.matrix)
        npt.assert_array_almost_equal(s.frac_coords, s_clean.frac_coords)
        assert s.composition.element_composition.almost_equals(
            s_clean.composition.element_composition
        )
        for site1, site2 in zip(s_clean, s):
            # Only a site without disordering can have attribute "specie".
            assert isinstance(site1.specie, (Element, DummySpecies))
            assert site1.specie.symbol == site2.specie.symbol


def test_duplicate():
    s1 = Structure(Lattice.cubic(3.0), ["Li", "Li"], [[0, 0, 0], [0.5, 0.5, 0.5]])
    s2 = Structure(Lattice.cubic(3.0), ["Li+", "Li-"], [[0, 0, 0], [0.5, 0.5, 0.5]])
    s3 = Structure(Lattice.cubic(3.0), ["H+", "Li-"], [[0, 0, 0], [0.5, 0.5, 0.5]])

    assert is_duplicate(s1, s2, remove_decorations=True)
    assert not is_duplicate(s1, s2, remove_decorations=False)
    assert not is_duplicate(s2, s3, remove_decorations=True)
    assert not is_duplicate(s2, s3, remove_decorations=False)
