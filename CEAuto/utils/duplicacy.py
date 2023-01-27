"""Check duplicacy between structures."""

from pymatgen.core import Structure, Element, Species, DummySpecies
from pymatgen.analysis.structure_matcher import StructureMatcher

from smol.cofe.space.domain import Vacancy


def clean_up_decoration(s):
    """Remove all decoration from a structure.

    Used before comparing two structures before sending to compute.
    Args:
        s(Structure):
            A structure.
    Returns:
        Structure:
            The cleaned up structure containing only Element.
    """
    def get_element(p):
        if isinstance(p, Species):
            return p.element
        elif isinstance(p, Element):
            return p
        elif isinstance(p, DummySpecies):
            return DummySpecies(p.symbol)
        else:
            raise ValueError(f"Instance {p} must be a Species,"
                             f" Element or DummySpecies!")

    elements = []
    frac_coords = []
    for p, x in zip(s.species, s.frac_coords):
        if isinstance(p, Vacancy):
            continue
        else:
            elements.append(get_element(p))
            frac_coords.append(x)

    return Structure(s.lattice, elements, frac_coords)


def is_duplicate(s1, s2, remove_decorations=False, matcher=None):
    """Check duplication between structures.

    Args:
        s1(Structure):
            A structure to be checked.
        s2(Structure):
            Same as s1.
        remove_decorations(bool): optional
            Whether or not to remove all decorations from species (i.e,
            charge and other properties). Default to false.
        matcher(StructureMatcher): optional
            A StructureMatcher to compare two structures. Using the same
            _site_matcher as cluster_subspace is highly recommended.
    Returns:
        bool
    """
    matcher = matcher or StructureMatcher()
    if not remove_decorations:
        return matcher.fit(s1, s2)
    else:
        s1_clean = clean_up_decoration(s1)
        s2_clean = clean_up_decoration(s2)
        return matcher.fit(s1_clean, s2_clean)
