"""Check duplicacy between structures."""
import numpy as np
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core import DummySpecies, Element, Species, Structure
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
            raise ValueError(
                f"Instance {p} must be a Species," f" Element or DummySpecies!"
            )

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
    # Must attempt primitive cell, and should not skip reduction,
    # otherwise elemental structures with different supercell will
    # not be considered as the same!
    if not remove_decorations:
        return matcher.fit(s1, s2)
    else:
        s1_clean = clean_up_decoration(s1)
        s2_clean = clean_up_decoration(s2)
        return matcher.fit(s1_clean, s2_clean)


def is_corr_duplicate(s1, proc1, s2=None, proc2=None, features2=None):
    """Check whether two structures have the same correlations.

    Note: This is to mostly used criteria for checking structure
    duplicacy, because two structures with the same correlation
    vector should typically not be included in the training set
    together! Also, comparing correlation vectors should be much
    faster that comparing two structures, because comparing two
    structures might involve reducing them to primitive cells
    in advance, which can occasionally be very slow.

    Args:
        s1 (Structure):
           A structure to be checked.
        proc1 (CompositeProcessor):
           A processor established with the super-cell matrix of s1.
           (Must be ClusterExpansionProcessor rather than
           ClusterDecompositionProcessor!)
        s2 (Structure): optional
           Same as s1, but if a feature vector is already given,
           no need to give s2.
        proc2 (CompositeProcessor): optional
           Same as proc1. But if a feature vector is already given,
           no need to give.
        features2 (1D arrayLike): optional
           The feature vector of s2. If not given, must give both s2
           and proc2.
    """
    occu1 = proc1.occupancy_from_structure(s1)
    features1 = proc1.compute_feature_vector(occu1) / proc1.size

    if (s2 is None or proc2 is None) and features2 is None:
        raise ValueError("Must either give both s2 and proc2," " or give feature2.")
    if features2 is None:
        occu2 = proc2.occupancy_from_structure(s2)
        features2 = proc2.compute_feature_vector(occu2) / proc2.size

    return np.allclose(features1, features2)
