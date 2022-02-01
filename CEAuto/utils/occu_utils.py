__author__ = "Fengyu Xie"

"""
Utility functions to handle encoded occupation arrays.
"""
import numpy as np

from smol.cofe import ClusterSubspace
from smol.moca import CEProcessor
from smol.moca.ensemble.sublattice import Sublattice
from smol.cofe.space.domain import get_site_spaces


def get_all_sublattices(processor):
    """Get a list of all sublattices from a processor.

    Will include all sublattices, active or not.

    This is only to be used by the charge neutral ensembles.

    Args:
        processor (Processor):
            A processor object to extract sublattices from.
    Returns:
        list of Sublattice, containing all sites, even
        if only occupied by one specie.
    """
    # Must keep the same order as processor.unique_site_spaces.
    unique_site_spaces = tuple(set(get_site_spaces(
                               processor.cluster_subspace.structure)))

    return [Sublattice(site_space,
            np.array([i for i, sp in enumerate(processor.allowed_species)
                     if sp == list(site_space.keys())]))
            for site_space in unique_site_spaces]


# Wrap up structure_from_occu method from processor module
def structure_from_occu(occu, prim, sc_matrix):
    """Decodes structure from encoded occupation array.

    Args:
        occu(1D arraylike):
            encoded occupation string
        prim(pymatgen.Structure):
            primitive cell containing all occupying species information.
            It is your responisibility to ensure that it is exactly the
            one you used to initialize cluster expansion.
        sc_matrix(3*3 arraylike):
            Supercell matrix. It is your responsibility to check size
            matches the length of occu
    Returns:
        Decoded pymatgen.Structure object.
    """
    dummy_cspace = ClusterSubspace.from_cutoffs(prim, cutoffs={2: 0.01})
    dummy_coefs = np.zeros(dummy_cspace.num_corr_functions)
    dummy_proc = CEProcessor(dummy_cspace, sc_matrix, dummy_coefs)
    return dummy_proc.structure_from_occupancy(occu)


def occu_from_structure(s, prim, sc_matrix):
    """Encodes structure to occupation array.

    Args:
        s(pymatgen.Structure):
            Supercell structure.
        prim(pymatgen.Structure):
            primitive cell containing all occupying species information.
            It is your responisibility to ensure that it is exactly the
            one you used to initialize cluster expansion.
        sc_matrix(3*3 arraylike):
            Supercell matrix. It is your responsibility to check size
            matches the length of occu
    Returns:
        Decoded pymatgen.Structure object.
    """
    dummy_cspace = ClusterSubspace.from_cutoffs(prim, cutoffs={2: 0.01})
    dummy_coefs = np.zeros(dummy_cspace.num_corr_functions)
    dummy_proc = CEProcessor(dummy_cspace, sc_matrix, dummy_coefs)
    return np.array(dummy_proc.occupancy_from_structure(s), dtype=int).tolist()


def occu_to_species_stat(occupancy, all_sublattices, active_only=False):
    """Make compstat format from occupation array.

    Get a statistics table of each specie on sublattices from an encoded
    occupancy array.
    Args:
        occupancy(np.ndarray like):
            An array representing encoded occupancy, can be list.
        all_sublattices(smol.moca.Sublattice):
            All sublattices in the super cell, regardless of activeness.
        active_only(Boolean):
            If true, will count un-restricted sites only. Default to false.

    Return:
        species_stat(2D list of ints/floats)
            Is a statistics of number of species on each sublattice.
            1st dimension: sublattices
            2nd dimension: number of each specie on that specific sublattice.
            Dimensions same as moca.sampler.mcushers.CorrelatedUsher.bits.
    """
    occu = np.array(occupancy, dtype=int)

    return [[int(round((occu[s.active_sites if active_only else s.sites]
                        == sp_id).sum()))
            for sp_id in range(len(s.species))]
            for s in all_sublattices]

