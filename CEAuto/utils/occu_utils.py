__author__ = "Fengyu Xie"

"""
Utility functions to handle encoded occupation arrays.
"""
import numpy as np

from smol.cofe import ClusterSubspace
from smol.moca import CEProcessor


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
