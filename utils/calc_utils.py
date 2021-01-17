"""
Utils that helps to calculate a few properties.
"""
__author__ = 'Fengyu Xie'

import numpy as np
from copy import deepcopy

from pymatgen import Structure
from pymatgen.analysis.ewald import EwaldSummation

from smol.cofe.configspace.domain import get_specie,Vacancy
from .format_utils import structure_from_occu

#Ewald energy from occu
def get_ewald_from_occu(occu,prim,sc_mat):
    """
    Calculate ewald energies from an ENCODED occupation array.
    Args:
        occu(Arraylike of integers): 
            Encoded occupation array
        prim(pymatgen.Structure):
            Primitive cell structure. Must contain all
            species information, if species are decorated.
        sc_mat(2D arraylike of integers):
            Supercell matrix.
    Output:
        Float. Ewald energy of the input occupation.
    """
    supercell_decode = structure_from_occu(occu,prim,sc_matrix)

    return EwaldSummation(supercell_decode).total_energy
