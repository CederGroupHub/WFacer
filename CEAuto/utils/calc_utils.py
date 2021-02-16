"""
Utils that helps to calculate a few properties.
"""
__author__ = 'Fengyu Xie'

from pymatgen.analysis.ewald import EwaldSummation
from pymatgen import Element

from smol.cofe.space.domain import Vacancy

from .occu_utils import structure_from_occu

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
    supercell_decode = structure_from_occu(occu,prim,sc_mat)

    all_elements = True
    for s in supercell_decode:
        if not isinstance(s.specie,(Vacancy,Element)):
            all_elements = False
            break

    if all_elements:
        return 0

    return EwaldSummation(supercell_decode).total_energy
