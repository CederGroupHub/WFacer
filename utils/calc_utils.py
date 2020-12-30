"""
Utils that helps to calculate a few properties.
"""
__author__ = 'Fengyu Xie'

import numpy as np
from copy import deepcopy

from pymatgen import Structure
from pymatgen.analysis.ewald import EwaldSummation

from smol.cofe.configspace.domain import get_specie,Vacancy

#Ewald energy from occu
def get_ewald_from_occu(occu,sc_sublat_list,bits,prim,sc_mat):
    """
    Calculate ewald energies from an ENCODED occupation array.
    Inputs:
        occu(Arraylike of integers): 
            Encoded occupation array
        sc_sublat_list(List of List of integers):
            List of sites in sublattices in a supercell.
        bits(List of List of str or Species/Vacancy/DummySpecie):
            Same as the bits attibute in StructureEnumerator class.
        prim(pymatgen.Structure):
            Primitive cell structure
        sc_mat(2D arraylike of integers):
            Supercell matrix.
    Output:
        Float. Ewald energy of the input occupation.
    """
    scs = int(round(abs(np.linalg.det(mat))))
    if len(occu) != len(prim)*scs:
        raise ValueError("Supercell size mismatch with occupation array!")

    occu_decode = []
    occu_filter = []
    for s_id,sp_id in enumerate(occu):
        sl_id = None
        for i, sl in enuemrate(sc_sublat_list):
            if s_id in sl:
                sl_id = s_id
                break
        if sl_id is None:
            raise ValueError("Site id: {} not found in supercell sublattice list: {}!"\
                             .format(s_id,sc_sublat_list))
        sp_decode = get_specie(bits[sl_id][sp_id])
        #Remove Vacancy to avoid pymatgen incompatibility
        if not isinstance(sp_decode,Vacancy):
            occu_decode.append(sp_decode)
            occu_filter.append(s_id)
        

    supercell = deepcopy(prim).make_supercell(sc_mat)
    lat = supercell.lattice
    frac_coords = supercell.frac_coords

    supercell_decode = Structure(lat,occu_decode,frac_coords[occu_filter])

    return EwaldSummation(supercell_decode).total_energy
