"""
Utils that helps to calculate a few properties.
"""
__author__ = 'Fengyu Xie'

import numpy as np
from copy import deepcopy

from pymatgen import Structure
from pymatgen.analysis.ewald import EwaldSummation
from pymatgen.analysis.phase_diagram import PhaseDiagram,PDEntry

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

## Adapted from smol.cofe.wrangler
def weights_energy_above_composition(structures, energies, temperature=2000):
    """Compute weights for energy above the minimum reduced composition energy.

    Args:
        structures (list):
            list of pymatgen.Structures
        energies (ndarray):
            energies of corresponding structures.
        temperature (float):
            temperature to used in boltzmann weight

    Returns: weights for each structure.
        array
    """
    e_above_comp = _energies_above_composition(structures, energies)
    return np.exp(-e_above_comp / (kB * temperature))

def weights_energy_above_hull(structures, energies, cs_structure,
                              temperature=2000):
    """Compute weights for structure energy above the hull of given structures.

    Args:
        structures (list):
            list of pymatgen.Structures
        energies (ndarray):
            energies of corresponding structures.
        cs_structure (Structure):
            The pymatgen.Structure used to define the cluster subspace
        temperature (float):
            temperature to used in boltzmann weight.

    Returns: weights for each structure.
        array
    """
    e_above_hull = _energies_above_hull(structures, energies, cs_structure)
    return np.exp(-e_above_hull / (kB * temperature))

def _energies_above_hull(structures, energies, ce_structure):
    """Compute energies above hull.

    Hull is constructed from phase diagram of the given structures.
    """
    pd = _pd(structures, energies, ce_structure)
    e_above_hull = []
    for s, e in zip(structures, energies):
        entry = PDEntry(s.composition.element_composition, e)
        e_above_hull.append(pd.get_e_above_hull(entry))
    return np.array(e_above_hull)


def _pd(structures, energies, cs_structure):
    """Generate a phase diagram with the structures and energies."""
    entries = []

    for s, e in zip(structures, energies):
        entries.append(PDEntry(s.composition.element_composition, e))

    max_e = max(entries, key=lambda e: e.energy_per_atom).energy_per_atom
    max_e += 1000
    for el in cs_structure.composition.keys():
        entry = PDEntry(Composition({el: 1}).element_composition, max_e)
        entries.append(entry)

    return PhaseDiagram(entries)

def _energies_above_composition(structures, energies):
    """Compute structure energies above reduced composition."""
    min_e = defaultdict(lambda: np.inf)
    for s, e in zip(structures, energies):
        comp = s.composition.reduced_composition
        if e / len(s) < min_e[comp]:
            min_e[comp] = e / len(s)
    e_above = []
    for s, e in zip(structures, energies):
        comp = s.composition.reduced_composition
        e_above.append(e / len(s) - min_e[comp])
    return np.array(e_above)
