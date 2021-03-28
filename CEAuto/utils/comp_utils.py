__author__ = 'Fengyu Xie'

"""
Compositional space utilities function. Also includes functions that deals with table filps in charge neutral compspace.
"""

from .math_utils import combinatorial_number
from smol.cofe.space.domain import get_species

# Check compositional restrictions
def check_comp_restriction(comp,sl_sizes,comp_restrictions=None):
    """
    Check whether a composition satisfies compositional limits.
    Args:
        comp(List[Composition]):
            Normalized composition on each sublattice. Same as in compspace.py.
        comp_restrictions(List[dict]|Dict):

            Restriction on certain species.
            If this is a dictionary, this dictionary provide constraint of the 
            atomic fraction of specified species in the whole structure;
            If this is a List of dictionary, then each dictionary provides
            atomic fraction constraints on each sublattice.

            For each dictionary, the keys should be Specie/Vacancy object
            or String repr of a specie (anything readable by get_specie() in
            smol.cofe.configspace.domain). If you decorate your expansion species
            with some property other than charge, you should use the latter as key.
            And all the values shall be tuples consisting of 2 float numbers, 
            in the form of (lb,ub). 
      
            Example:
               [{Specie('Li+'):(0.1,0.9),...},...]
            or:
               [{'Li+':(0.1,0.9),...},...]              

            lb constrains the lower bound of atomic fraction, while ub constrains
            the upperbound. lb <= x <= ub.

            You may need to specify this in phases with high concentration of 
            vancancy, so you structure does not collapse.
            
            By default, is None (no constraint is applied.)
    Returns:
        Boolean.
    """
    if comp_restrictions is None:
        return True

    if isinstance(comp_restrictions,dict):
        for sp,(lb,ub) in comp_restrictions.items():
            sp_name = get_species(sp)
            sp_num = 0
            all_sl_size = 0
            for sl_comp,sl_size in zip(comp,sl_sizes):
                if sp_name in sl_comp:
                    sp_num += sl_comp[sp_name]*sl_size   
                    all_sl_size += sl_size
            sp_frac = float(sp_num)/all_sl_size
            if not (sp_frac<=ub and sp_frac >=lb):
                return False

    #Sublattice-specific restriction.
    elif isinstance(comp_restrictions,list) and len(comp_restrictions)>=1 \
        and isinstance(comp_restrictions[0],dict):
        for sl_restriction, sl_comp in zip(comp_restrictions,comp):
            for sp,(lb,ub) in sl_restriction.items():
                sp_name = get_species(sp)
                if sp_name not in sl_comp:
                    sp_frac = 0
                else:
                    sp_frac = sl_comp[sp_name]
 
                if not (sp_frac<=ub and sp_frac >=lb):
                    return False                   

    return True


# measure size of the config space.
def get_Noccus_of_compstat(compstat,scale_by=1):
    """
    Get number of possible occupancies in a supercell with
    a certain composition. Used to reweight samples in
    the compositional space.
    Args:
        compstat(List[List[float]]):
            Number of species on each sublattice, recorded
            in a 2D list. See CompSpace documentation for
            detail.
        scale_by(int):
            Since the provided compstat is usually normalize
            d by supercell size, we often have to scale it
            back by the supercell size before using this
            function. If the scaled compstat has values that
            can not be rounded to an integer, that means 
            the current supercell size can not host the 
            composition, and will raise an error.
    Returns:
        int, number of all possible occupancy arrays.
    """
    int_comp = scale_compstat(compstat,by=scale_by)

    noccus = 1
    for sl_int_comp in int_comp:
        N_sl = sum(sl_int_comp)
        for n_sp in sl_int_comp:
            noccus = noccus*combinatorial_number(N_sl,n_sp)
            N_sl = N_sl - n_sp

    return noccus

# Scale normalized compstat back to integer
def scale_compstat(compstat,by=1):
    """
    Scale compositonal statistics into integer table.
    Args:
        compstat(List[List[float]]):
            Number of species on each sublattice, recorded
            in a 2D list. See CompSpace documentation for
            detail.
        scale_by(int):
            Since the provided compstat is usually normalize
            d by supercell size, we often have to scale it
            back by the supercell size before using this
            function. If the scaled compstat has values that
            can not be rounded to an integer, that means 
            the current supercell size can not host the 
            composition, and will raise an error.
    Returns:
        scaled compstat, all composed of ints.  
    """
    int_comp = []
    for sl_comp in compstat:
        sl_int_comp = []
        for n_sp in sl_comp:
            n_sp_int = int(round(n_sp*by))
            if abs(n_sp*by-n_sp_int) > 1E-3:
                raise ValueError("Composition can't be rounded after scale by {}!".format(by))

            sl_int_comp.append(n_sp_int)
        int_comp.append(sl_int_comp)   

    return int_comp

def normalize_compstat(compstat, sc_size=1):
    return [[float(n)/sc_size for n in sl] for sl in compstat]
