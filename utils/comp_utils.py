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
        compstat(List[Composition]):
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
           for sl_comp,sl_size in zip(comp,sl_sizes):
               if sp_name in sl_comp:
                   sp_num += sl_comp[sp_name]*sl_size   
           sp_frac = float(sp_num)/sum(sl_sizes)
           if not (sp_frac<=ub and sp_frac >=lb):
               return False

    #Sublattice-specific restriction.
    elif isinstance(comp_restrictions,list) and len(comp_restrictions)>=1 \
        and isinstance(comp_restrictions[0],dict):
        for sl_restriction, sl_comp in zip(comp_restrictions,comp):
            for sp in sl_restriction:
                sp_name = get_species(sp)
                if sp_name not in sl_comp:
                    sp_frac = 0
                else:
                    sp_frac = sl_comp[sp_name]
 
                if not (sp_frac<=ub and sp_frac >=lb):
                   return False                   

    return True

# Composition linkage number for Charge neutral semi-grand flip rules
def get_n_links(comp_stat,operations):
    """
    Get the total number of configurations reachable by a single flip in operations
    set.
    comp_stat:
        a list of lists, same as the return value of comp_utils.occu_to_compstat, 
        is a statistics of occupying species on each sublattice.
    operations:
        a list of dictionaries, each representing a charge-conserving, minimal flip
        in the compositional space.
    Output:
        n_links:
            A list of integers, length = 2*len(operations), giving number of possible 
            flips along each operations.
            Even index 2*i : operation i forward direction
            Odd index 2*i+1: operation i reverse direction
    """
    n_links = [0 for i in range(2*len(operations))]

    for op_id,operation in enumerate(operations):
        #Forward direction
        n_forward = 1
        n_to_flip_on_sl = [0 for i in range(len(comp_stat))]
        for sl_id in operation['from']:
            for sp_id in operation['from'][sl_id]:
                n = comp_stat[sl_id][sp_id]
                m = operation['from'][sl_id][sp_id]
                n_forward = n_forward*combinatorial_number(n,m)
                n_to_flip_on_sl[sl_id] += m

        for sl_id in operation['to']:
            for sp_id in operation['to'][sl_id]:
                n = n_to_flip_on_sl[sl_id]
                m = operation['to'][sl_id][sp_id]
                n_forward = n_forward*combinatorial_number(n,m)
                n_to_flip_on_sl[sl_id] -= m

        for n in n_to_flip_on_sl:
            if n!=0:
                raise ValueError("Number of species on both sides of operation can not match!")

        #Reverse direction    
        n_reverse = 1
        n_to_flip_on_sl = [0 for i in range(len(comp_stat))]
        for sl_id in operation['to']:
            for sp_id in operation['to'][sl_id]:
                n = comp_stat[sl_id][sp_id]
                m = operation['to'][sl_id][sp_id]
                n_reverse = n_reverse*combinatorial_number(n,m)
                n_to_flip_on_sl[sl_id] += m

        for sl_id in operation['from']:
            for sp_id in operation['from'][sl_id]:
                n = n_to_flip_on_sl[sl_id]
                m = operation['from'][sl_id][sp_id]
                n_reverse = n_reverse*combinatorial_number(n,m)
                n_to_flip_on_sl[sl_id] -= m

        for n in n_to_flip_on_sl:
            if n!=0:
                raise ValueError("Number of species on both sides of operation can not match!")

        n_links[2*op_id] = n_forward
        n_links[2*op_id+1] = n_reverse

    return n_links
