__author__="Fengyu Xie"

"""
This file is reserved as an integrable part of smol.moca.ensembles.
"""

import numpy as np

from itertools import combinations,product,permutations
import random

from copy import deepcopy

from .utils.enum_utils import *
from .utils.comp_utils import *

####
# Flips enumeration
####
def get_flip_canonical(occu, nbits,\
                       sublat_merge_rule=None,\
                       sc_making_rule='pmg'):
    """
    Find a flip operation to an occupation in canonical ensemble.
    """
    if sublat_merge_rule is not None:
        N_sts_prim = sum([len(sl) for sl in sublat_merge_rule])  
    else:
        N_sts_prim = len(nbits)

    if len(occu)%N_sts_prim != 0:
        raise SCSIZEMISMATCH
    sc_size = len(occu)//N_sts_prim

    sublat_list = get_sublat_list(N_sts_prim,sc_size=sc_size,\
                  sublat_merge_rule=sublat_merge_rule,\
                  sc_making_rule = sc_making_rule)
    n_sls = len(sublat_list)

    valid_combos = []
    for sublat in sublat_list:
        valid_sublat_combos = []
        for i in range(len(sublat)-1):
            for j in range(i,len(sublat)):
                i_sc_id = sublat[i]
                j_sc_id = sublat[j]
                if occu[i_sc_id]!=occu[j_sc_id]:
                    valid_sublat_combos.append((i_sc_id,j_sc_id))
        valid_combos.extend(valid_sublat_combos)
        #print('valid_combos:',valid_combos)

    if len(valid_combos)==0:
        return None
    else:
        st1,st2 = random.choice(valid_combos)
        #Swap
        return [(st1,int(occu[st2])),(st2,int(occu[st1]))]

def get_flip_semigrand(occu, nbits, operations,\
                       sublat_merge_rule = None,\
                       sc_making_rule = 'pmg',\
                       max_n_links=0):
    """
    Find a flip in charge neutral semigrand canonical ensemble.
  
    Inputs:
    occu:
        An occupation array
    operations:
        ce_elements.comp_space.CompSpace.min_flips.
        A dictionary representing all minimal charge-conserved flips
        to choose from.
    nbits:
        A list of integer lists. See get_n_bits in util.comp_utils
    sublat_merge_rule:
        Sublattice merging rule in primitive cell
        A list specifying which sites in a primitive cell should be 
        considered as a same sublattice.
    max_n_link:
        A key parameter in CN-SGMC flip selection.
        In CN-SGMC we call a flip that changes composition as a 
        'grand-canonical flip', a flip that conserves compositon
        as a 'canonical filp '. If number of total possible 
        grand-canonical flip is n:
            1, suppose n<=max_n_link, then we first choose to either
        do a canonical flip (at p=(max_n_link-n)/max_n_link) or a semigrand 
        canonical flip (at p=n/max_n_link). Then choose either a canonical
        flip or a grand canonical flip equally from all possible flips of 
        their type.
            2, suppose n>max_n_link, then we update max_n_link to n,
        and only choose one grand canonical flip from all possible grand
        canonical flips at equal probability. 
 
     Updated or not, the new max_n_link will be passed as an output of this
     function. As you can imagine, as the user navigate across the compostional
     space by flips, this number will finally converge, just as the thermo
     properties.
 
     By a unproved observation, n_links might have maximum on the extremums 
     of the comp space. So we can initalize with max(n_link(extremums))
 
     Return:
        flip: a list of tuples, (site_id_to_flip, flip_to_nbit)
        new_max_n_links: maximum n_links value after comparison with the
                         current node.
    """
    if sublat_merge_rule is not None:
        N_sts_prim = sum([len(sl) for sl in sublat_merge_rule])  
    else:
        N_sts_prim = len(nbits)

    if len(occu)%N_sts_prim != 0:
        raise SCSIZEMISMATCH
    sc_size = len(occu)//N_sts_prim

    sublat_list = get_sublat_list(N_sts_prim,sc_size=sc_size,\
                  sublat_merge_rule=sublat_merge_rule,\
                  sc_making_rule=sc_making_rule)

    n_sls = len(sublat_list)

    #Get a composition statistics list on each sublattice from the 
    #current occupation array
    comp_stat = occu_to_compstat(occu, nbits,\
                                 sublat_merge_rule = sublat_merge_rule,\
                                 sc_making_rule = sc_making_rule)

    #Get the number of reachable configurations by a single semigrand flip
    n_links_current_operations = get_n_links(comp_stat,operations)
    n_links_current = sum(n_links_current_operations)

    if n_links_current > max_n_links:
        #print("A compositional node with higher number of out links other than extremum discovered.")
        n_links_return = n_links_current
    else:
        n_links_return = max_n_links

    p_semiflip = float(n_links_current)/n_links_return
    p_canoflip = 1 - p_semiflip

    if random.random() <= p_canoflip:
        #choose to do a canonical flip
        ca_flip = get_flip_canonical(occu,nbits,\
                                     sublat_merge_rule=sublat_merge_rule,\
                                     sc_making_rule = sc_making_rule)
    
        return ca_flip, n_links_return

    else:
        #choose a random semigrand flip
        #First choose from one of the operation directions:

        chosen_f_id = choose_section_from_partition(n_links_current_operations)
        operation = operations[chosen_f_id//2]
        direction = chosen_f_id%2 # 0 forward, 1 backward.

        sp_stat = occu_to_spstat(occu, nbits,\
                                 sublat_merge_rule=sublat_merge_rule,\
                                 sc_making_rule = sc_making_rule)
        #occu_to_spstat should be implemented in utils.comp_utils like occu_to_compstat

        chosen_sites_flip_from = [[] for sl in nbits]
        chosen_sps_flip_to = [[] for sl in nbits]

        if direction == 0:
            for sl_id in operation['from']:
                for sp_id in operation['from'][sl_id]:
                    m_from = operation['from'][sl_id][sp_id]
                    chosen_sites = random.sample(sp_stat[sl_id][sp_id],m_from)
                    chosen_sites = sorted(chosen_sites)
                    #remove duplicacy
                    chosen_sites_flip_from[sl_id].extend(chosen_sites)

            from_sites_for_sublats = [[i for i in range(len(sl))] for sl in chosen_sites_flip_from]
            for sl_id in operation['to']:
                chosen_sps_flip_to[sl_id] = [None for st in chosen_sites_flip_from[sl_id]]
                for sp_id in operation['to'][sl_id]:
                    m_to = operation['to'][sl_id][sp_id]
                    from_sites = from_sites_for_sublats[sl_id]
                    chosen_sites = random.sample(from_sites,m_to)
                    chosen_sites = sorted(chosen_sites)
                    for st_id in chosen_sites:
                        chosen_sps_flip_to[sl_id][st_id] = sp_id
                        from_sites_for_sublats[sl_id].remove(st_id)

            for sl in from_sites_for_sublats:
                if len(sl)>0:
                    raise ValueError("Flip not number conserved!")

        else:
            for sl_id in operation['to']:
                for sp_id in operation['to'][sl_id]:
                    m_from = operation['to'][sl_id][sp_id]
                    chosen_sites = random.sample(sp_stat[sl_id][sp_id],m_from)
                    chosen_sites = sorted(chosen_sites)
                    #remove duplicacy
                    chosen_sites_flip_from[sl_id].extend(chosen_sites)

            from_sites_for_sublats = [[i for i in range(len(sl))] for sl in chosen_sites_flip_from]
            for sl_id in operation['from']:
                chosen_sps_flip_to[sl_id] = [None for st in chosen_sites_flip_from[sl_id]]
                for sp_id in operation['from'][sl_id]:
                    m_to = operation['from'][sl_id][sp_id]
                    from_sites = from_sites_for_sublats[sl_id]
                    chosen_sites = random.sample(from_sites,m_to)
                    chosen_sites = sorted(chosen_sites)
                    for st_id in chosen_sites:
                        chosen_sps_flip_to[sl_id][st_id] = sp_id
                        from_sites_for_sublats[sl_id].remove(st_id)

            for sl in from_sites_for_sublats:
                if len(sl)>0:
                    raise ValueError("Flip not number conserved!")

        flip_list = []
        for sl_id,sl_sites in enumerate(chosen_sites_flip_from):
            for st_id,site in enumerate(sl_sites):
                flip_list.append((site, chosen_sps_flip_to[sl_id][st_id]))

    return flip_list,n_links_return
