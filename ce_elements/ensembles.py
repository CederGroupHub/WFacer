import numpy as np

from itertools import combinations,product,permutations

from copy import deepcopy

this_file_path = os.path.abspath(__file__)
this_file_dir = os.dirname(this_file_path)
parent_dir = os.dirname(this_file_dir)
sys.path.append(parent_dir)

from utils.enum_utils import *
from utils.specie_utils import *
from utils.comp_utils import *

####
# Flips enumeration
####
def get_n_links(occu):
    raise NotImplementedError


def get_flip_canonical(occu, sc_size =1,\
                       sublat_merge_rule=None):
    """
    Find a flip operation to an occupation in canonical ensemble.
    """

    if len(occu)%sc_size!=0:
        raise ValueError("Supercell size not correct!")

    N_sts_prim = len(occu)//sc_size
    sublat_list = get_sublat_list(N_sts_prim,sc_size=sc_size,\
                  sublat_merge_rule=sublat_merge_rule)
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

def get_flip_semigrand(occu, operations, nbits, \
                       sc_size=1, sublat_merge_rules = None,\
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
   sc_size:
       supercell size of the current configuration
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
       new_max_n_links: ... 
   """
    if len(occu)%sc_size!=0:
        raise ValueError("Supercell size not correct!")

    N_sts_prim = len(occu)//sc_size
    sublat_list = get_sublat_list(N_sts_prim,sc_size=sc_size,\
                  sublat_merge_rule=sublat_merge_rule)
    n_sls = len(sublat_list)

    #Get a composition statistics list on each sublattice from the 
    #current occupation array
    comp_stat = occu_to_compstat(occu, nbits, \
                                 sublat_merge_rule=sublat_merge_rule,\
                                 sc_size = sc_size,\
                                 sc_making_rule = sc_making_rule)
