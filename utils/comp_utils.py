import random

import numpy as np

from itertools import combinations,product

from copy import deepcopy
"""
This file contains functions related to implementing and navigating the 
compositional space.

In CEAuto, we first define a starting, charge-neutral, and fully occupied
('Vac' is also an occupation) composition in the compositional space.

Then we find all possible unitary, charge and number conserving flipping 
combinations of species, and define them as axis in the compositional space.

A composition is then defined as a vector, with each component corresponding
to the number of filps that have to be done on a 'flip axis' in order to get 
the target composition from the defined, starting composition.

For some supercell size, a composition might not be 'reachable' be cause 
supercell_size*atomic_ratioo is not an integer. For this case, you need to 
select a proper enumeration fold for your compositions(see enum_utils.py),
or choose a proper supercell size.
"""
NUMCONERROR = ValueError("Operation error, flipping not number conserved.") 
OUTOFSUBLATERROR = ValueError("Operation error, flipping between different sublattices.")

####
# Basic mathematics
####
def GCD(a,b):
    """ The Euclidean Algorithm """
    a = abs(a)
    b = abs(b)
    while a:
        a, b = b%a, a
    return b    

def concat_tuples(l):
    s = ()
    for sub_tup in l:
        s = s+sub_tup
    return s

def tuple_diff(l1,l2):
    #returns l1-l2
    return tuple([e for e in l1 if e not in l2])

####
# Composition related tools
####

def get_sublat_list(N_sts_prim,sc_size=1,sublat_merge_rule=None,sc_making_rule='pmg'):
    """
    Get site indices in each sublattice.
    Inputs:
        N_sts_prim: 
            number of sites in a prim cell
        sc_size: 
            supercell size. If already primitive cell, should be 1.
        sublat_merge_rule: 
            rules used to merge sites into sublattice. if None, will
            consider each site in a prim cell to be a sublattice.
        sc_making_rule:
            rules of creating supercell from primitive cell. this 
            affects the indexing rule of duplicated prim sites in 
            the supercell. Now only support the rule of pymatgen, which
            duplicates each prim site by sc_size times, then stack the
            duplicative blocks in their intial order as they were in 
            prim cell.
    """
    if sc_making_rule != 'pmg':
        raise NotImplementedError

    if sublat_merge_rule is None:
        sublat_list = [list(range(i*sc_size,(i+1)*sc_size)) \
                       for i in range(N_sts_prim)]
        #This is pymatgen indexing rule.
    else:
        sublat_list = []
        for sublat_sts_in_prim in sublat_merge_rule:
            sublat_sts_in_sc = []
            for sublat_st_in_prim in sublat_sts_in_prim:
                idx = sublat_st_in_prim
                sublat_sts_in_sc.extend(list(range(idx*sc_size,(idx+1)*sc_size)))                   
            sublat_list.append(sublat_sts_in_sc)
    return sublat_list

#    Now 'bits' are represented in utils.specie_utils.CESpecies, and are 
#    Generated in ce_elements.exp_structure.ExpansionStructure

#    bits shall be a 2D list, with the first dimension equals to the number of 
#    sublattices, and the second dimension equals to num of species occupying this
#    sublattice. Each element will be an object of CESpecies.

def get_n_bits(bits):
    """
    Represent occupying species with integers
    """
    return [list(range(len(sublat))) for sublat in bits]

def get_sublat_id(st_id_in_sc,sublat_list):
    for sl_id,group in enumerate(sublat_list):
        if st_id_in_sc in group:
            return sl_id
    return None

def get_init_comp(bits):
    """
    This generates a charge neutral initial composition in the comp space.
    """
    raise NotImplementedError

def get_all_axis(bits):
    """
    Get all axis in a charge-neutral composition space.
    Each axis represents a charge and site-conserved, elementary flip combination.
    For example:
    'Ca2+ -> Mg2+', and 'Li+ -> Mn2+'+'F- -> O2-'.
    Vacancies 'Vac' are considered as a type of specie with 0 charge, thus in our
    formulation, the system is in number-conserved, semi-grand canonical ensemble.
    Inputs:
        bits: a list of CESpecies on each sublattice.
    Outputs:
        neutral_combs:
            a list that stores all charge neutral filps. Species are encoded in their
            n_bit form.
        operations:
            this list also stores flip informations, but usually we don't use it in
            other parts of CEAuto. it's created for easy visualization only.
    """
    n_bits = get_n_bits(bits)

    unit_swps = []
    unit_n_swps = []
    for sl_id,sl_sps in enumerate(bits):
        unit_swps.extend([(sp,sl_sps[-1],sl_id) for sp in sl_sps[:-1]])
        unit_n_swps.extend([(sp,n_bits[sl_id][-1],sl_id) for sp in n_bits[sl_id][:-1]])
        #(sp_before,sp_after,sublat_id)

    chg_of_swps = [p[0].oxidation_state-p[1].oxidation_state for p in unit_swps]
    
    #Dimensionality of the charge neutral space.
    zero_swps = [swp for swp,chg in zip(unit_n_swps,chg_of_swps) if chg==0]
    non_zero_swps = [swp for swp,chg in zip(unit_n_swps,chg_of_swps) if chg!=0]
    non_zero_chgs = [chg for swp,chg in zip(unit_n_swps,chg_of_swps) if chg!=0]
    dim = len(non_zero_swps)-1
    neutral_combs = [[(swp,1)] for swp in zero_swps] #list[[(swp_1,n_swp_1),(swp_2,n_swp_2)],...]
    if dim>0:
        for i in range(dim):
            swp1 = non_zero_swps[i]
            swp2 = non_zero_swps[i+1]
            chg1 = non_zero_chgs[i]
            chg2 = non_zero_chgs[i+1]
            gcd = GCD(chg1,chg2)
            if chg1*chg2>0:
                n1 = chg2//gcd
                n2 = -chg1//gcd
            else:
                n1 = chg2//gcd
                n2 = chg1//gcd

            neutral_combs.append([(swp1,n1),(swp2,n2)])

    operations = []
    for swp_combo in neutral_combs:
        operation = {'to':{},'from':{}}
        if len(swp_combo)==1:
            swp,n = swp_combo[0]
            swp_to,swp_from,sl_id = swp
            operation['from'][(swp_from,sl_id)] = n
            operation['to'][(swp_to,sl_id)] = n
            operations.append(operation)

        else:
            for swp,n in swp_combo:
                if n>0:
                    swp_to,swp_from,sl_id = swp
                    n_swp = n
                elif n<0:
                    swp_from,swp_to,sl_id = swp
                    n_swp = -n
                else:
                    continue            
    
                if (swp_to,sl_id) not in operation['to']:
                    operation['to'][(swp_to,sl_id)] = n_swp
                else:
                    operation['to'][(swp_to,sl_id)] += n_swp
                if (swp_from,sl_id) not in operation['from']:
                    operation['from'][(swp_from,sl_id)] = n_swp
                else:
                    operation['from'][(swp_from,sl_id)] += n_swp
            #deduplicate 'from' and 'to'
            operation_dedup = {'to':{},'from':deepcopy(operation['from'])}

            for sp_to,sl_id in operation['to']:
                if (sp_to,sl_id) in operation_dedup['from']:
                    if operation['from'][(sp_to,sl_id)]>\
                       operation['to'][(sp_to,sl_id)]:
                        operation_dedup['from'][(sp_to,sl_id)]-=\
                        operation['to'][(sp_to,sl_id)]
                    elif operation['from'][(sp_to,sl_id)]<\
                       operation['to'][(sp_to,sl_id)]:
                        operation_dedup['to'][(sp_to,sl_id)]=\
                        operation['to'][(sp_to,sl_id)]-\
                        operation['from'][(sp_to,sl_id)]
                        operation_dedup['from'].pop((sp_to,sl_id))
                    else:
                        operation_dedup['from'].pop((sp_to,sl_id))
                else:
                    operation_dedup['to'][(sp_to,sl_id)]=\
                    operation['to'][(sp_to,sl_id)]
        #adjust the operation dictionary to make all values positive.
            operations.append(operation_dedup)

    return neutral_combs,operations

def visualize_operations(operations,bits):
    """
    This function turns an operation dict into an string for easy visualization,
    """
    operation_strs = []
    for operation in operations:
        from_strs = []
        to_strs = []
        for (swp_from,sl_id),n in operation['from'].items():
            from_name = bits[sl_id][swp_from].specie_string
            from_strs.append('{} {}({})'.format(n,from_name,sl_id))
        for (swp_to,sl_id),n in operation['to'] .items():
            to_name = bits[sl_id][swp_to].specie_string
            to_strs.append('{} {}({})'.format(n,to_name,sl_id)) 
        from_str = ' + '.join(from_strs)
        to_str = ' + '.join(to_strs)
        operation_strs.append(from_str+' -> '+to_str) 
    return '\n'.join(operation_strs)

def vec_to_comp(vec,init_comp,neutral_combs):
    """
    Turns a CEAuto composition vector into a CEAuto composition
    object.
    Init comp: ex. comp=[[1.0],[1.0]]
                   bits = [[CESpecie.from_string('Ca2+')],\
                    [CESpecie.from_string('O2-')]]
                   means:
                   [{'Ca2+':1.0},{'02-':1.0}] in old CEAuto
    Therefore, when decoding a composition, you must combine both
    the comp list, and the bits list.
    """
    comp = deepcopy(init_comp)
    for dx,comb in zip(vec,neutral_combs):
        for (swp_to,swp_from,sl_id),n in comb:
            comp[sl_id][swp_from]-=dx*n
            comp[sl_id][swp_to]+=dx*n

    is_legal_comp = True
    for sl in comp:
        for sp_id,n in enumerate(sl):
            if n<0:  
               is_legal_comp = False
               break
    
    if not is_legal_comp:
        raise ValueError('The replacement vector can not be converted into a reachable compostion.')
    else:
        return comp

def comp_to_vec(comp,init_comp, neutral_combs):
    """
        Get the composition vector from a composition vector.
    """
    #flatten the composition into a vector
    #n_bits = get_n_bits(bits)

    dcomp_flat = []

    for sl_id in range(len(init_comp)):
        for sp_id in range(len(init_comp[sl_id])-1):
            dn = comp[sl_id][sp_id]-init_comp[sl_id][sp_id]
            dcomp_flat.append(dn)
    #flatten the unitary swappings into basis vectors.
    #print("dcomp_flat:",dcomp_flat)
    Nb = len(dcomp_flat)
    combs_flat = []
    for comb in neutral_combs:
        comb_flat = [0 for i in range(Nb)]
        for (swp_to,swp_from,sl_id),n in comb:
            bit_id = sum([len(init_comp[i])-1 for i in range(sl_id)])+swp_to
            comb_flat[bit_id]+=n
        combs_flat.append(comb_flat)

    n = len(combs_flat)
    A = np.zeros((n,n))
    b = np.zeros(n)
    for i in range(n):
        b[i] = float(np.dot(combs_flat[i],dcomp_flat))
        for j in range(n):
            A[i][j] = float(np.dot(combs_flat[i],combs_flat[j]))
    try:
        comp_vec = np.linalg.inv(A)@b
    except:
        raise ValueError("Given composition not reachable by unitary replacements!")
   
    return comp_vec

def occu_to_comp(occu, bits,sc_size=1,sublat_merge_rule=None):
    """
        Get composition values from an occupation state list given by the monte carlo part.
        But in charge-conserved semigrand, the composition will be processed in the program
        as a vertor on the 'unitary swapping basis'
    Inputs:
        occu: len(occu)=num_of_sites, not num_of_sublats
        bits: the bits table given by get_bits
        sc_size: size of the supercell, in interger
    Outputs:
        Comp: form [{'Li+':5,'Ti4+':1},{'O2-':6}], etc. len(Comp)=len(sublat_list)
    """
    comp = []
    
    if len(occu)%sc_size!=0:
        raise ValueError("Supercell size not correct!")

    N_sts_prim = len(occu)//sc_size
    sublat_list = get_sublat_list(N_sts_prim,sc_size=sc_size,\
                  sublat_merge_rule=sublat_merge_rule)
    for sublat in bits:
        comp.append([0 for i in range(len(sublat))])

    for i,sp in enumerate(occu):
        idx = get_sublat_id(i,sublat_list)
        comp[idx][int(sp)]+=1

    return comp

def visualize_comp(comp,bits):
    vis_comp = []
    for sl_id,sl_bts in enumerate(bits):
        vis_comp.append({})
        for b_id,b in enumerate(sl_bts):
            vis_comp[sl_id][b.specie_string]=comp[sl_id][b_id]
    
    return vis_comp
####
# Ensemble related tools
####


def get_flip_canonical(occu, bits, sc_size =1,\
                       sublat_merge_rule=None):
    """
    Find a flip operation to an occupation in canonical ensemble.
    """
    n_bits = get_n_bits(bits)

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
    

def get_flip_semigrand(occu, bits, operations, sc_size=1,\
             sublat_merge_rule = None):
    """
    Find a flip operation to an occupation in charge-neutral semi 
    grand canonical ensemble.
    """
    #requires check!
    flip = None
    n_bits = get_n_bits(bits)

    if len(occu)%sc_size!=0:
        raise ValueError("Supercell size not correct!")

    N_sts_prim = len(occu)//sc_size
    sublat_list = get_sublat_list(N_sts_prim,sc_size=sc_size,\
                  sublat_merge_rule=sublat_merge_rule)    
    n_sls = len(sublat_list)

    sl_sites = sublat_list

    sl_stats_init = [[[] for i in range(len(n_bits[sl_id]))] for sl_id in range(n_sls)]

    for sl_id in range(n_sls):
        for st_id in sl_sites[sl_id]:
            bit = int(occu[st_id])
            sl_stats_init[sl_id][bit].append(st_id)

    sl_nums_init = [[len(sl_sp) for sl_sp in sl] for sl in sl_stats_init]

    #list all possible unitary operations on this occupation, then randomly choose one of them
    #since you have to list all operations, the number of which usually goes in O(N^3), 
    #the usage of a large supercell is highly discouraged!

    valid_flips = []
    valid_dvecs = []

    for op_id,operation in enumerate(operations):
        #When sampling flips, make sure to check that all flips have a equal probability to be sampled!
        #In the previous implementation, I made a mistake. I sampled compositional direction first, then
        #enumerated flips that goes in the chosen direction, therefore adding a prefactor to each flip's
        #probability to be sampled. You can verify that this prefactor ridiculously amplifies flips that
        #goes back to pure state, so you almost can't reach mixed states at all!
        if len(operation['from'])==1 and len(operation['to'])==1:
            #Then this must be a single flip.
            (swp_from,sl_id1),n1 = list(operation['from'].items())[0]
            (swp_to,sl_id2),n2 = list(operation['to'].items())[0]
            if sl_id1!=sl_id2:
                raise OUTOFSUBLATERROR
            if n1 != n2:
                raise NUMCONERROR
            valid_pos_flips = [[(combo,swp_to)] for combo in combinations(sl_stats_init[sl_id1][swp_from],n1)]
            valid_neg_flips = [[(combo,swp_from)] for combo in combinations(sl_stats_init[sl_id1][swp_to],n1)]

        elif len(operation['from'])==2 and len(operation['to'])==2:
            #Then this flip is a combination of two flips on diffrent sublattices
            (swp_from_1,sl_id1),n1 = list(operation['from'].items())[0]
            (swp_from_2,sl_id2),n2 = list(operation['from'].items())[1]
            (swp_to_1,sl_id3),n3 = list(operation['to'].items())[0]
            (swp_to_2,sl_id4),n4 = list(operation['to'].items())[1]
                
            flipped_combos_on_sl1 = list(combinations(sl_stats_init[sl_id1][swp_from_1],n1))
            flipped_combos_on_sl2 = list(combinations(sl_stats_init[sl_id2][swp_from_2],n2))
            flipped_combos_on_sl3 = list(combinations(sl_stats_init[sl_id3][swp_to_1],n3))
            flipped_combos_on_sl4 = list(combinations(sl_stats_init[sl_id4][swp_to_2],n4))
                                           
            if sl_id1 == sl_id3 and sl_id2 == sl_id4:
                if n1!=n3 or n2!=n4:
                    raise NUMCONERROR
                valid_pos_flips = [[(combo1,swp_to_1),(combo2,swp_to_2)] for combo1,combo2 in \
                  product(flipped_combos_on_sl1,flipped_combos_on_sl2)]                   
                valid_neg_flips = [[(combo1,swp_from_1),(combo2,swp_from_2)] for combo1,combo2 in \
                  product(flipped_combos_on_sl3,flipped_combos_on_sl4)] 

            elif sl_id1 == sl_id4 and sl_id2 == sl_id3:
                if n1!=n4 or n2!=n3:
                    raise NUMCONERROR
                valid_pos_flips = [[(combo1,swp_to_2),(combo2,swp_to_1)] for combo1,combo2 in \
                  product(flipped_combos_on_sl1,flipped_combos_on_sl2)]                   
                valid_neg_flips = [[(combo1,swp_from_2),(combo2,swp_from_1)] for combo1,combo2 in \
                  product(flipped_combos_on_sl3,flipped_combos_on_sl4)]

            else:
                raise OUTOFSUBLATERROR

        elif len(operation['from'])==1 and len(operation['to'])==2:
            (swp_from_1,sl_id1),n1 = list(operation['from'].items())[0]
            (swp_to_1,sl_id3),n3 = list(operation['to'].items())[0]
            (swp_to_2,sl_id4),n4 = list(operation['to'].items())[1]
            if sl_id1 != sl_id3 or sl_id1 !=sl_id4:
                raise OUTOFSUBLATERROR
            if n1!=n3+n4:
                raise NUMCONERROR

            flipped_combos_on_sl1 = list(combinations(sl_stats_init[sl_id1][swp_from_1],n1))
            flipped_combos_on_sl3 = list(combinations(sl_stats_init[sl_id3][swp_to_1],n3))
            flipped_combos_on_sl4 = list(combinations(sl_stats_init[sl_id4][swp_to_2],n4))

            valid_pos_flips = []
            for combo in flipped_combos_on_sl1:
                to_1_combs = list(combinations(combo,n3))
                to_2_combs = [tuple_diff(combo,to_1_comb) for to_1_comb in to_1_combs]
                valid_pos_flips.extend([[(to_1_comb,swp_to_1),(to_2_comb,swp_to_2)] \
                                        for to_1_comb,to_2_comb in zip(to_1_combs,to_2_combs)])

            valid_neg_flips = [[(combo3, swp_from_1),(combo4,swp_from_1)] for combo3,combo4 in\
                               zip(flipped_combos_on_sl3,flipped_combos_on_sl4)]

        elif len(operation['to'])==1 and len(operation['from'])==2:
            (swp_to_1,sl_id3),n3 = list(operation['to'].items())[0]
            (swp_from_1,sl_id1),n1 = list(operation['from'].items())[0]
            (swp_from_2,sl_id2),n2 = list(operation['from'].items())[1]
            if sl_id1 != sl_id3 or sl_id2 !=sl_id3:
                raise OUTOFSUBLATERROR
            if n3!=n1+n2:
                raise NUMCONERROR

            flipped_combos_on_sl1 = list(combinations(sl_stats_init[sl_id1][swp_from_1],n1))
            flipped_combos_on_sl2 = list(combinations(sl_stats_init[sl_id2][swp_from_2],n2))
            flipped_combos_on_sl3 = list(combinations(sl_stats_init[sl_id3][swp_to_1],n3))

            valid_neg_flips = []
            for combo in flipped_combos_on_sl3:
                to_1_combs = list(combinations(combo,n1))
                to_2_combs = [tuple_diff(combo,to_1_comb) for to_1_comb in to_1_combs]
                valid_neg_flips.extend([[(to_1_comb,swp_from_1),(to_2_comb,swp_from_2)] \
                                        for to_1_comb,to_2_comb in zip(to_1_combs,to_2_combs)])

            valid_pos_flips = [[(combo1, swp_to_1),(combo2,swp_to_1)] for combo3,combo4 in\
                               zip(flipped_combos_on_sl1,flipped_combos_on_sl2)]

        else:
            raise ValueError("Composition axis not given in standard format!")

        #Sorting formats out to del_corr acceptable format
        for flip in valid_pos_flips:
            reformatted_flip = []
            for flipped_sts,flip_to_sp in flip:
                reformatted_flip.extend([(flipped_st,flip_to_sp) for flipped_st in flipped_sts])
            valid_flips.append(reformatted_flip)
            valid_dvecs.append((op_id,1))
                  
        for flip in valid_neg_flips:
            reformatted_flip = []
            for flipped_sts,flip_to_sp in flip:
                reformatted_flip.extend([(flipped_st,flip_to_sp) for flipped_st in flipped_sts])
            valid_flips.append(reformatted_flip)
            valid_dvecs.append((op_id,-1))

    #print(valid_flips)
    #print(valid_dvecs)

    chosen_flip_id = random.randint(0,len(valid_flips)-1)
    chosen_flip = valid_flips[chosen_flip_id]

    chosen_comb_id, chosen_direction = valid_dvecs[chosen_flip_id]
    dvec = np.zeros(len(operations))
    dvec[chosen_comb_id] = chosen_direction
    
    return chosen_flip,dvec

