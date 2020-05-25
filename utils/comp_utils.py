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

def get_bits(structure,sc_size=1,sublat_merge_rule=None):
    """
    Get species occupying each sublattice from a pymatgen.
    Structure object.
    Previous pyabinito used pymatgen.specie order.
    Now using string order for all species, including 'Vac'.
    Note:
        We recommend you to define 'bits' on your own, rather
    than getting in from a pymatgen.structure.
    Inputs:
        structure: pymatgen.structure
        sc_size: supercell size if this structure is a 
                 supercell
        sublat_merge_rule: 
                 rules used to merge sites into sublattice
    """
    if len(structure)%sc_size!=0:
        raise ValueError("Supercell size wrong! Number of sites in structure\
                          can not be divided by super cell size!")
    N_sts_prim = len(structure)//sc_size
    sublat_list = get_sublat_list(N_sts_prim,sc_size=sc_size,\
                  sublat_merge_rule=sublat_merge_rule)

    all_bits = []
    for group in sublat_list:
        bits = []
        site = structure[group[0]]
        bits = [str(sp) for sp in site.species.keys()]
        if site.species.num_atoms < 0.99:
            bits.append("Vac")
        bits = sorted(bits)

        all_bits.append(bits)
    return all_bits

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

def get_all_axis(bits):
    """
    Get all axis in a charge-neutral composition space.
    Each axis represents a charge and site-conserved, elementary flip combination.
    For example:
    'Ca2+ -> Mg2+', and 'Li+ -> Mn2+'+'F- -> O2-'.
    Vacancies 'Vac' are considered as a type of specie with 0 charge, thus in our
    formulation, the system is in number-conserved, semi-grand canonical ensemble.
    Inputs:
        bits: a list yielded by utils.comp_util.get_bits
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

    chg_of_swps = [GetIonChg(p[1])-GetIonChg(p[0]) for p in unit_swps]
    
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
            n1 = chg2//gcd
            n2 = -chg1//gcd
            neutral_combs.append([(swp1,n1),(swp2,n2)])

    operations = []
    for swp_combo in neutral_combs:
        operation = {'to':{},'from':{}}
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
        #adjust the operation dictionary to make all values positive.
        operations.append(operation)

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
            from_name = bits[sl_id][swp_from]
            from_strs.append('{} {}({})'.format(n,from_name,sl_id))
        for (swp_to,sl_id),n in operation['to'] .items():
            to_name = bits[sl_id][swp_to]
            to_strs.append('{} {}({})'.format(n,to_name,sl_id)) 
        from_str = ' + '.join(from_strs)
        to_str = ' + '.join(to_strs)
        operation_strs.append(from_str+' -> '+to_str) 
    return '\n'.join(operation_strs)

def vec_to_comp(vec,init_comp,neutral_combs, bits):
    """
    Turns a CEAuto composition vector into composition dictionary.
    For example, in Ca/MgO: vec = (1.0) -> comp=
    [{'Ca2+':1.0},{'O2-':1.0}]
    The input init_comp should also have the same form as output comp,
    namely a list of dictionaries, each item in list corresponds to 
    the composition on a sub-lattice.
    """
    comp = deepcopy(init_comp)
    for dx,comb in zip(vec,neutral_combs):
        for (swp_to,swp_from,sl_id),n in comb:
            to_name = bits[sl_id][swp_to]
            from_name = bits[sl_id][swp_from]      
            comp[sl_id][from_name]-=dx*n
            comp[sl_id][to_name]+=dx*n
    is_legal_comp = True
    for sl in comp:
        for sp in sl:
            if sl[sp]<0:  
               is_legal_comp = False
               break
    
    if not is_legal_comp:
        raise ValueError('The replacement vector can not be converted into a reachable compostion.')

    else:
        return comp

def comp_to_vec(comp,init_comp, neutral_combs, bits):
    """
        Get the composition vector from a composition vector.
    """
    #flatten the composition into a vector
    n_bits = get_n_bits(bits)

    dcomp_flat = []
    n_sl = len(bits)
    for sl_id in range(n_sl):
        for sp in bits[sl_id][:-1]:
            if sp in comp[sl_id] and sp in init_comp[sl_id]:
                dn = comp[sl_id][sp]-init_comp[sl_id][sp]
            else:
                if sp in init_comp[sl_id]:
                    dn = -init_comp[sl_id][sp]
                elif sp in comp[sl_id]:
                    dn = comp[sl_id][sp]
            dcomp_flat.append(dn)
    #flatten the unitary swappings into basis vectors.
    #print("dcomp_flat:",dcomp_flat)
    Nb = len(dcomp_flat)
    combs_flat = []
    for comb in neutral_combs:
        comb_flat = [0 for i in range(Nb)]
        for (swp_to,swp_from,sl_id),n in comb:
            bit_id = sum([len(n_bits[i])-1 for i in range(sl_id)])+swp_to
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
    N_sts_prim = len(occu)//sc_size
    sublat_list = get_sublat_list(N_sts_prim,sc_size=sc_size,\
                  sublat_merge_rule=sublat_merge_rule)
    for sublat in bits:
        comp.append({})

    for i,sp in enumerate(occu):
        idx = get_sublat_id(i,sublat_list)
        sp_name = bits[idx][sp]
        if sp_name not in comp[idx]:
            comp[idx][sp_name]=1
        else:
            comp[idx][sp_name]+=1
    return comp

def get_flip(bits, N_sts_prim, neutral_combs, occu, sc_size=1,\
             sublat_merge_rule = None):
    """
    Apply, or reverse apply a neutral flip combination.
    Will continue random searching until an available flip is found.
    """
    flip = None
    n_bits = get_n_bits(bits)
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

    for comb_id,comb in enumerate(neutral_combs):
        #When sampling flips, make sure to check that all flips have a equal probability to be sampled!
        #In the previous implementation, I made a mistake. I sampled compositional direction first, then
        #enumerated flips that goes in the chosen direction, therefore adding a prefactor to each flip's
        #probability to be sampled. You can verify that this prefactor ridiculously amplifies flips that
        #goes back to pure state, so you almost can't reach mixed states at all!
        valid_pos_flips_forswps = []
        valid_neg_flips_forswps = []
        for swp,n in comb:
            if n>0:
                swp_to,swp_from,sl_id = swp
                n_swp = n
            elif n<0:
                swp_from,swp_to,sl_id = swp
                n_swp = -n
            else:
                continue
            #We can do this because all the flips should be number conserved.

            valid_pos_flips_forswp= [(swped_sites,swp_to) for swped_sites in \
                         combinations(sl_stats_init[sl_id][swp_from],n_swp)]
            valid_pos_flips_forswps.append(valid_pos_flips_forswp)

            valid_neg_flips_forswp= [(swped_sites,swp_from) for swped_sites in \
                         combinations(sl_stats_init[sl_id][swp_to],n_swp)]
            valid_neg_flips_forswps.append(valid_neg_flips_forswp)

        valid_pos_flips = list(product(*valid_pos_flips_forswps))
        valid_neg_flips = list(product(*valid_neg_flips_forswps))

        #Sorting formats out to del_corr acceptable format
        for flip in valid_pos_flips:
            reformatted_flip = []
            for flipped_sts,flip_to_sp in flip:
                reformatted_flip.extend([(flipped_st,flip_to_sp) for flipped_st in flipped_sts])
            valid_flips.append(reformatted_flip)
            valid_dvecs.append((comb_id,1))
                  
        for flip in valid_neg_flips:
            reformatted_flip = []
            for flipped_sts,flip_to_sp in flip:
                reformatted_flip.extend([(flipped_st,flip_to_sp) for flipped_st in flipped_sts])
            valid_flips.append(reformatted_flip)
            valid_dvecs.append((comb_id,-1))

    chosen_flip_id = random.randint(0,len(valid_flips)-1)
    chosen_flip = valid_flips[chosen_flip_id]

    chosen_comb_id, chosen_direction = valid_dvecs[chosen_flip_id]
    dvec = np.zeros(len(neutral_combs))
    dvec[chosen_comb_id] = chosen_direction
    
    return chosen_flip,dvec

