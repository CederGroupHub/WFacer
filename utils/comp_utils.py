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
                sublat_sts_in_sc.extend(list(range(idx*sc_size,(idx+1)*sc_size)))                   sublat_list.append(sublat_sts_in_sc)
    return sublat_list

def get_bits(structure,sc_size=1,sublat_merge_rule=None):
    """
    Get species occupying each sublattice from a pymatgen.
    Structure object.
    Previous pyabinito used pymatgen.specie order.
    Now using string order for all species, including 'Vac'.
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
    sublat_list = get_sublat_list(N_sts_prim,sc_size,sublat_merge_rule)

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

