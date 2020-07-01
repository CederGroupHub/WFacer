####
# Composition related tools
####
SCSIZEMISMATCH = \
    ValueError("Length of occupation array can't match primitive cell size!")


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

def occu_to_compstat(occu,nbits,sublat_merge_rule=None,\
                     sc_making_rule='pmg'):
    """
    Turns a digital occupation array into species statistic list, which
    has the same shape as nbits.
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

    compstat = [[0 for i in range(len(sl))] for sl in nbits]
    for site_id,sp_id in enumerate(occu):
        sl_id = get_sublat_id(site_id,sublat_list)
        compstat[sl_id][sp_id]+=1

    return compstat

def occu_to_spstat(occu,nbits,sublat_merge_rule=None,\
                   sc_making_rule='pmg'):
    """
    Turns a digital occupation array into species statistic list, which
    has the same shape as nbits.
    [[site_id_occupied_by_specie for specie in sublattice] for sublattice in nbits]
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

    spstat = [[[] for i in range(len(sl))] for sl in nbits]

    for site_id,sp_id in enumerate(occu):
        sl_id = get_sublat_id(site_id,sublat_list)
        spstat[sl_id][sp_id].append(site_id)

    return spstat
