import random

import numpy as np
import polytope as pc

from itertools import combinations,product

from copy import deepcopy

from monty.json import MSONable
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
CHGBALANCEERROR = ValueError("Charge balance cannot be achieved with these species.")

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

def integerize_vector(v, dim_limiter=7,dtol=1E-5):
    """
    Given vector v, find a vector whose components are co-primal
    integers.
    Inputs:
        dim_limiter: 
            Maximum allowed integer component. Set to 7 by default,
            because maximum possible charge of an ion is 7.
            Must be an integer!
        dtol:
            point-to-line distance tolerance. If the distance from 
            the detected grid point to line r = tv is larger than 
            dtol, then v is not intergerizable.
    Outputs:
        v_int:
            Intergerized v.
    """
    max_comp = np.max(np.abs(v))
    max_c_id = np.argmax(np.abs(v))
    if max_comp == 0:
        raise ValueError("Vector is zero, can not be integerized.")

    v_scaled = np.array(v)/max_comp
    ev = np.array(v)/np.linalg.norm(v)   

    d = len(v)
    all_dim_ids = list(range(d))
    init_branch = np.zeros(d)

    for magnif in range(1,dim_limiter+1):    
        for n_ones in range(d+1):
            for combo in combinations(all_dim_ids,n_ones):
                branch = deepcopy(init_branch)
                branch[list(combo)] = 1
                grid = np.floor(v_scaled*maginf)+branch

                h_vec = grid-np.dot(ev,grid)*ev
                h = np.linalg.norm(h_vec)
                if h<dtol:
                    return np.array(grid,dtype=int64)

    print("Warning: given vector {} can not be integerized!".format(v))
    return None

def edges_from_vertices(vertices,p):
    """
    Find edges from combinations of vertices.
    """
    valid_edges = []
    N_v = len(vertices)
    v_ids = list(range(N_v))
    is_simplex = True
    for i,j in combinations(v_ids,2):
        if not (vertices[i]+vertices[j])/2 in p:
            valid_edges.append((i,j))
        else:
            is_simplex = False

    return valid_edges, is_simplex

def gram_schmidt(A):
    """
    Do Gram-schmidt orthonormalization to row vectors of a, and returns the result.
    If matrix is over-ranked, will remove redundacies automatically.
    Inputs:
        A: array-like
    Returns:
        A_ortho: array-like, orthonormal matrix.
    """
    n,d = A.shape

    if np.allclose(A[0],np.zeros(d)):
        raise ValueError("First row is a zero vector, can not run algorithm.")

    new_rows = [A[0]/np.linalg.norm(A[0])]
    for row in A[1:]:
        new_row = row
        for other_new_row in new_rows:
            new_row = new_row - np.dot(other_new_row,new_row)*other_new_row
        if not np.allclose(new_row,np.zeros(d)):
            new_rows.append( new_row/np.linalg.norm(new_row) )
    return np.vstack(new_rows)

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
def get_comp_space(bits):
    """
        This function generates a compositional space from a bits list, will include vertices of 
        and unit vectors along edges of the charge neutral composition space. 
        Inputs:
            bits: bit list, same as appeared in get_n_bits
        Outputs:
            v_comps: compositions corresponding to vertices
            edges: edges of the compositional space, give in a list of tuples, each refering to 
                   two adjacent vertices on an edge.
    """
    n_bits = get_n_bits(bits)

    unit_swps = []
    unit_n_swps = []
    #facets are sublattice normalization constraints, and variable normalization
    #constraints
    facets = []
    unit_swp_id = 0

    for sl_id,sl_sps in enumerate(bits):
        if len(sl_sps) < 1:
            raise ValueError('Sublattice bits should not be empty.')
        unit_swps.extend([(sp,sl_sps[-1],sl_id) for sp in sl_sps[:-1]])
        unit_n_swps.extend([(sp,n_bits[sl_id][-1],sl_id) for sp in n_bits[sl_id][:-1]])
        facets.append( [unit_swp_id+idx for idx in range(len(sl_sps)-1)] )
        unit_swp_id+=(len(sl_sps)-1)
        #(sp_before,sp_after,sublat_id)

    chg_of_swps = [p[0].oxidation_state-p[1].oxidation_state for p in unit_swps]
    d = len(unit_swps)

    tot_bkgrnd_chg = sum([sl[-1].oxidation_state for sl in bits])
    
    #If charge neutral constraint forms a valid hyperplane, then we should reduce the 
    #dimensionality formed by the compotional space by doing a translation and a hyper-
    #rotation, to make one of the original axis perpendicular to this constraned hyper-
    #plane, and all other axis falls in this hyperplane. Then we find vertices in this
    #hyperplane.
    #We have to reduce dimensionality first, because polytope.Polytope can not deal with 
    #polytopes in subspaces of a higher-dimensional space. It will treat this subspace
    #as an empty set.
    #H representation of a polytope can be written as Ax <= b where A is a n*d matrix, 
    #while x is a d dimensional vector.

    if tot_bkgrnd_chg==0 and np.allclose(chg_of_swps,np.zeros(d)):
        print("Given specie table does not constitute a charged cluster expansion.")
        is_charged = False
        A_rows = []
        b_rows = []
        for constrained_ids in facets:
            row = np.zeros(d)
            row[constrained_ids] = 1
            A_rows.append(row)
            b_rows.append(1)
        A = np.vstack(A_rows)
        b = np.array(b_rows)
        A = np.concatenate((A,-1*np.identity(d)),axis=0)
        b = np.concatenate((b,np.zeros(d)),axis=0)
        p = pc.Polytope(A,b)
        vertices = pc.extreme(p)
        edges = edges_from_vertices(vertices,p)

    elif not np.allclose(chg_of_swps,np.zeros(d)):
        #Charge constraint is valid. Should reduce a dimension
        print("Given specie table constitutes a charged cluster expansion.")
        is_charged = True
        #Choose an axis cross point with the constrained subspace
        #Do transformation x' = R(x-t) to rotate and translate the compositional
        #space into the constrained hyperplane.
        subspc_norm = np.array(chg_of_swps)
        t = None
        for idx, slope in enumerate(subspc_norm):
            if slope!=0 and t is None:
                intercept = float(tot_bkgrnd_chg)/slope
                t = np.zeros(d)
                t[idx]=intercept
        #Get rotation matrix with G-S orthogonalization.(Just concatenate the original basis
        #to en, and do G-S algorithm. Any excessive parts will be removed.
        new_basis = np.vstack((subspc_norm,np.identity(d)))
        new_basis = gram_schmidt(new_basis)
        R = new_basis       
        A_trans = A@R.T
        b_trans = b-A@t
        #Subspace is r_trans = [0,...,...]
        A_sub = A_trans[:,1:]
        b_sub = b_trans
        #In subspace, this polytope has non-zero volume, and therefore can be properly handled.
        p_sub = pc.Polytope(A_sub,b_sub)
        vertices_sub = pc.extreme(p)
        N_v = len(vertices_sub)
        edges = edges_from_vertices(vertices_sub,p)
        vertices_trans = np.hstack((np.zeros(N_v),vertices_sub))
        vertices = vertices_trans@R + t
    else:
        raise CHGBALANCEERROR
    print("Compositonal space vertices:\n",vertices)
    print("Edges:")

    edge_vecs = []
    for edge in edges:  
        #get irreducible unitary flips ("integerized edge vectors")
        limiter = int(np.max(np.abs(chg_of_swps)))
        edge_vec = integerize_vector(vertices[edge[0]]-vertices[edge[1]],dim_limiter=limiter)
        print(edge,' ',edge_vec)
        edge_vecs.append(edge_vec)
    
    #Rewrite edge vectors as specie operations
    operations = []
    for ev in edge_vecs:
        operation = {'from':{},'to':{}}
        #'from' side: annihilate species, 'to' side: generate species
        for f_id,flip_num in enumerate(ev):
            if flip_num>0:
                flip_to,flip_from,sl_id = unit_n_swps[f_id]
            elif flip_num<0:
                flip_from,flip_to,sl_id = unit_n_swps[f_id]
            else:
                continue

            if sl_id not in operation['from']:
                operation['from'][sl_id]={}
            if sl_id not in operation['to']:
                operation['to'][sl_id]={}
            if flip_from not in operation['from'][sl_id]:
                operation['from'][sl_id][flip_from]=flip_num
            if flip_to not in operation['to'][sl_id]:
                operation['to'][sl_id][flip_to]=flip_num

    operations.append(operation)
    return unit_n_swps,vertices,edges,operations


####
# Flip enumerations
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

class 
   
