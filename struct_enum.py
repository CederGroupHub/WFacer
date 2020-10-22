__author__="Fengyu Xie"

"""
This module implements a StructureEnumerator class for CE sampling.
"""
import warnings
import random
from copy import deepcopy
import numpy as np

from monty.json import MSONable

from pymatgen import Structure,Lattice,Element
from pymatgen.analysis.ewald import EwaldSummation
from pymatgen.analysis.structure_matcher import StructureMatcher

from smol.cofe.extern.ewald import EwaldTerm
from smol.cofe.configspace.clusterspace import ClusterSubspace
from smol.cofe.configspace.domain import get_allowed_species,get_specie, Vacancy
from smol.cofe.expansion import ClusterExpansion
from smol.moca import CanonicalEnsemble,Sampler


from .comp_space import CompSpace
from .utils import *

def _is_proper_sc(sc_matrix,lat,max_cond=8,min_angle=30):
    """
    Assess the skewness of a given supercell matrix. If the skewness is 
    too high, then this matrix will be dropped.
    Inputs:
        sc_matrix(Arraylike):
            Supercell matrix
        lat(pymatgen.Lattice):
            Lattice vectors of a primitive cell
        max_cond(float):
            Maximum conditional number allowed of the supercell lattice
            matrix. By default set to 8, to prevent overstretching in one
            direction
        min_angle(float):
            Minmum allowed angle of the supercell lattice. By default set
            to 30, to prevent over-skewing.
    Output:
       Boolean
    """
    newmat = np.matmul(lat.matrix, sc_matrix)
    newlat = Lattice(newmat)
    angles = [newlat.alpha,newlat.beta,newlat.gamma,\
              180-newlat.alpha,180-newlat.beta,180-newlat.gamma]

    return abs(np.linalg.cond(newmat))<=max_cond and \
           min(angles)>min_angle

def _enumerate_matrices(max_det, lat,\
                           transmat=[[1,0,0],[0,1,0],[0,0,1]],\
                           max_sc_cond = 8,\
                           min_sc_angle = 30,\
                           n_select=20):
    """
    Enumerate proper matrices with maximum det up to a number.
    4 steps are used in the size enumeration.
    Inputs:
        max_det(int):
            Maximum allowed determinant size of enumerated supercell
            matrices
        lat(pymatgen.Lattice):
            Lattice vectors of a primitive cell
        transmat(2D arraylike):
            Symmetrizaiton matrix to apply on the primitive cell.
        max_cond(float):
            Maximum conditional number allowed of the supercell lattice
            matrix. By default set to 8, to prevent overstretching in one
            direction
        min_angle(float):
            Minmum allowed angle of the supercell lattice. By default set
            to 30, to prevent over-skewing.
        n_select(int):
            Number of supercell matrices to select.
    Outputs:
        List of 2D lists.
    """
    scs=[]

    if max_det>=4:
        for det in range(max_det//4, max_det//4*4+1, max_det//4):
            scs.extend(Get_diag_matrices(det))
    else:
        for det in range(1,max_det+1):
            scs.extend(Get_diag_matrices(det))       

    scs = [np.matmul(sc,transmat,dtype=np.int64).tolist() for sc in scs \
           if _is_proper_sc(np.matmul(sc,transmat), lat,\
                            max_sc_cond = max_sc_cond,\
                            min_sc_angle = min_sc_angle)]

    ns = min(n_select,len(scs))

    selected_scs = random.sample(scs,ns)

    return selected_scs

def _get_ewald_from_occu(occu,sc_sublat_list,bits,prim,sc_mat):
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
        if sp_decode != Vacancy():
            occu_decode.append(sp_decode)
            occu_filter.append(s_id)
        

    supercell = deepcopy(prim).make_supercell(sc_mat)
    lat = supercell.lattice
    frac_coords = supercell.frac_coords

    supercell_decode = Structure(lat,occu_decode,frac_coords[occu_filter])

    return EwaldSummation(supercell_decode).total_energy

def _CUR_decompose(G, C, R):
    """ calcualate U s.t. G = CUR """
    
    C_inv = np.linalg.pinv(C)
    R_inv = np.linalg.pinv(R)
    U = np.dot(np.dot(C_inv, G), R_inv)
    
    return U

def _select_rows(femat,n_select=10,old_femat=[],method='CUR',keep=[]):

    """
    Selecting a certain number of rows that recovers maximum amount of kernel
    information from a matrix, or given an old feature matrix, select a certain
    number of rows that maximizes information gain from a new feature matrix.

    Inputs: 
        femat(2D arraylike):
            New feature matrix to select from
        n_select(int):
            Number of new rows to select
        old_femat(2D arraylike):
            Old feature matrix to compare with
        method(str):
            Row selection method.
            'CUR'(default):
                select by recovery of CUR score
            'random':
                select at full random
        keep(List of ints):
            indices of rows that will always be selected. By default, no row
            has that priviledge.
    Outputs:
        List of ints. Indices of selected rows in femat.

    """
    A = np.array(femat)
    n_pool,d = A.shape
    domain = np.eye(d)
    # Using an identity matrix as domain matrix
    
    if len(keep) > n_pool:
        raise ValueError("Rows to keep more than rows you have!")      
    n_add = max(min(n_select-len(keep),n_pool-len(keep)),0)

    trial_indices = deepcopy(keep)
    total_indices = [i for i in range(n_pool) if i not in keep]

    if len(old_femat) == 0: #init mode

        G = A@A.T

        for i in range(n_add):
            err = 1e8
            return_index = None

            if method == 'CUR':             
                for i in range(100):  
                #Try one row each time, optimize for 100 iterations
                    trial_index = random.choice(total_indices)
                    trial_indices_current = trial_indices+[trial_index]
                    
                    C = G[:, trial_indices_current]
                    R = G[trial_indices_current,:]
        
                    U = _CUR_decompose(G, C, R)
        
                    err_trial = np.linalg.norm(G - np.dot(np.dot(C, U),R))
        
                    if err_trial < err:
                        return_index = trial_index
                        err = err_trial

            elif method = 'random':
                return_index = random.choice(total_indices)
 
            else:
                raise NotImplementedError

            trial_indices.append(return_index)
            total_indices.remove(return_index)

    else: #add mode
        
        old_A = np.array(old_femat)        
        old_cov = old_A.T @ old_A
        old_inv = np.linalg.pinv(old_cov)
        # Used Penrose-Moore inverse

        reduction = np.zeros(n_add)

        if method == 'CUR':
            for i_id, i in enumerate(total_indices):
                trial_A = np.concatenate((old_A, A[i].reshape(1, d)), axis =0)

                trial_cov = trial_A.T @ trial_A
                trial_inv = np.linalg.pinv(trial_cov)

                reduction[i_id] = np.sum(np.multiply( (trial_inv-old_inv), domain))
                
            add_indices = [total_indices[iid] for iid in np.argsort(reduction)[:n_add]]

        elif method == 'random':
            add_indices = sorted(random.sample(total_indices,n_add))

        else:
            raise NotImplementedError

        trial_indices = trial_indices + add_indices
        total_indices = [i for i in total_indices if i not in add_indices]

    trial_indices = sorted(trial_indices)
    total_indices = sorted(total_indices)
               
    return trial_indices
    
def _flatten_2d(2d_lst,remove_nan=True):
    """
    Sequentially flatten any 2D list into 1D, and gives a deflatten rule.
    Inputs:
        2d_lst(List of list):
            A 2D list to flatten
        remove_nan(Boolean):
            If true, will disregard all None values in 2d_lst when compressing.
            Default is True
    Outputs:
        flat(List):
            Flatten list
        deflat_rule(List of int):
            Deflatten rule. Each integer speciefies the length of each sub-list
            in the original 2D list. When deflattening, will serially read and
            unpack to each sublist based on these numbers.
    """
    deflat_rule = []
    flat = []
    for sl in 2d_lst:
        if remove_nan:
            sl_return = [item for item in sl if item is not None]
        else:
            sl_return = sl
        flat.extend(sl_return)
        deflat_rule.append(len(sl_return))

    return flat,deflat_rule

def _deflat_2d(flat_lst,deflat_rule,remove_nan=True):
    """
    Sequentially decompress a 1D list into a 2D list based on rule.
    Inputs:
        flat(List):
            Flatten list
        deflat_rule(List of int):
            Deflatten rule. Each integer speciefies the length of each sub-list
            in the original 2D list. When deflattening, will serially read and
            unpack to each sublist based on these numbers.
        remove_nan(Boolean):
            If true, will first deflat on deflat rule, then remove all 
            None values in resulting 2d_lst.
            Default is True.

    Outputs:
        2d_lst(List of list):
            Deflatten 2D list
    """
    if sum(deflat_rule)!= len(flat_lst):
        raise ValueError("Deflatten length does not match original length!")

    2d_lst = []
    it = 0
    for sl_len in deflat_rule:
        sl = []
        for i in range(it,it+sl_len):
            if flat_lst[i] is None and remove_nan:
                continue
            sl.append(flat_lst[i])
        2d_lst.append(sl)
        it = it + sl_len
    return 2d_lst

class StructureEnumerator(MSONable):
    """
    Attributes:
        prim(Structure):
            primitive cell of the structure to do cluster expansion on.

        sublat_list(List of lists):
            Stores primitive cell indices of sites in the same sublattices
            If none, sublattices will be automatically generated.

        previous_ce(ClusterExpansion):
            A cluster expansion containing information of cluster expansion
            in previously enumerated structures. Used when doing mc sampling.

        previous_fe_mat(2D Arraylike):
            Feature matrices of previously generated structures. By default,
            is an empty list, which means no previous structure has been 
            generated. 

        transmat(3*3 arraylike):
            A transformation matrix to apply to the primitive cell before
            enumerating supercell shapes. This can help to increase the 
            symmetry of the enumerated supercell. For example, for a rocksalt
            primitive cell, you can use [[1,-1,-1],[-1,1,-1],[-1,-1,1]] as a
            transmat to modify the primitive cell as cubic.

        max_natoms(Int): 
            maximum number of atoms allowed in each enumerated structure.
            By default, set to 200, to restrict DFT computation cost.
            Currently values over 200 are not recommended!

        max_sc_cond(float):
            Maximum allowed lattice matrix conditional number of the enumerated 
            supercells.

        min_sc_angle(float):
            Minumum allowed lattice angle of the enumerated supercells.
 
        max_sc_cond and min_sc_angle controls the skewness of a supercell, so you
        can avoid potential structural instability during DFT structural relaxation.

        comp_restrictions(Dict or List of Dict or None):
            Restriction on certain species.
            If this is a dictionary, this dictionary provide constraint of the 
            atomic fraction of specified species in the whole structure;
            If this is a List of dictionary, then each dictionary provides
            atomic fraction constraints on each sublattice.

            For each dictionary, the keys should be Specie/Vacancy object
            or String repr of a specie (anything readable by get_specie() in
            smol.cofe.configspace.domain). And the values shall be tuples 
            consisting of 2 float numbers, in the form of (lb,ub). 
            lb constrains the lower bound of atomic fraction, while ub constrains
            the upperbound. lb <= x <= ub.

            You may need to specify this in phases with high concentration of 
            vancancy, so you structure does not collapse.
            
            By default, is None (no constraint is applied.)
   
        comp_enumstep(int):
            Enumeration step for compositions. If otherwise specified, will thin
            enumerated compositions by this value. For example, if we have BCC 
            Ag-Li alloy, and a supercell of totally 256 sites. If step = 1, we 
            can have 257 compositions. But if we don't want to enumerate so many,
            we can simply write step = 4, so 4 sites are replaced each time, we 
            get totally 65 compositions.

        basis_type(string):
            Type of basis used in cluster expansion. Needs to be specified if you 
            initalize enumeration from an existing CE, and its basis is different 
            from 'indicator'!
            If you used custom basis, just type 'custom' for this term. But hopefully
            this will not happen too often.
    
        select_method(str): 
            Method used in structure selection from enumerated pool.
            'CUR'(default):
                Select by highest CUR scores
            'random':
                Select randomly
            Both methods guarantee inclusion of the ground states at initialization.
    """

    def __init__(self,prim,sublat_list = None,\
                 previous_ce = None,\
                 previous_fe_mat = [],\
                 transmat=[[1,0,0],[0,1,0],[0,0,1]],max_natoms=200,\
                 max_sc_cond = 8, min_sc_angle = 30,\
                 comp_restrictions=None,comp_enumstep=1,\
                 basis_type = 'indicator',\
                 select_method = 'CUR'):

        self.prim = prim

        bits = get_allowed_species(self.prim)
        if sublat_list is not None:
            #User define sublattices, which you may not need very often.
            self.sublat_list = sublat_list
            self.sl_sizes = [len(sl) for sl in self.sublat_list]
            self.bits = [bits[sl[0]] for sl in self.sublat_list]
        else:
            #Automatic sublattices, same rule as smol.moca.Sublattice:
            #sites with the same compositioon are considered same sublattice.
            self.sublat_list = []
            self.bits = []
            for s_id,s_bits in enumerate(bits):
                if s_bits in self.bits:
                    s_bits_id = self.bits.index(s_bits)
                    self.sublat_list[s_bits_id].append(s_id)
                else:
                    self.sublat_list.append([s_id])
                    self.bits.append(s_bits)
            self.sl_sizes = [len(sl) for sl in self.sublat_list]

        #Check if this cluster expansion should be charged
        self.is_charged_ce = False
        for sl_bits in self.bits:
            for bit in sl_bits:
                if type(bit)!= Element and bit_oxi_state!=0:
                    self.is_charged_ce = True
                    break

        self.basis_type = basis_type
        if previous_ce is not None:
            self.ce = previous_ce
        else:
            #An empty cluster expansion with the points and ewald term only
            #Default is indicator basis
            c_spc = ClusterSubspace.from_radii(self.prim,{2:0.01},\
                                    basis = self.basis_type)
            if self.is_charged_ce:
                c_spc.add_external_term(EwaldTerm())
                coef = np.zeros(c_spc.n_bit_orderings+1)
                coef[-1] = 1.0
            else:
                coef = np.zeros(c_spc.n_bit_orderings)

            self.ce = ClusterExpansion(c_spc,coef,[])
            
        self.previous_femat = np.array(previous_femat)
        self.transmat = transmat
        self.max_natoms = max_natoms
        self.max_sc_cond = max_sc_cond
        self.min_sc_angle = min_sc_angle

        self.comp_space = CompSpace(self.bits,sl_sizes = self.sl_sizes)
        self.comp_restrictions = comp_restrictions
        self.comp_enumstep = comp_enumstep

        self.select_method = select_method

        #These data will be saved as pre-grouped, multi-index form, not plain pandas
        #DF, in case pandas.groupby takes too much time to group data by sc matrix and
        #composition. A self.data property will still be available for easy access to
        #a pandas-like DF
        self._sc_matrices = None
        self._sc_comps = None
        self._eq_occus = None

        self._enum_strs = None
        self._enum_occus = None
        self._enum_corrs = None
        #id starts from 0
        self._enum_ids = None
    
    @property
    def n_strs(self):
        """
        Number of enumerated structures.
        """
        if self._enum_strs is None or self._enum_occus is None or self._enum_corrs is None
           or self._enum_ids is None:
            return 0
        else:
            return sum([len(key_strs) for key_strs in self._enum_strs])

    @property
    def sc_matrices(self):
        """
        Supercell matrices used for structural enumeration. If none yet, will be 
        enumerated.
        Return:
            A list of 2D lists.
        """
        if self._sc_matrices is None:
            trans_size = int(round(abs(np.linalg.det(self.transmat))))
            max_det = self.max_natoms // (len(self.prim) * trans_size)
            self._sc_matrices =  _enumerate_matrices(max_det, self.prim.lattice,\
                                                        transmat=self.transmat,\
                                                        max_sc_cond = self.max_sc_cond,\
                                                        min_sc_angle = self.min_sc_cond)
        return self._sc_matrices

    def set_sc_matrices(self,matrices=[]):
        """
        Interface method to preset supercell matrices before enumeration. If no preset
        is given, supercell matrices will be automatically enumerated.
        Input:
            matrices: A list of Arraylike. Should be np.ndarrays or 2D lists.
        """
        new_sc_matrices = []
        for mat in matrices:
            if type(mat) == list and type(mat[0])== list:
                new_sc_matrices.append(mat)
            elif type(mat) == np.ndarray:
                new_sc_matrices.append(mat.tolist())
            else:
                warnings.warn('Given matrix {} is not in a valid input format. Dropped.'.format(mat))
        if len(new_sc_matrices)==0:
            raise ValueError('No supercell matrices will be reset!')
        else:
            print("Reset supercell matrices to:\n{}\nAll previous compositions will be cleared and re-enumed!"
                  .format(matrices))
            self._sc_matrices = new_sc_matrices
            self._sc_comps = None

    @property
    def sc_comps(self):
        """
        Enumerate proper compositions under each supercell matrix.
        Return:
            List of tuples, each tuple[0] is supercell matrix, tuple[1] is a list of composition
            per sublattice. 
            These will be used as keys to group structures, therefore to avoid unneccessary dedups 
            between groups.
        """
        if self._sc_comps is None:
            self._sc_comps = []
            for mat in self.sc_matrices:
                scs = int(round(abs(np.linalg.det(mat))))
                self._sc_comps.extend([(mat,comp) for comp in 
                                        self.comp_space.frac_grids(sc_size=scs/self.comp_enumstep,\
                                                                   form='composition')
                                        if self._check_comp(comp) 
                                      ])
        return self._sc_comps

    def _check_comp(self,comp):
        """
        Check whether a given composition violates self.comp_restrictions.
        """
        if type(self.comp_restrictions) == dict:
            for sp,(lb,ub) in self.comp_restrictions.items():
                sp_name = get_specie(sp)
                sp_num = 0
                for sl_comp,sl_size in zip(comp,sl_sizes):
                    if sp_name in sl_comp:
                        sp_num += sl_comp[sp_name]*sl_size   
                sp_frac = float(sp_num)/sum(sl_sizes)
                if not (sp_frac<=ub and sp_frac >=lb):
                    return False

        elif type(self.comp_restrictions) == list and len(self.comp_restrictions)>=1 \
             and type(self.comp_restrictions[0]) == dict:
            for sl_restriction, sl_comp in zip(self.comp_restrictions,comp):
                for sp in sl_restriction:
                    sp_name = get_specie(sp)
                    if sp_name not in sl_comp:
                        sp_frac = 0
                    else:
                        sp_frac = sl_comp[sp_name]

                    if not (sp_frac<=ub and sp_frac >=lb):
                       return False                   

        return True

    @property
    def eq_occus(self):
        """
        A list to store equilibrated occupations under different supercell matrices and compositions.
        If none yet, will be randomly generated.
        """
        if self._eq_occus is None:
            self._eq_occus = [None for sc,comp in self.sc_comps]
        return self._eq_occus

    def _initialize_occu_under_sccomp(self,sc_mat,comp):
        """
        Get an initial occupation under certain supercell matrix and composition.
        Inputs:
            sc_mat(3*3 ArrayLike):
                Supercell matrix
            comp(Union([List[pymatgen.Composition],List[SiteSpace], List[dict] ])):
                Compositions on each sublattice. Fractional.
        Output:
            Arraylike of integers. Encoded occupation array.
        """
        scs = int(round(abs(np.linalg.det(sc_mat))))
        
        sc_sublat_list = []
        #Generating sublattice list for a supercell, to be used in later radomization code.
        for sl in self.sublat_list:
            sl_sites = []
            for s in sl:
                sl_sites.extend(list(range(s*scs,(s+1)*scs)))
            sc_sublat_list.append(sl_sites)

        #randomly initalize 50 occus, pick 10 with lowest ewald energy (if is_charged_ce),
        #Then choose one final as initalization randomly
        int_comp = []
        for sl_frac_comp, sl_sites in zip(comp,sc_sublat_list):
            sl_int_comp = {}
            for k,v in sl_frac_comp.items():
            #Tolerance of irrational compositions
                if abs(v*len(sl_sites)-round(v*len(sl_sites)))>0.1:
                    raise ValueError("Sublattice compostion {} can not be achieved with sublattice size {}."\
                                     .format(sl_frac_comp,len(sl_sites)))
                sl_int_comp[k] = int(round(v*len(sl_sites)))
            int_comp.append(sl_int_comp)

        rand_occus = []
        for i in range(50):
            #Occupancy is coded
            occu = [None for i in range(len(self.prim)*scs)]
            for sl_id, (sl_int_comp, sl_sites) in enumerate(zip(int_comp,sc_sublat_list)):
                sl_sites_shuffled = random.shuffle(deepcopy(sl_sites))

                n_assigned = 0
                for sp,n_sp in sl_int_comp.items():
                    for s_id in sl_sites_shuffled[n_assigned:n_assigned+n_sp]:
                        sp_name = get_specie(sp)
                        sp_id = self.bits[sl_id].index(sp_name)
                        occu[s_id] = sp_id
                    n_assigned += n_sp

            for sp in occu:
                if sp is None:
                    raise ValueError("Unassigned site in occupation: {}, composition is: {}!".format(occu,comp))    

            rand_occus.append(occu)

        if self.is_charged_ce:
            rand_occus = sorted(rand_occus,key=lambda occu:\
                                _get_ewald_from_occu(occu,sc_sublat_list,self.bits,self.prim,sc_mat))

        return random.choice(rand_occus[:10])

    def generate_structures(self,n_per_key = 3, keep_gs = True):
        """
        Enumerate structures under all different key = (sc_matrix, composition). The eneumerated structures
        will be deduplicated and selected based on CUR decomposition score to maximize sampling efficiency.
        Will run on initalization mode if no previous enumearation present, and on addition mode if previous
        enum present.
        Inputs:
            n_per_key(int):
                will select n_per_key*N_keys number of new structures. If pool not large enough, select all of
                pool. Default is 3.
            keep_gs(Bool):
                if true, will always select current ground states. Default is True.
    
        No outputs. Updates in object private attributes.
        """
        N_keys = len(self.sc_comps)
        if self._enum_str is None or self._enum_occus is None or self._enum_ids is None or self._enum_corrs is None:
        #Initialization branch. No deduplication with previous enums needed.
            print('*Structures initialization.')
            self._enum_str = [[] for i in range(N_keys)]
            self._enum_occus = [[] for i in range(N_keys)]
            self._enum_corrs = [[] for i in range(N_keys)]
            self._enum_ids = [[] for i in range(N_keys)]
            
        eq_occus_update = []
        enum_str = []
        enum_occus = []
        enum_corrs = []

        n_enum = 0 
        keep_fids = [] #Flatten ids of ground states that must be keeped
        for (sc_mat,comp),eq_occu,old_strs in zip(self.sc_comps,self.eq_occus,self._enum_str):

            str_pool,occu_pool = self._enum_configs_under_sccomp(sc_mat,comp,eq_occu)
            corr_pool = [list(self.ce.cluster_subspace.corr_from_structure(s)) for s in str_pool]

            #Update GS
            gs_occu = deepcopy(occu_pool[0])
            eq_occus_update.append(gs_occu)

            dedup_ids = []
             for s1_id, s1 in enumerate(str_pool):
                 dupe = False
                 for s2_id, s2 in enumerate(old_strs):
                     if sm.fit(s1,s2):
                         dupe = True
                         break
                 if not dupe:
                     dedup_ids.append(s1_id)
 
            #gs(id_0 in occu_pool) will always be selected if unique
            str_pool = [str_pool[d_id] for d_id in dedup_ids]
            occu_pool = [occu_pool[d_id] for d_id in dedup_ids]
            corr_pool = [corr_pool[d_id] for d_id in dedup_ids]
            n_enum += len(str_pool)
            
            if 0 in dedup_ids: #New GS detected, should be keeped
                keep_fids.append(n_enum-len(str_pool))
        
        if not keep_gs:
            keep_fids = []

        print('Enumerated {} unique structures. Selecting.'.format(n_enum))

        #Flatten data for processing. deflat_rules will be the same.
        str_pool_flat, deflat_rule = _flatten_2d(str_pool)
        occu_pool_flat, deflat_rule = _flatten_2d(occu_pool)
        corr_pool_flat, deflat_rule = _flatten_2d(corr_pool)
        old_femat, old_delfat_rule = _flatten_2d(self._enum_corrs)

        selected_fids = _select_rows(corr_pool_flat,n_select=n_per_key*N_keys,
                                     old_femat =_flatten_2d(self._enum_corrs),
                                     method=self.select_method,
                                     keep=keep_fids) 
            
        #Muting unselected structure, prepare for deflattening
        str_pool_flat = [i for i_id,i in enumerate(str_pool_flat) if i_id in selected_fids else None]
        occu_pool_flat = [i for i_id,i in enumerate(occu_pool_flat) if i_id in selected_fids else None]
        corr_pool_flat = [i for i_id,i in enumerate(corr_pool_flat) if i_id in selected_fids else None]

        #Deflatten
        str_pool = _deflat_2d(str_pool_flat,deflat_rule)
        occu_pool = _deflat_2d(occu_pool_flat,deflat_rule)
        corr_pool = _deflat_2d(corr_pool_flat,deflat_rule)

        cur_id = deepcopy(self.n_strs)
        n_strs_init = deepcopy(self.n_strs)

        for key_id in range(N_keys):
            for i_id in range(len(str_pool[key_id])):
                self._enum_strs[key_id].append(str_pool[key_id][i_id])
                self._enum_occus[key_id].append(occu_pool[key_id][i_id])
                self._enum_corrs[key_id].append(corr_pool[key_id][i_id])
                self._enum_ids[key_id].append(cur_id)
                #So id starts from 0!
                cur_id += 1

        self._eq_occus = eq_occus_update
        print("*Added with {} new unique structures.".format(self.n_strs-n_strs_init))
        print("*Ground states updated.")
            
    def clear_structures(self):
        """
        Clear enumerated structures.
        """
        print("Warning: Previous enumerations cleared.")
        self._enum_str = None
        self._enum_corrs = None
        self._enum_occus = None
        self._enum_ids = None

    #POTENTIAL FUTURE ADDITION: add_single_structure

    def _enum_configs_under_sccomp(self,sc_mat,comp,eq_occu=None):
        """
        Built in method to generate occupations under a supercell matrix and a fixed composition.
        Assuming that atoms in the supercells are generated by pymatgen.structure.make_supercell
        from the primitive cell, which simply replicates and stacks atom in their initial order.
        For example: [Ag, Cu] -> [Ag, Ag, Cu, Cu]

        Inputs:
            sc_mat(3*3 ArrayLike):
                Supercell matrix
            comp(Union([List[pymatgen.Composition],List[SiteSpace], List[dict] ])):
                Compositions on each sublattice. Fractional.           
            eq_occu(List of ints):
                Occupation array of ground state under that composition. If None, will anneal to
                calculate
        Return:
            rand_strs_dedup:
                List of deduped pymatgen.Structures
            rand_occus_dedup:
                List of deduped occupation arrays. All in list of ints.
        """

        print("\nEnumerating under supercell: {}, composition: {}.".format(sc_mat,comp))
 
        is_indicator = (self.basis_type == 'indicator')
        scs = int(round(abs(np.linalg.det(sc_mat))))

        #Anneal n_atoms*100 per temp, Sample n_atoms*500
        n_steps_anneal = scs*len(self.prim)*100
        n_steps_sample = scs*len(self.prim)*500
        thin = max(1,n_steps_sample//200)

        anneal_series = [2000,1340,1020,700,440,280,200,120,80,20]
        sample_series = [500,1500,10000]

        ensemble = CanonicalEnsemble.from_cluster_expansion(self.ce, sc_mat, 
                                                            temperature = 1000, 
                                                            optimize_inidicator=is_indicator)
        sampler = Sampler.from_ensemble(ensemble)
        processor = ensemble.processor
        sm = StructureMatcher()
 
        if eq_occu is None:
        #If not annealed before, will anneal and save GS
            print("**Initializing occupation.")
            init_occu = self._initialize_occu_under_sccomp(sc,comp)
 
            print("****Annealing to the ground state.")
            sampler.anneal(anneal_series,n_steps_anneal,
                           initial_occupancies=np.array([init_occu]))
 
            print('*****Equilibrium GS found!')
            gs_occu = list(sampler.samples.get_minimum_energy_occupancy())
        else:
        #If annealed before, will use old GS
            gs_occu = eq_occu
 
        #Will always contain GS structure at the first position in list
        rand_occus = [gs_occu]
        #Sampling temperatures
        
        for T in sample_series:
            print('**Getting samples under {} K.'.format(T))
            sampler.samples.clear()
            sampler._kernel.temperature = T
            #Equilibriate
            print("****Equilibration run.")
            sampler.run(n_steps_sample,
                        initial_occupancies=np.array([gs_occu]),
                        thin_by=thin,
                        progress=True)
            sa_occu = sampler.samples.get_occupancies()[-1]
            sampler.samples.clear()
            #Sampling
            print("****Generative run.")
            sampler.run(n_steps_sample,
                        initial_occupancies=np.array([sa_occu]),
                        thin_by=thin,
                        progress=True)
            #default flat=True will remove n_walkers dimension. See moca docs.
            rand_occus.extend(np.array(sampler.samples.get_occupancies()).tolist())          

        rand_strs = [processor.structure_from_occupancy(occu) for occu in rand_occus]
        #Internal deduplication
        rand_dedup = []
        for s1_id,s1 in enumerate(rand_strs):
            duped = False
            for s2_id,s2 in enumerate(rand_dedup):
                if sm.fit(s1,s2):
                    duped = True
                    break
            if not duped:
                rand_dedup.append(s1_id)

        print('{} unique structures generated.'.format(len(rand_dedup)))
        rand_strs_dedup = [rand_strs[s_id] for s_id in rand_dedup]
        rand_occus_dedup = [rand_occus[s_id] for s_id in rand_dedup]

        return rand_strs_dedup, rand_occus_dedup

## TODO: self.data, as_dict, from_dict
