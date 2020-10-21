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

def _enumerate_sc_matrices(max_det, lat,\
                           transmat=[[1,0,0],[0,1,0],[0,0,1]],\
                           max_sc_cond = 8,\
                           min_sc_angle = 30,\
                           n_select=20):
    """
    Enumerate proper supercell matrices with maximum size up to a number.
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

        select_method(string):
            Method used to select most uncorrelated structures from the monte-carlo
            sample pool. Currently supporting:
            'CUR': 
                Doing a CUR decompsition and select structures with highest scores.
                Also known as Nystrom selection. (Default)
            'CX':
                Doing a CX decomposiiton and select structures with highest scores.

        basis(string):
            Type of basis used in cluster expansion. Needs to be specified if you 
            initalize enumeration from an existing CE, and its basis is different 
            from 'indicator'!
            If you used custom basis, just type 'custom' for this term. But hopefully
            this will not happen too often.
    """

    def __init__(self,prim,sublat_list = None,\
                 previous_ce = None,\
                 previous_fe_mat = [],\
                 transmat=[[1,0,0],[0,1,0],[0,0,1]],max_natoms=200,\
                 max_sc_cond = 8, min_sc_angle = 30,\
                 comp_restrictions=None,comp_enumstep=1,\
                 select_method='CUR',
                 basis_type = 'indicator'):

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

        self._sc_matrices = None
        self._sc_comps = None
        self._eq_occus = None

        self._entry_pool = []

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
            self._sc_matrices =  _enumerate_sc_matrices(max_det, self.prim.lattice,\
                                                        transmat=self.transmat,\
                                                        max_sc_cond = self.max_sc_cond,\
                                                        min_sc_angle = self.min_sc_cond)
        return self._sc_matrices

    def set_sc_matrices(self,matrices):
        """
        Interface method to preset supercell matrices before enumeration. If no preset
        is given, supercell matrices will be automatically enumerated.
        Input:
            matrices: A list of Arraylike. Should be np.ndarrays or 2D lists.
        """
        self._sc_matrices = []
        for mat in matrices:
            if type(mat) == list and type(mat[0])== list:
                self._sc_matrices.append(mat)
            elif type(mat) == np.ndarray:
                self._sc_matrices.append(mat)
            else:
                warnings.warn('Given matrix {} is not in a valid input format.'.format(mat))
        if len(self._sc_matrices)==0:
            self._sc_matrices = None
            raise ValueError('No supercell matrices added!')
        else:
            print("Reset supercell matrices to:\n",matrices)

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
            self._eq_occus = [self._initialize_occu_under_sccomp(sc,comp) for sc,comp in self.sc_comps]

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

    def _enum_configs_under_sccomps(self):
        """
        Built in method to generate occupations under a supercell matrix and a fixed composition.
        Assuming that atoms in the supercells are generated by pymatgen.structure.make_supercell
        from the primitive cell, which simply replicates and stacks atom in their initial order.
        For example: [Ag, Cu] -> [Ag, Ag, Cu, Cu]

        Return:
            List of Lists of deduped pymatgen.Structure under each sc_mat and comp in self.sc_comps
        """
        for (sc_mat,comp),init_occu in zip(self.sc_comps,self.eq_occus):
            print("Enumerating under supercell: {}, composition: {}.".format(sc_mat,comp))
    
            is_indicator = (self.basis_type == 'indicator')
            sm = StructureMatcher()
    
            scs = int(round(abs(np.linalg.det(sc_mat))))
            #Anneal n_atoms*100, Sample n_atoms*500
            n_steps_anneal = scs*len(self.prim)*100
            n_steps_sample = scs*len(self.prim)*500
    
            ensemble = CanonicalEnsemble.from_cluster_expansion(self.ce, sc_mat, 
                                                                temperature = 1000, 
                                                                optimize_inidicator=is_indicator)
            sampler = Sampler.from_ensemble(ensemble)
            temp_series = 3000/np.linspace(1,50,10)

##TODO: Finish sampler



        
        print('Initial encoded occupation:\n',init_occu)
        print('Initial decoded occupancy:\n',processor.decode_occupancy(init_occu))


