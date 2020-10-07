__author__="Fengyu Xie"

"""
This module implements a StructureEnumerator class for CE sampling.
"""
import warnings
import numpy as np

from monty.json import MSONable

from pymatgen import Structure,Lattice

from smol.cofe.extern.ewald import EwaldTerm
from smol.cofe.configspace.clusterspace import ClusterSubspace
from smol.cofe.configspace.domain import get_allowed_species,get_specie
from smol.cofe.expansion import ClusterExpansion
from smol.moca.processor import *

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

        comp_restrictions(Dict or List or Dict or None):
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

        enum_method(string):
            Occupation enumeration method. Currently supporting:
            'full-mc':
                Doing a monte-carlo sampling with full ce. If we don't have a 
                previous ce, then do a mc with only the ewald term. (Default)
            'ew-mc': 
                Doing a monte-carlo sampling with only ewald term. States with 
                high ewald energy will be dropped.
            'random': 
                Full random occupation (not recommended)

        select_method(string):
            Method used to select most uncorrelated structures from the monte-carlo
            sample pool. Currently supporting:
            'CUR': 
                Doing a CUR decompsition and select structures with highest scores.
                Also known as Nystrom selection. (Default)
            'CX':
                Doing a CX decomposiiton and select structures with highest scores.
    """

    def __init__(self,prim,sublat_list = None,\
                 previous_ce = None,\
                 previous_fe_mat = [],\
                 transmat=[[1,0,0],[0,1,0],[0,0,1]],max_natoms=200,\
                 max_sc_cond = 8, min_sc_angle = 30,\
                 comp_restrictions=None,\
                 enum_method='full-mc',\
                 select_method='CUR'):

        self.prim = prim

        bits = get_allowed_species(self.prim)
        if sublat_list is not None:
            self.sublat_list = sublat_list
            self.sl_sizes = [len(sl) for sl in self.sublat_list]
            self.bits = [bits[sl[0]] for sl in self.sublat_list]
        else:
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

        #if enum_method = *-mc, processor will be used to sample structures.
        self.ew_term = EwaldTerm()

        if previous_ce is not None:
            self.ce = previous_ce
        else:
            #An empty cluster expansion with ewald term only
            c_spc = ClusterSubspace.from_radii(self.prim,{2:0.01})
            c_spc.add_external_term(self.ew_term)
            coef = np.zeros(c_spc.n_bit_orderings+1)
            coef[-1] = 1

            self.ce = ClusterExpansion(c_spc,coef,[])
            
        self.previous_femat = np.array(previous_femat)
        self.transmat = transmat
        self.max_natoms = max_natoms
        self.max_sc_cond = max_sc_cond
        self.min_sc_angle = min_sc_angle

        self.comp_restrictions = comp_restrictions

        self.enum_method = enum_method
        self.select_method = select_method

        self._sc_matrices = None


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
