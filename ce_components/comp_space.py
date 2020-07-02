import numpy as np
import polytope as pc

from itertools import combinations,product

from copy import deepcopy

from monty.json import MSONable

import os,sys
this_file_path = os.path.abspath(__file__)
this_file_dir = os.path.dirname(this_file_path)
parent_dir = os.path.dirname(this_file_dir)
sys.path.append(parent_dir)
sys.path.append(this_file_dir)

from utils.enum_utils import *
from specie import *
from utils.comp_utils import *

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
OUTOFSUBSPACEERROR = ValueError("Given coordinate falls outside the subspace.")


####
# Finding minimun charge-conserved, number-conserved flips to establish constrained
# coords system.
####

def get_unit_swps(bits):
    """
    Get all possible single site flips on each sublattice, and the charge changes that 
    they gives rise to.
    For example, 'Ca2+ -> Mg2+', and 'Li+ -> Mn2+'+'F- -> O2-' are all such type of 
    flips.
    Inputs:
        bits: 
            a list of CESpecies on each sublattice. For example:
            [[CESpecie.from_string('Ca2+'),CESpecie.from_string('Mg2+'))],
             [CESpecie.from_string('O2-')]]
    Outputs:
        unit_n_swps: 
            a flatten list of all possible, single site flips represented in n_bits, 
            each term written as:
            (flip_to_nbit,flip_from_nbit,sublattice_id_of_flip)
        chg_of_swps: 
            change of charge caused by each flip. A flatten list of integers
        swp_ids_in_sublat:
             a list specifying which sublattice should each flip belongs to. Will
             be used to set normalization constraints in comp space.
    """
    n_bits = get_n_bits(bits)

    unit_swps = []
    unit_n_swps = []
    swp_ids_in_sublat = []
    cur_swp_id = 0
    for sl_id,sl_sps in enumerate(bits):
        unit_swps.extend([(sp,sl_sps[-1],sl_id) for sp in sl_sps[:-1]])
        unit_n_swps.extend([(sp,n_bits[sl_id][-1],sl_id) for sp in n_bits[sl_id][:-1]])
        swp_ids_in_sublat.append([cur_swp_id+i for i in range(len(sl_sps)-1)])
        cur_swp_id += (len(sl_sps)-1)
        #(sp_before,sp_after,sublat_id)

    chg_of_swps = [p[0].oxidation_state-p[1].oxidation_state for p in unit_swps]

    return unit_n_swps,chg_of_swps,swp_ids_in_sublat

def flipvec_to_operations(unit_n_swps,prim_lat_vecs):
    """
    This function translates flips from their vector from into their dictionary
    form.
    Each dictionary is written in the form below:
    {
     'from': 
           {sublattice_id:
               {specie_nbit_id: 
                   number_of_this_specie_to_be_anihilated_from_this_sublat
               }
                ...
           }
           ...
     'to':
           { 
           ...     numbr_of_this_specie_to_be_generated_on_this_sublat
           }
    }
    """
    operations = []
    for flip_vec in prim_lat_vecs:
        operation = {'from':{},'to':{}}
        for n_flip,flip in zip(flip_vec,unit_n_swps):
            flip_to,flip_from,sl_id = flip
            if sl_id not in operation['from']:
                operation['from'][sl_id]={}
            if sl_id not in operation['to']:
                operation['to'][sl_id]={}
            if flip_from not in operation['from'][sl_id]:
                operation['from'][sl_id][flip_from]=0
            if flip_to not in operation['to'][sl_id]:
                operation['to'][sl_id][flip_to]=0
            operation['from'][sl_id][flip_from]+=n_flip
            operation['to'][sl_id][flip_to]+=n_flip
            #deduplicate
            operation_dedup = {'from':deepcopy(operation['from']),'to':{}}
            for sl_id in operation['to']:
                operation_dedup['to'][sl_id]={}
                for sp_id in operation['to'][sl_id]:
                    if sp_id in operation['from'][sl_id]:
                        left = operation['from'][sl_id][sp_id]
                        right = operation['to'][sl_id][sp_id]
                        if left > right:
                            operation_dedup['from'][sl_id][sp_id]-=right
                        elif left == right:
                            operation_dedup['from'][sl_id].pop(sp_id)
                        else:
                            operation_dedup['to'][sl_id][sp_id]=(right-left)
                            operation_dedup['from'][sl_id].pop(sp_id)
                    else:
                        operation_dedup['to'][sl_id][sp_id]=\
                        operation['to'][sl_id][sp_id]
            #Remove empty terms
            operation_clean = {'from':{},'to':{}}
            for sl_id in operation_dedup['from']:
                if len(operation_dedup['from'][sl_id])!=0:
                    operation_clean['from'][sl_id]=\
                    deepcopy(operation_dedup['from'][sl_id])
            for sl_id in operation_dedup['to']:
                if len(operation_dedup['to'][sl_id])!=0:
                    operation_clean['to'][sl_id]=\
                    deepcopy(operation_dedup['to'][sl_id])

        operations.append(operation_clean)

    return operations    

def visualize_operations(operations,bits):
    """
    This function turns an operation dict into an string for easy visualization,
    """
    operation_strs = []
    for operation in operations:
        from_strs = []
        to_strs = []
        for sl_id in operation['from']:
            for swp_from,n in operation['from'][sl_id].items():
                from_name = bits[sl_id][swp_from].specie_string
                from_strs.append('{} {}({})'.format(n,from_name,sl_id))
        for sl_id in operation['to']:
            for swp_to,n in operation['to'][sl_id].items():
                to_name = bits[sl_id][swp_to].specie_string
                to_strs.append('{} {}({})'.format(n,to_name,sl_id))

        from_str = ' + '.join(from_strs)
        to_str = ' + '.join(to_strs)
        operation_strs.append(from_str+' -> '+to_str) 

    return '\n'.join(operation_strs)

####
# Compsitional space class
####

class CompSpace(MSONable):
    """
        This class generates a CN-compositional space from a list of CESpecies and sublattice
        sizes.

        A composition in CEAuto can be expressed in two forms:
        1, A Coordinate in unconstrained space, with 'single site flips' as basis vectors, and
           a 'background occupation' as the origin.
           We call this 'unconstr_coord'
        2, A Coordinate in constrained, charge neutral subspace, with 'charge neutral, number
           conserving elementary flips as basis vectors, and a charge neutral composition as 
           the origin.(Usually selected as one vertex of the constrained space.)
           We call this 'constr_coord'.

        For example, if bits = [[Li+,Mn3+,Ti4+],[P3-,O2-]] and sl_sizes = [1,1] (LMTOF rock-salt), then:
           'single site flips' basis are:
                Ti4+ -> Li+, Ti4+ -> Mn3+, O2- -> P3-
           'Background occupation' origin shall be:
                (Ti4+ | O-),supercell size =1
            The unconstrained space's dimensionality is 3.

           'charge neutral, number conserving elementary flips' bais shall be:
                3 Mn3+ -> 2 Ti4+ + Li+, Ti4+ + P3- -> O2- + Mn3+
           'charge neutral composition' origin can be chosen as:
                (Mn3+ | P-),supercell size = 1
            The constrained subspace's dimensionality is 2.

        Given composition:
            (Li0.5 Mn0.5| O), supercell size=1
            It's coordinates in the 1st system will be (0.5,0.5,0)
            In the second system that will be (0.5,1.0)

        When the system is always charge balanced (all the flips are charge conserved, background occu
        has 0 charge), then representation 1 and 2 are the same.

        Compspace class provides methods for you to convert between these two representations easily,
        write them into human readable form. It will also allow you to enumerate all possible integer 
        compositions given supercell size. Most importantly, it defines the CEAuto composition 
        enumeration method. For the exact way we do enumeration, please refer to the documentation of 
        each class methods.
            

    """
    def __init__(self,bits,sl_sizes=None):
        """
        Inputs:
            bits: 
                bit list, same as appeared in get_n_bits. 
                Sorting bits before using is highly recommended.
            sl_sizes: 
                Sublattice sizes in a PRIMITIVE cell. A list of integers. 
                len(bits)=# of sublats=len(sl_sizes).
                If None given, sl_sizes will be reset to [1,1,....]
        """
        self.bits = bits
        self.nbits = get_n_bits(bits)
        if sl_sizes is None:
            self.sl_sizes = [1 for i in range(len(self.bits))]
        elif len(sl_sizes)==len(bits):
            self.sl_sizes = sl_sizes
        else:
            raise ValueError("Sublattice number mismatch: check bits and sl_sizes parameters.")
  
        self.N_sts_prim = sum(self.sl_sizes)

        self.unit_n_swps,self.chg_of_swps,self.swp_ids_in_sublat = get_unit_swps(self.bits)

        self._constr_spc_basis = None
        self._constr_spc_vertices = None
        #Minimum supercell size required to make vetices coordinates all integer.
        self._polytope = None

        self._min_sc_size = None
        self._min_int_vertices = None
        self._min_grid = None
        #self._constr_spc_origin = self._constr_spc_vertices[0]
    
    @property
    def bkgrnd_chg(self):
        chg = 0
        for sl_bits,sl_size in zip(self.bits,self.sl_sizes):
            chg += sl_bits[-1].oxidation_state*sl_size
        return chg

    @property
    def unconstr_dim(self):
        return len(self.unit_n_swps)
 
    @property
    def is_charge_constred(self):
        d = len(self.chg_of_swps)
        return not(np.allclose(np.zeros(d),self.chg_of_swps) and self.bkgrnd_chg==0)

    @property
    def dim(self):
        """
        Dimensionality of the constrained space.
        """
        d = self.unconstr_dim
        if not self.is_charge_constred:
            return d
        else:
            return d-1

    @property
    def constr_spc_basis(self):
        """
        Get 'minimal charge-neutral flips basis' in vector representation. 
        Given any compositional space, all valid, charge-neutral compoisitons are 
        integer grids on this space or its subspace. What we do is to get the primitive
        lattice vectors of the lattice defined by these grid points.
        For example:
        [[Li+,Mn3+,Ti4+],[P3-,O2-]] system, minimal charge and number conserving flips 
        are:
        3 Mn3+ <-> Li+ + 2 Ti4+, 
        Ti4+ + P3- <-> Mn3+ + O2- 
        Their vector forms are:
        (1,-3,0), (0,1,-1)  

        Type: list of np.arrays.
        """        
        if self._constr_spc_basis is None:
            self._constr_spc_basis = \
                 get_integer_basis(self.chg_of_swps,sl_flips_list=self.swp_ids_in_sublat)
        return self._constr_spc_basis

    @property
    def min_flips(self):
        """
        Dictionary representation of minimal charge conserving flips.
        """
        _operations = flipvec_to_operations(self.unit_n_swps,\
                                           self.constr_spc_basis)
        return _operations

    @property
    def min_flip_strings(self):
        """
        Human readable minial charge conserving flips, written in ionic equations.
        """
        return visualize_operations(self.min_flips,self.bits)

    @property
    def polytopes(self):
        """
        Express the configurational space (supercellsize=1) as a polytope.Polytope object.
        Shall be expressed in type 2 basis
        """
        if self._polytope is None:
            facets_unconstred = []
            for sl_flp_ids,sl_size in zip(self.swp_ids_in_sublat,self.sl_sizes):
                a = np.zeros(self.unconstr_dim)
                a[sl_flp_ids]=1
                bi = sl_size
                facets_unconstred.append((a,bi))
            #sum(x_i) for i in sublattice <= 1
            A_n = np.vstack([a for a,bi in facets_unconstred])
            b_n = np.array([bi for a,bi in facets_unconstred])
            # x_i >=0 for all i
            A = np.vstack(A_n,-1*np.identity(self.unconstr_dim))
            b = np.concatenate((b_n,np.zeros(self.unconstr_dim)))
 
            if not self.is_charge_constred:
                #polytope = pc.Polytope(A,b) Ax<=b.
                R = np.idendity(self.unconstr_dim)
                t = np.zeros(self.unconstr_dim)
                self._polytope = (A,b,R,t)          
            else:
                # x-t = R.T * x', where x'[-1]=0. Dimension reduced by 1.
                # We have to reduce dimension first because polytope package
                # can not handle polytope in a subspace. It will consider the
                # subspace as an empty set.
                R = np.vstack(self.constr_spc_basis+[np.array(self.chg_of_swps)])
                t = np.zeros(self.unconstr_dim)
                t[0] = -self.bkgrnd_chg/self.chg_of_swps[0]
                A_sub = A@R.T
                A_sub = A_sub[:,:-1]
                #slice A, remove last col, because the last component of x' will
                #always be 0
                b_sub = b-A@t
                self._polytope = (A_sub,b_sub,R,t)
        return self._polytope

    # A, b , R, t all np.arrays.
    @property
    def A(self):
        return self.polytope[0]

    @property
    def b(self):
        return self.polytope[1]
   
    #R and t are only valid in unit comp space (sc_size=1)!!!
    @property
    def R(self):
        return self.polytope[2]

    @property
    def t(self):
        return self.polytope[3]

    def is_in_subspace(self,x,sc_size=1,slack_tol=1E-5):
        """
        Given an unconstrained coordinate and its corresponding supercell size,
        check if it is in the constraint subspace.
        Returns a boolean.

        slack_tol:
            Maximum allowed slack to constraints
        """
        x = np.array(x)/sc_size

        if not self.is_charge_constred:
            b = self.A@x
            for bi_p,bi in zip(b,self.b):
                if bi_p-bi > slack_tol:
                    return False

        else:
            x_prime = np.linalg.inv((self.R).T)@(x_prime-self.t)
            if abs(x_prime[-1]) > slack_tol:
                return False
            b = self.A@x_prime[:-1]
            for bi_p,bi in zip(b,self.b):
                if bi_p-bi > slack_tol:
                    return False

        return True

    @property
    def constr_spc_vertices(self):
        """
        Find extremums of the constrained compositional space, when supercell size = 1.
        Shall be expressed in type 1 basis

        Type: np.array
        """
        if self._constr_spc_vertices is None:
            if not self.is_charge_constred:
                A,b,_,_=self._polytope
                poly = pc.Polytope(A,b)
                self._constr_spc_vertices = pc.extreme(poly)
            else:
                A,b,R,t=self._polytope
                poly_sub = pc.Polytope(A,b)
                vert_sub = pc.extreme(poly_sub)
                n = vert_sub.shape[0]
                vert = np.hstack(vert_sub,np.zeros(n))
                #Transform back into original space
                self._constr_spc_vertices = vert@R + t

        if len(self._constr_spc_vertices)==0:
            raise CHGBALANCEERROR

        return self._constr_spc_vertices

    @property
    def min_sc_size(self):
        """
        Minimal supercell size to get integer composition.
        In this function we also vertices of the compositional space after all coordinates
        are multiplied by self.min_sc_size. 
        """
        if self._min_sc_size or self._min_int_vertices is None:
            self._min_int_vertices, self._min_sc_size = \
                integerize_multiple(self.constr_spc_vertices)
        return self._min_sc_size

    @property
    def min_int_vertices(self):
        """
        Type: np.array
        """
        if self._min_sc_size or self._min_int_vertices is None:
            min_sc_size = self.min_sc_size
        return self._min_int_vertices

    @property
    def min_grid(self):
        """
        Get the minimum compositional grid: multiply the primitive cell compositional space
        by self.min_sc_size, and find all the integer grids in the new, magnified space.

        The way we enumerate compositions in CEAuto will be, we simply choose a supercell
        size that is a multiple of self.min_sc_size, and magnify min_grid by 
        (sc//self.min_sc_size)

        Type: a list of lists
        """
        if self._min_grid is None:
            limiters_ub = np.max(self.min_int_vertices,axis=0)
            limiters_lb = np.min(self.min_int_vertices,axis=0)
            limiters = list(zip(limiters_lb,limiters_ub))
            right_side = -1*self.bkgrnd_chg*self.min_sc_size
        
            grid = get_integer_grid(self.chg_of_swps,right_side=right_side,\
                                limiters = limiters)

            self._min_grid = []
            for p in grid:
                if self.is_in_subspace(p,sc_size=self.min_sc_size):
                    self._min_grid.append(p)

        return self._min_grid

    def enumerate_comps(self,magnif=1):
        """
        Enumerate sampled compositions by magnifying minimal grid by a magnification.
        Supercell size = self.min_sc_size*magnification.
        """
        return [[c*magnif for c in cp] for cp in self.min_grid]

    def unconstr_to_constr_coords(self,x,sc_size=1,to_int=False):
        """
        Unconstrained coordinate system to constrained coordinate system.
        In constrained coordinate system, a composition will be written as
        number of flips required to reach this composition from a starting 
        composition.        

        to_int: if true, round coords to integers.
        """
        if not self.is_in_subspace(x,sc_size=sc_size):
            raise OUTOFSUBSPACEERROR

        #scale down to unit comp space
        x = np.array(x)/sc_size
        if not self.is_charge_constred:
            x_prime = deepcopy(x)
        else:
            x_prime = np.linalg.inv((self.R).T)@(x-self.t)
            x_prime = x_prime[:-1]

        #scale back up to sc_size
        x_prime = x_prime*sc_size

        if to_int:
            x_prime = np.round(x_prime)
            x_prime = np.array(x_prime,dtype=np.int64)

        return x_prime

    def constr_to_unconstr_coords(self,x_prime,sc_size=1,to_int=False):
        """
        Constrained coordinate system to unconstrained coordinate system.
        """
        #scale down to unit comp space
        x_prime = np.array(x_prime)/sc_size
        if not self.is_charge_constred:
            x = deepcopy(x_prime)
        else:
            x = deepcopy(x_prime)
            x = np.concatenate((x,np.array([0])))
            x = (self.R).T@x + self.t
       
        #scale back up
        x = x*sc_size
        if not self.is_in_subspace(x,sc_size=sc_size):
            raise OUTOFSUBSPACEERROR

        if to_int:
            x = np.round(x)
            x = np.array(x_prime,dtype=np.int64)

        return x
 
    def unconstr_coords_to_compstat(self,x,sc_size=1,to_int=False):
        """
        Translate unconstrained coordinate to statistics of specie numbers on 
        each sublattice. Will have the same shape as self.nbits
        """
        v_id = 0
        compstat = [[0 for i in range(len(sl_nbits))] for sl_nbits in self.nbits]
        for sl_id,sl_nbits in enumerate(self.nbits):
            sl_sum = 0
            for b_id,bit in enumerate(sl_nbits[:-1]):
                compstat[sl_id][b_id] = x[v_id]
                sl_sum += x[v_id]
                v_id +=1
            compstat[sl_id][-1] = self.sl_sizes[sl_id]*sc_size - sl_sum
            if compstat[sl_id][-1] < 0:
                raise OUTOFSUBSPACEERROR

        if to_int:
            for sl_id,sl_nbits in enumerate(self.nbits):
                for b_id,bit in enumerate(sl_nbits):
                    compstat[sl_id][b_id] = int(round(compstat[sl_id][b_id]))

        return compstat

    def as_dict(self):
        bits_d = [[sp.as_dict() for sp in sl_sps] for sl_sps in self.bits]
        # constr_spc_basis is a list of np.arrays
        constr_spc_basis = [ev.tolist() for ev in self.constr_spc_basis]
        # constr_spc_vertices is a np.array
        constr_spc_vertices = self.conste_spc_vertices.tolist()
        # polytope is a tuple of np.arrays        
        poly = [item.tolist() for item in self.polytope]
        #min_int_vertices is a np.array
        min_int_vertices = self.min_int_vertices.tolist()
        #min_grid is a list of lists

        return {
                'bits': bits_d,
                'sl_sizes': self.sl_sizes,
                'constr_spc_basis': constr_spc_basis,
                'constr_spc_vertices': constr_spc_vertices,
                'polytope': poly,
                'min_sc_size': self.min_sc_size,
                'min_int_vertices': min_int_vertices,
                'min_grid': self.min_grid,
                '@module': self.__class__.__module__,
                '@class': self.__class__.__name__
               }

    @classmethod
    def from_dict(cls,d):
        bits = [[CESpecie.from_dict(sp_d) for sp_d in sl_sps] for sl_sps in d['bits']]
        
        obj = cls(bits,d['sl_sizes'])        
 
        if 'constr_spc_basis' in d:
            constr_spc_basis = d['constr_spc_basis']
            constr_spc_basis = [np.array(ev) for ev in constr_spc_basis]
            obj._constr_spc_basis = constr_spc_basis

        if 'constr_spc_vertices' in d:
            constr_spc_vertices = d['constr_spc_vertices']          
            obj._constr_spc_vertices = np.array(constr_spc_vertices)

        if 'polytope' in d:            
            poly = d['polytope']
            poly = [np.array(item) for item in poly]
            obj._polytope = poly

        if 'min_sc_size' in d:
            obj._min_sc_size = d['min_sc_size']

        if 'min_int_vertices' in d:
            min_int_vertices = d['min_int_vertices']
            obj._min_int_vertices = np.array(min_int_vertices)

        if 'min_grid' in d:
            obj._min_grid = d['min_grid']

        return obj 
