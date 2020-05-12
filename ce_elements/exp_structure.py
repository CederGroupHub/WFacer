#U=Integrate all comp analysis into here

import sys
import os
this_file_path = os.path.abspath(__file__)
this_file_dir = os.dirname(this_file_path)
parent_dir = os.dirname(this_file_dir)
sys.path.append(parent_dir)
from utils.specie_utils import *
from utils.enum_utils import *

from monty.json import MSONable

import numpy as np

class Sublattice(MSONable):
    def __init__(self,lattice,sites_in_prim,possible_sps,fractional=True):
        """
        Cluster expansion sublattice. Species are allowed to be flipped or swapped within
        a sublattice during occupation enum, but not allowed to be flipped or swapped 
        between sublattices.
        lattice:
            a pymatgen.lattice object, definining primitive cell vectors

        sites_in_prim:
            primitive cell coordinates of sites in this sublattice.
            For example, all 16a sites in a cubic spinel primitive cell.

        possible_sps:
            Species that are allowed to occupy this sublattice. Is a list of strings.
            For example, ['Li+','Mn2+','Mn3+'].
            Or can be a dict:
            {'Na+':(0.0,0.1),'K+':(0.0,1.0)}
            In which the tuple constrains the allowed occupation ratio of each specie.
            The left number is a lower-limit, and the right one is an upper-limit.
            Vacancies are denoted as specie string 'Vac'

        fractional:
            if True, all coordnates of sites are encoded in fractional coordinates.
            if False, all coordnates should be cartesian.
        """
        self.lattice = lattice
        self.frac_to_cart = lattice.matrix
        self.cart_to_frac = np.linalg.inv(lattice.matrix)
        sites_in_prim = np.array(sites_in_prim)        

        if not fractional:
            self.sites = sites_in_prim@self.cart_to_frac

        self.carts = self.lattice.get_cartesian_coords(self.sites)

        if type(possible_sps)==dict:
            self.species = [k for k,v in sorted(possible_sps.items())]
            self.constraints = [v for k,v in sorted(possible_sps_items())]
        elif type(possible_sps)==list:
            self.species = sorted(possible_sps)
            self.constraints = [(0.0,1.0) for i in range(len(self.species))]
        else:
            raise ValueError("Species not given in desired dict or list format!")

        self.charges = [get_oxi(sp_str) for sp_str in self.species]
        #get_oxi to be implemented in utils.specie_utils       
        self.N_sps = len(self.species)
       
    def enumerate_comps(self,fold=8):
        """
        Enumerates a possible compositions of a sublattice that satisfies self.constraints.
        Inputs:
            fold: use 1/fold as a step in composition space. For example, fold=2 when species
                  are 'Na','K' gives the enumerated list:
                  [{'Na:0.0,'K':1.0},{'Na':0.5,'K':0.5},{'Na':1.0,'K':0.0}]
        Outputs:
            A list containing enumerated compositions.
        """
        if fold<self.N_sps:
            raise ValueError("Number of enumeration folds smaller than number of species.")
        #Implemented in utils.enum_utils
        partitions = enumerate_partitions(n_part=self.N_sps,enum_fold=fold,constrs=self.constraints)
        enum_comps = [{sp:x for x,sp in zip(p,self.species)} for p in partitions]
        return enum_comps

class ExpansionStructure(MSONable)
    def __init__(self,lattice,an_sublats,ca_sublats=None,)
