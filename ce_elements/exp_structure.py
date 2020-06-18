#U=Integrate all comp analysis into here

import sys
import os
this_file_path = os.path.abspath(__file__)
this_file_dir = os.dirname(this_file_path)
parent_dir = os.dirname(this_file_dir)
sys.path.append(parent_dir)
from utils.specie_utils import *
from utils.enum_utils import *
from utils.comp_utils import *

from monty.json import MSONable

import numpy as np
from functools import reduce
from operators import and_,or_

class ExpansionStructure(MSONable)
    def __init__(self,lattice,prim_coords,bits,comp_limits=None\
                 sublat_merge_rule=None,anion_markings=None):
        """
        This class is a prototype used to generate cluster expansion sample structures.
        Also contains a socket to mapping methods that returns the occupation array.
        Inputs:
            lattice: 
                A pymatgen.Lattice object, defining the primitive cell vecs of the cluster expansion.
            bits:
                A list indicating occupying species on each sublattice. The species should be 
                in the form of utils.specie_utils.CESpecie.
                For example: [[ca,mg],[o]], where ca = CESpecie.from_string('Ca2+'), etc.

            Note: sublattices with only one possible occupation will not be expanded at all.
 
            prim_coords:
                The Fractional coordinates of each site in primitive cell.
            comp_limits:
                Limits of the atomic fractions of each specie. If specified, should have the same shape
                as bits. For example, when bits=[[ca,mg],[o]], you can have: 
                [[(0.0,0.1),(0.2,1.0)],[(0.0,1.0)]] limiting atomic fraction of ca within 0-10% while 
                limiting atomic fraction of mg within 20%~100%
                When doing cluster expansion with vacancies, we highly recommend you to control the max
                concentration of vacancies to avoid abrupt structure distortion.
              
            sublat_merge_rule:
                A list indicating which sites in prim_coords are considered as the same sublattice.
                For example: [[0,1],[2]] means 3 sites divided into 2 sublattices.
            markings:
                A list of booleans marking each sublattices as anion sublattice or not.
                Specifying anion sublattice will be critical in getting the anion framework matcher.
        """
        self.lattice = lattice

        self.bits = bits
        #check validity. Anions and cations can not be mixed in a same sublattice.
        for sl_id,sl_bits in enumerate(bits):
            for b in sl_bits[1:]:
                if b.oxidation_state*sl_bits[0].oxidation_state<0:
                    raise ValueError("Sublattice {} has mixed cation and anions!".format(sl_id))
        self.nbits = get_n_bits(bits)
        self.neutral_combs,self.operations = get_all_axis(bits)
        #A neutral composition to draw composition vector from
        self.init_comp = get_init_comp(bits)

        self.prim_frac_coords = prim_coords
        #Use fractional coordinates!
        self.prim_cart_coords = self.prim_frac_coords@self.lattice.matrix

        self.sublat_merge_rule = sublat_merge_rule
        N_sts_prim = len(self.prim_frac_coords)
        self.sublat_list = get_sublat_list(N_sts_prim,sc_size=1,sublat_merge_rule=\
                           self.sublat_merge_rule)
       
        if comp_limits is None:
            self.comp_limits = [[(0.0,1.0) for b in sl_bits] for sl_bits in self.bits]
        else:
            for sl_id,sl_bits in enumerate(self.bits):
                if len(comp_limits[sl_id])!=self.bits[sl_id]:
                    raise ValueError("Composition limits are given, but not in the same shape as bits.")
            self.comp_limits = comp_limits

        if markings is not None and len(markings)==len(sublats):
            self.markings = markings
        else:
            #Then we shall generate anion framework automatically
            print("Warning: Auto generating sublattice type markings.")
            self.markings = []
            for sl_bits in bits:
                is_anion = True
                for b in sl_bits:
                    if b.oxidation_state>0:
                        is_anion = False
                        break
                self.markings.append(is_anion)
        

    @classmethod
    def from_prim_struct(cls,prim,comp_limits=None,sublat_merge_rule=None,anionic_markings=None):
        """
        Initialize a cluster expansion with a pymatgen.structure that has partially occupied
        sites. Not very recommended because this may reduce your flexibility of defining
        sublattices, and also this can not merge molecular fragments into a single specie.
        Inputs:
            prim: disordered primitive cell
            comp_limits: same thing as in self.__init__
            sublat_merge_rule: specifies which sites are in the same sublattice
            anionic_markings: specifies whether a sublattice is considered anionic or not.

        Note:
            If you initalize a CE with this method, you shall expect the species be ordered
            in their specie string's dictionary order. This ordering rule also applys to 'Vac'
        """        
        sublat_list = get_sublat_list(len(prim),sc_size=1,sublat_merge_rule=sublat_merge_rule)

        bits = []
        for sl_site_ids in sublat_list:
            st_id = sl_site_ids[0]
            sl_bit_strings = [str(sp) for sp in prim[st_id].species.keys()]
            sl_bit_strings = sorted(sl_bit_strings)
            sl_species = [CESpecie.from_string(sp) for sp in sl_bit_strings]

            bits.append(sl_species)

        return cls(lattice=prim.lattice,prim_coords=prim.frac_coords,\
                   bits = bits,\
                   comp_limits=comp_limits,\
                   sublat_merge_rule=sublat_merge_rule,\
                   anionic_markings=anionic_markings)

    
