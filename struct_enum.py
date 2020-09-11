__author__="Fengyu Xie"

"""
This module implements a StructureEnumerator class for CE sampling.
"""
import numpy as np

from monty.json import MSONable

from smol.cofe.extern.ewald import EwaldTerm
from smol.cofe.configspace.clusterspace import ClusterSubspace
from smol.cofe.configspace.domain import get_allowed_species
from smol.cofe.expansion import ClusterExpansion
from smol.moca.processor import *

from .comp_space import CompSpace

class StructureEnumerator(MSONable):
    """
    Attributes:
        prim(Structure):
            primitive cell of the structure to do cluster expansion on
        sublat_list(List of lists):
            Stores primitive cell indices of sites in the same sublattices
            If none, sublattices will be automatically generated.
        previous_ce(ClusterExpansion):
            A cluster expansion containing information of cluster expansion
            in previously enumerated structures. Used when doing mc sampling.

    """

    def __init__(self,prim,sublat_list = None,\
                 previous_ce = None,\
                 max_natoms=200,\
                 comp_restrictions=None,\
                 enum_method='random',\
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

        #if enum_method = mc, processor will be used to sample structures.
        if previous_ce is not None:
            self.ce = previous_ce
        else:
            #An empty cluster expansion with ewald term only
            c_spc = ClusterSubspace.from_radii(self.prim,{2:0.01})
            ew_term = EwaldTerm()
            c_spc.add_external_term(ew_term)
            coef = np.zeros(c_spc.n_bit_orderings+1)
            coef[-1] = 1

            self.ce = ClusterExpansion(c_spc,coef,[])
            

