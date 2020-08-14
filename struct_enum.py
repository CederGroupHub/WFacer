__author__="Fengyu Xie"

"""
This module implements a StructureEnumerator class for CE sampling.
"""
from monty.json import MSONable

from .comp_space import CompSpace

class StructureEnumerator(MSONable):

    def __init__(self,lattice,frac_coords,site_spaces,\
                 max_natoms=200,prim_sublat_ids=None,\
                 enum_method='random',\
                 select_method=''):


    @classmethod
    def from_prim(cls):
