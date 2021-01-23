"""
A wrapper to convert options and potential history files into
a bunch of elements that other CEAuto modules will need.
"""
__author__ = "Fengyu Xie"

import itertools
import numpy as np

from monty.json import MSONable
from pymatgen import Structure,Element

from smol.cofe.space.domain import get_allowed_species,Vacancy
from smol.cofe.space.extern import *

from .comp_space import CompSpace

class InputsWrapper(MSONable):

    """
    This class wrapps options and history into objects required by 
    other modules. Can be saved and re-initialized from the save.
    Direct initialization is not recommended. You are supposed to 
    initialize it with auto_load().

    Args: 
        lat_data(Dict):
            Dictionary of deserialized objects, storing everthing
            about the expansion lattice, including 'bits', 'lattice',
            'frac_coords','prim', etc.
        options(Dict):
            other options used in modules.
        history(List[Dict]):
            history cluster expansions information. Same as history 
            option in other modules.
    """
    
    def __init__(self, lat_data, options={}, history=[]):

        #Parse Lattice data
        self.bits = lat_data.get('bits')
        self.lattice = lat_data.get('lattice')
        self.frac_coords = lat_data.get('frac_coords')
        self.sublat_list = str_data.get('sublat_list')
        self.prim = str_data.get('prim')

        if self.prim is None and (self.bits is None or \
           self.lattice is None or self.frac_coords is None or \
           self.sublat_list is None):
            raise ValueError("Lattice information not fully provided!")

        if self.prim is not None and (self.bits is None or \
           self.sublat_list is None):

            prim_bits = get_allowed_species(self.prim)
            if self.sublat_list is not None:
                #User define sublattices, which you may not need very often.
                self.bits = [prim_bits[sl[0]] for sl in self.sublat_list]
            else:
                #Automatic sublattices, same rule as smol.moca.Sublattice:
                #sites with the same compositioon are considered same sublattice.
                self.sublat_list = []
                self.bits = []
                for s_id,s_bits in enumerate(prim_bits):
                    if s_bits in bits:
                        s_bits_id = self.bits.index(s_bits)
                        self.sublat_list[s_bits_id].append(s_id)
                    else:
                        self.sublat_list.append([s_id])
                        self.bits.append(s_bits)
        self.sl_sizes = [len(sl) for sl in self.sublat_list]
        self._compspace = Compspace(self.bits,self.sl_sizes)


        self.is_charged_ce = False
        for sp in itertools.chain(*self.bits):
            if not isinstance(sp,(Vacancy,Element)) and \
               sp.oxi_state!=0:
                self.is_charged_ce = True
                break

        if self.prim is not None and (self.lattice is None or \
           self.frac_coords is None):
            self.lattice = self.prim.lattice
            self.frac_coords = self.prim.frac_coords

        #Modify prim to a charge neutral composition that has all the species
        #included. This will make a lot of things easier.
        if self.prim is None or self.prim.charge!=0:
            typical_comp = self._compspace.get_random_point_in_unit_spc\
                                          (form='composition')

            N_sites = sum(self.sl_sizes)
            prim_comps = [{} for i in range(N_sites)]
            for i in range(N_sites):
                for sl_id,sl in self.sublat_list:
                    if i in sl:
                        prim_comps[i] = typical_comp[sl_id]
                        break
                
            self.prim = Structure(self.lattice,prim_comps,self.frac_coords)

        #Parse cluster subspace
        self.radius = options.get('radius')
        if self.radius is None or len(self.radius)==0:
            d_nns = []
            for i,site1 in enumerate(self.prim):
                d_ij = []
                for j,site2 in enumerate(self.prim):
                    if j<i: continue;
                    if j>i:
                        d_ij.append(site1.distance(site2))
                    if j==i:
                        d_ij.append(min([self.prim.lattice.a,\
                                    self.prim.lattice.b,self.prim.lattice.c]))
                d_nns.append(min(d_ij))
            d_nn = min(d_nns)
    
            self.radius= {}
            # Default cluster radius
            self.radius[2]=d_nn*4.0
            self.radius[3]=d_nn*2.0
            self.radius[4]=d_nn*2.0

        self.subspace = ClusterSubspace.from_cutoffs(self.prim,self.radius,\
                                    basis = self.basis_type)

        if self.is_charged_ce:
            self.subspace.add_external_term(EwaldTerm())

        self.other_extern_types = options.get('other_extern_types',[])
        self.other_extern_args = options.get('other_extern_args',[])

        for name,args in zip(self.other_extern_types,self.other_extern_args):
            self.subspace.add_external_term(globals()[name](**args))

        #Parse other option

        #Parse history
