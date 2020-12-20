__author__ = 'Fengyu Xie'

"""
Featurization module. Turns structures into feature vectors, and extracts the scalar properties to expand.
By default, if no other proerties are specified, will only expand normalized energies.
"""

import pandas as pd
import numpy as np
import os
from collections import OrderedDict
from monty.json import MSONable

from pymatgen import Structure,Lattice

from smol.cofe.space.domain import get_allowed_species
from smol.cofe import ClusterSubspace,ClusterExpansion
from smol.cofe.extern.ewald import EwaldTerm

from .assign import *

#### Feature assigners

#### Property extractors


class Featurizer(MSONable):
    """
    Featurization of calculation results.
    Attributes:
        prim(Structure):
            primitive cell of the structure to do cluster expansion on.
        sublat_list(List of lists on ints):
            Stores primitive cell indices of sites in the same sublattices. If none, sublattices will be automatically generated.
        basis_type(str):
            Type of basis function. By default, will use indicator basis.
        radius(Dict):
            Cluster radius. If None given, by default, given minimum interatomic distance d, will set pair radius to 4*d, triplets 
            and quads to 2*d.
        previous_ce(smol.cofe.ClusterExpansion):
            A previous cluster expansion. By default, is None. If this is given, will featurize based on this clusterexpansion
            indstead.
        assignments(List of str):
            Assigners called before mapping into feature vectors. For example, if we do cluster expansion with charge, since vasp
            calculated structures does not mark charges, we have to assign charges to atoms before mapping.

            All items in this list must be a class name in .assign.assignments. If multiple assignment class names are given,
            assignments will be done in the order of this list. If None given, will infer required assignments from the form of given 
            prim.

            Currently, we only support mixture of gaussian charge assignment from magnetization. You can implement you own assignments
            in .assign.assignments, and add local processing methods at the head of this file, accordingly.
        other_exp_props(List of str):
            Calculated properties to extract for expansion. Currently none of other proerties than 'e_prim' is supported. You can add
            your own properties extractors on you own.
    """
    allowed_assignments = {'charge':['MagChargeAssignment']}
    allowed_other_properties = []   

    def __init__(self,prim,sublat_list=None,basis_type='indicator',radius=None,
                 previous_ce=None,
                 assignments=None,
                 other_props=[]):

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
            self.basis_type = self.ce.cluster_subspace.orbits[0].basis_type
        else:
            #An empty cluster expansion with the points and ewald term only
            #Default is indicator basis
            if radius is not None and len(radius)>0:
                self.radius = radius
            else:
                d_nns = []
                for i,site1 in enumerate(self.prim):
                    d_ij = []
                    for j,site2 in enumerate(self.prim):
                        if j<i: continue;
                        if j>i:
                            d_ij.append(site1.distance(site2))
                        if j==i:
                            d_ij.append(min([self.prim.lattice.a,self.prim.lattice.b,self.prim.lattice.c]))
                    d_nns.append(min(d_ij))
                d_nn = min(d_nns)
    
                self.radius= {}
                # Default cluster radius
                self.radius[2]=d_nn*4.0
                self.radius[3]=d_nn*2.0
                self.radius[4]=d_nn*2.0

            c_spc = ClusterSubspace.from_radii(self.prim,self.radius,\
                                    basis = self.basis_type)

            if self.is_charged_ce:
                c_spc.add_external_term(EwaldTerm())
                coef = np.zeros(c_spc.n_bit_orderings+1)
                coef[-1] = 1.0
            else:
                coef = np.zeros(c_spc.n_bit_orderings)

            #This CE is used for featurization only, so values of coefficients don't matter. 
            self.ce = ClusterExpansion(c_spc,coef,[])

        #Handling assignment types.
        if assignments is not None:
            self.assignments = OrderedDict({})
            for assign_name in assignments:
                assign_type = None
                for tp in allowed_assignments:
                    if assign_name in allowed_assignments[tp]:
                        assign_type = tp
                if assign_type is None:
                    raise ValueError('Assignment {} is not implemented.'.format(assign_name))
                elif assign_type not in self.assignments:
                    self.assignments[assign_type] = assign_name
                else:
                    print('Warning: Assignment {} if of type {}, which is already scheduled.'\
                          .format(assign_name,assign_type))
        else:
        #Currently only supports charge assignments. If you implement more assginments in the future, please 
        #modify the following inference conditions, as well.
            self.assignments = OrderedDict({})
            if self.is_charged_ce:
                self.assignments['charge']='MagChargeAssignment'
            #### POTENTIAL ADDITIONS HERE

        #handling properties 
        self.other_props = []      
        for p_name in other_props:
            if p_name not in allowed_other_properties:
                raise ValueError('Property name: {} not supported!'.format(p_name))
            elif p_name not in self.other_props:
                self.other_props.append(p_name)
            else:
                continue

    def featurize(self,sc_table,comp_table,fact_table,calc_manager):
        """
        Load and featurize the fact table with vasp data.
        sc_table(pd.DataFrame):
            supercell matrix dimension table file.
        comp_table(pd.DataFrame):
            compositions dimension table file.
        fact_table(pd.DataFrame):
            Fact table, containing all structure information, and is
            to be filled.
        calc_manager(CalcManager):
            A calculations manager object. Either interacts with a local
            directory, vasp_run, or with a mongodb.
            Must provide methods to access calculated properties from 
            CONTCAR, vasprun.xml and OUTCAR. (They will also interact
            with computational resources.)
        """
        ##Loading and assignment

        ##Feature mapping

    def get_properties(self,vasp_run='vasp_run'):
        """
        """

    def as_dict(self):

    @classmethod
    def from_dict(cls,d):
