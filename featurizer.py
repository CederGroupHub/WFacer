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

from pymatgen import Structure,Lattice,Element,Specie,DummySpecie,Species
from pymatgen.core.periodic_table import get_el_sp

from smol.cofe.space.domain import get_allowed_species,Vacancy
from smol.cofe import ClusterSubspace,ClusterExpansion
from smol.cofe.extern.ewald import EwaldTerm

from .decorator import *
from .utils.format_utils import decode_from_dict

def decorate_single(s,decor_keys,decor_values):
    """
    This function decorates a single structure composed of pymatgen.Element in to structure of 
    pymatgen.Species or DummySpecies. Vacancies not considered.
    Inputs:
        s(pymatgen.Structure):
            Structure to be decorated.
        decor_keys(list of str):
            Names of properties to be decorated onto the structures
        decor_values(2D list, second dimension can be None):
            Values of properties to be assigned to each site. Shaped in:
            N_properties* N_sites
            Charges will be stored in each specie.oxidation_state, while 
            other properties will be stoered in specie._properties, if allowed by Species
            If any of the properties in the second dimension is None, will
            return None. (Decoration failed.)
    Return(Pymatgen.Structure of DummySpecies/Species or None):
        Decorated structure.
    """
    for val in decor_values:
        if val is None:
            return None
    
    #transpose to N_sites*N_properties
    decors_by_sites = list(zip(*decor_values))
    species_new = []

    for sp,decors_of_site in zip(s.species,decor_by_sites):
        try:
            sp_new = Specie(sp.symbol)
        except:
            sp_new = DummySpecie(sp.symbol)
        
        other_props = {}
        for key,val in zip(decor_keys,decors_of_site):
            if key == 'charge':
                sp_new._oxi_state = val
            else: #other properties
                if key in Species.supported_properties:
                    other_props[key] = val
                else:
                    raise ValueError("{} is not a supported pymatgen.Species property.".format(key))
        sp_new._properties = other_props
        species_new.append(sp_new)
    
    return Structure(s.lattice,species_new,s.frac_coords)
                

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
        decorators(List of .decorator.Decorator objects):
            Decorators names called before mapping into feature vectors. For example, if we do cluster expansion with charge, since vasp
            calculated structures does not mark charges, we have to assign charges to atoms before mapping.

            All items in this list must be a name in .decorator. If multiple decorators are given,
            decorations will be done in the order of this list. If None given, will check with prim, and see whether decorations
            are needed. If decorations are needed, but no decorator is given, will return an error. If multiple decorators are given
            on the same decoration type, for example, charge decoration by magnetization or bader charge, only the first one in list
            will be keeped. This duplication is not checked before model training and assignment, so you must check them on your own
            to avoid additional training cost.

            Currently, we only support mixture of gaussian charge decoration from magnetization. You can implement you own decoration
            in .decoration module, and add local processing methods at the head of this file, accordingly.

        other_props(List of str):
            Calculated properties to extract for expansion. Currently none of other proerties than 'e_prim' is supported. You can add
            your own properties extractors in calc_manager classes.
            This class does not check whether a proerty name is legal. Error messages will be given by calc_manager class.
            Check CEAuto.calc_manager docs for detail.
    """
    #Add to this dict if you implement more properties assignments
    decorator_requirements = {'MagChargeDecorator':['magnetization']}

    def __init__(self,prim,sublat_list=None,basis_type='indicator',radius=None,
                 previous_ce=None,
                 decorators=[],
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

            c_spc = ClusterSubspace.from_cutoffs(self.prim,self.radius,\
                                    basis = self.basis_type)

            if self.is_charged_ce:
                c_spc.add_external_term(EwaldTerm())
                coef = np.zeros(c_spc.num_corr_functions+1)
                coef[-1] = 1.0
            else:
                coef = np.zeros(c_spc.num_corr_functions)

            #This CE is used for featurization only, so values of coefficients don't matter. 
            self.ce = ClusterExpansion(c_spc,coef,np.array([coef.tolist()]))

        #Handling assignment types.
        if len(decorators)!=0:
            self.decorators = decorators
        else:
        #Currently only supports charge assignments. If you implement more assginments in the future, please 
        #modify the following inference conditions, as well.
            for sl_bits in self.bits:
                for b in sl_bits:
                    if not isinstance(b,(Vacancy,Element)):
                        raise ValueError('Cluster expasion requires Species, not Elements, \
                                          but no decorations are given!'\
                                          .format(list(b._properties.keys())))                       

        self.other_props = other_props

    def featurize(self,sc_table,fact_table,calc_manager):
        """
        Load and featurize the fact table with vasp data.
        sc_table(pd.DataFrame):
            supercell matrix dimension table file.
        comp_table(pd.DataFrame):
            compositions dimension table file.
        fact_table(pd.DataFrame):
            Fact table, containing all structure information, and is
            to be filled.
        calc_manager(CEAuto.CalcManager):
            A calculations manager object. Either interacts with a local
            directory, vasp_run, or with a mongodb.
            Must provide methods to access calculated properties from 
            CONTCAR, vasprun.xml and OUTCAR. (They will also interact
            with computational resources.)
        These four attibutes must be generated by the same CE flow.
        Return:
            featurized fact table
        """
        ##Loading and decoration. If decorators not trained, train decorator.
        eid_unassigned = fact_table[fact_table.calc_status=='CL'].entry_id
        #Check computation status, returns converged and failed indices.
        success_ids, fail_ids = calc_manager.check_computation_status(entry_ids = eid_unassigned)
        print('****{}/{} successful computations in the last run.'\
              .format(len(success_ids),len(fact_unassigned)))
        fact_table.loc[fact_table.entry_id.isin(fail_ids),'calc_status'] = 'CF'

        fact_unassigned = fact_table[fact_table.calc_status=='CL']\
                          .merge(sc_table,how='left',on='sc_id')

        #Loading structures
        structures_unassign = calc_manager.load_structures(\
                              entry_ids = fact_unassigned.entry_id)
       
        #Loading properties and doing decorations
        if len(self.decorators)>0:
            decorations = {}
            for decorator in self.decorators:
                d_name = decorator.__class__.__name__
                requirements = decorator_requirements[d_name]
                decor_inputs = calc_manager.load_properties(entry_ids = fact_unassigned.entry_id,\
                                                            prop_names = requirements,
                                                            include_pnames = False)
                if not decorator.trained:
                    print('******Training decorator {}.'.format(d_name))
                    decorator.train(structures_unassign,decor_inputs)
                
                decoration = decorator.assign(structures_unassign,properties)
                for prop_name,vals in decoration.items():
                    #Duplicacy removed here!
                    if prop_name not in decorations:
                        decorations[prop_name]=vals
    
            decor_keys = list(decorations.keys())
            decor_by_structures = list(zip(*list(decoration.values())))
            
            sid_assign_fails = []
            structures_unmaped = []
            for sid,(s_unassign,decors_in_str) in enumerate(zip(structures_unassign,decor_by_structures)):
                s_assign = decorate_single(s_unassign,decor_keys,decors_in_str)
                if s_assign is not None:
                    structures_unmaped.append(s_assign)
                else:
                    sid_assign_fails.append(sid)
            
            eid_assign_fails = fact_unassigned.iloc[sid_assign_fails].entry_id
            fact_table.loc[fact_table.entry_id.isin(eid_assign_fails),'calc_status'] = 'AF'
        else:
            structures_unmaped = structures_unassign

        fact_unmaped = fact_table[fact_table.calc_status=='CL']\
                          .merge(sc_table,how='left',on='sc_id')

        print('****{}/{} successful decorations in the last run.'\
              .format(len(fact_unmaped),len(fact_unassigned)))

        ##Feature mapping
        sid_map_fails = []
        occus_mapped = []
        corrs_mapped = []
        for sid,(s_unmap,mat) in enumerate(zip(structures_unmaped,fact_unmaped.matrix)):
            try:
                #First do a deformation to match lattice.
                sc = self.prim.copy()
                sc.make_supercell(mat)
                sc_lat = sc.lattice
                s_mod = Structure(sc_lat,s_unmap.species,s_unmap.frac_coords)
                #occupancies must be concoded
                occu = self.ce.cluster_subspace.\
                       occupancy_from_structure(s_mod,scmatrix=mat,encode=True)
                occu = list(occu)
                corr = self.ce.cluster_subspace.\
                       corr_from_structure(s_mod,scmatrix=mat)
                corr = list(corr)
                occus_mapped.append(occu)
                corrs_mapped.append(corr)
            except:
                sid_map_fails.append(sid)

        eid_map_fails = fact_unmaped.iloc[sid_map_fails].entry_id
        fact_table.loc[fact_table.entry_id.isin(eid_map_fails),'calc_status'] = 'MF'
        fact_table.loc[fact_table.calc_status=='CL','map_occu'] = occus_mapped
        fact_table.loc[fact_table.calc_status=='CL','map_corr'] = corrs_mapped
        fact_table.loc[fact_table.calc_status=='CL','calc_status'] = 'SC'

        print('****{}/{} successful mappings in the last run.'\
              .format(len(occus_mapped),len(fact_unmaped)))
        print('**Featurization finished.')

        return fact_table

    def get_properties(self,sc_table,fact_table,calc_manager):
        """
        Load expansion properties. By default, only loads energies.
        Properties will be noralized to per prim, if possible.
   
        All properties must be scalars.
        sc_table(pd.DataFrame):
            supercell matrix dimension table file.
        comp_table(pd.DataFrame):
            compositions dimension table file.
        fact_table(pd.DataFrame):
            Fact table, containing all structure information, and is
            to be filled.
        calc_manager(CEAuto.CalcManager):
            A calculations manager object. Either interacts with a local
            directory, vasp_run, or with a mongodb.
            Must provide methods to access calculated properties from 
            CONTCAR, vasprun.xml and OUTCAR. (They will also interact
            with computational resources.)
        Return:
            fact table with properties retrieved.
        NOTE: This must always be called after featurization!
        """
        fact_unchecked = fact_table[(fact_table.calc_status=='SC') & \
                                    (fact_table.e_prim.isna())]\
                         .merge(sc_table,how='left',on='sc_id')
        sc_sizes = fact_unchecked.matrix.map(lambda x: int(abs(round(np.linalg.det(x)))) )
        eid_unchecked = fact_unchecked.entry_id

        #loading un-normalized energies
        e_norms = calc_manager.load_properties(entry_ids = eid_unchecked,\
                                               normalize_by = sc_sizes,\
                                               prop_names = 'energy')
        #prop_names can be either one str or List. This method also provides normalization.
        #If not normalizable, will not normalize
        #Also provides a selection of format. If include_pnames = True, will return a 
        #Dictionary with {prop_name:[List of values]}

        other_props = calc_manager.oad_properties(entry_ids = eid_unchecked,
                                                   normalize_by = sc_sizes,\
                                                   prop_names = self.other_props,\
                                                   include_pnames = True)

        fact_table.loc[fact_table.entry_id.isin(eid_unchecked),'e_prim'] = e_norms
        fact_table.loc[fact_table.entry_id.isin(eid_unchecked),'other_props'] = other_props
        
        return fact_table

    def as_dict(self):
        return {'prim':self.prim.as_dict(), #Structure serialization safely keeps species.properties while composition does not. Strange indeed
                'sublat_list';self.sublat_list,
                'ce':self.ce.as_dict(),
                'basis_type':self.basis_type,
                'radius':self.radius,
                'decorators':[d.as_dict() for d in self.decorators]           
                'other_props':self.other_props
                "@module":self.__class__.__module__
                "@class":self.__class__.__name__
               }

    @classmethod
    def from_dict(cls,d):
        return cls(prim = Structure(d['prim']),
                   previous_ce = ClusterExpansion.from_dict(d['ce']),
                   sublat_list = d['sublat_list'],
                   basis_type = d['basis_type'],
                   radius = d['radius'],
                   decorators = [decode_from_dict(dd) for dd in d['decorators']],
                   other_props = d['other_props'])
